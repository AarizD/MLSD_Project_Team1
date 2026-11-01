#!/usr/bin/env bash
# Docker-free approximation of `docker compose up --scale worker=N` for machines/sandboxes without Docker:
# real Redis, the API, N RQ workers and the stub LLM service as separate processes. Same code and same queue
# semantics as the containers, but NO container CPU/memory limits and everything shares the host's cores.
#
#   scripts/local_stack.sh start 3      # N workers
#   scripts/local_stack.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=/tmp/claims_stack; mkdir -p "$RUN"
export STORE_BACKEND=redis QUEUE_BACKEND=rq REDIS_URL=redis://localhost:6379/0 RUNS_DIR="$RUN/runs" \
       LLM_PROVIDER=openai_compatible LLM_BASE_URL=http://localhost:9000/v1 LLM_API_KEY=stub-not-a-secret LLM_MODEL=stub-model \
       LLM_PRICE_IN_PER_1K=0.00015 LLM_PRICE_OUT_PER_1K=0.0006 STUB_LLM_DELAY_S=${STUB_LLM_DELAY_S:-1.5} \
       STUB_LLM_JITTER_S=${STUB_LLM_JITTER_S:-0.3} RATE_LIMIT_PER_MIN=0 LOG_LEVEL=WARNING OTEL_ENABLED=false
stop() {
  for f in "$RUN"/*.pid; do [ -e "$f" ] && kill "$(cat "$f")" 2>/dev/null || true; done
  rm -f "$RUN"/*.pid; sleep 1
}
case "${1:-}" in
  start)
    N=${2:-3}; stop
    redis-cli ping >/dev/null 2>&1 || redis-server --port 6379 --save "" --appendonly no --daemonize yes >/dev/null
    redis-cli flushall >/dev/null; rm -rf "$RUN/runs"
    setsid nohup python3 -m uvicorn stub_llm.app:app --port 9000 --log-level warning >"$RUN/stub.log" 2>&1 & echo $! >"$RUN/stub.pid"
    setsid nohup python3 -m uvicorn app.main:app --port 8000 --log-level warning >"$RUN/api.log" 2>&1 & echo $! >"$RUN/api.pid"
    for i in $(seq 1 "$N"); do
      WORKER_METRICS_PORT=$((9100 + i)) PROMETHEUS_MULTIPROC_DIR="$RUN/prom_$i" \
        setsid nohup python3 -m app.rq_worker >"$RUN/worker_$i.log" 2>&1 & echo $! >"$RUN/worker_$i.pid"
    done
    for i in $(seq 1 30); do curl -sf localhost:8000/health >/dev/null && curl -sf localhost:9000/health >/dev/null && break; sleep 1; done
    sleep 3; echo "up: $N worker(s); api :8000, stub :9000"; ;;
  stop) stop; echo stopped ;;
  *) echo "usage: $0 start [N_WORKERS] | stop"; exit 2 ;;
esac
