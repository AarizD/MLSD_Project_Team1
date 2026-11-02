# Runbook (D8)

The two most likely failure modes for this system, each as **symptom -> confirmation -> recovery**, plus Redis as a
third. This extends the runbook in `docs/ARCHITECTURE.md`. Commands assume you run them from the repo root with the stack
started by `docker compose` (add the same `-f ...` files you started it with, e.g. the load-test overlay).

---

## Failure mode 1: the model provider is down, slow, or rate-limiting us

**Why it is the most likely:** every run makes ~7 provider calls, and free tiers have low rate limits (brief, section 12).

**Symptoms**
- Runs finish with status `degraded` and route `human_review`; `escalation_reason` reads like
  `Model provider unavailable: provider HTTP 429 ... circuit open`.
- After the first few failures, new runs fail **fast** (about a second) instead of after the full retry/timeout budget.
- Retries and `degraded` outcomes climb in `python scripts/report.py --window 5m` (LLM retry rate, runs by status) and on Grafana.
- No confident answer is ever returned for these runs: a run that did not finish is not presented as one (A5).

**Confirm**
```bash
docker compose exec -T redis redis-cli hgetall cb:llm          # state open | half_open | closed, failure count
docker compose logs --tail=100 worker | grep -i -E "provider|circuit|429|503|timeout"
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $LLM_API_KEY" "$LLM_BASE_URL/models"   # is the provider itself reachable?
python scripts/report.py --window 5m
```
`cb:llm` is the **shared** circuit state. It lives in Redis because RQ forks a child process per job: an in-memory breaker would
be discarded after every job and each job would re-spend its whole retry budget on a provider already known to be down.

**Recover**
1. Fix the cause: renew/replace the key or quota, wait out the rate-limit window, or switch provider by changing
   `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` in `.env` and running `docker compose up -d --force-recreate worker`.
2. The breaker half-opens by itself after `CB_RESET_S` (default 30 s) and closes on the first successful call. To force it
   closed immediately once the provider is healthy: `docker compose exec -T redis redis-cli del cb:llm`.
3. Re-submit the claims that ended `degraded` (they were routed to a human, so nothing is lost, but they have no
   recommendation). Find them from the run ids your client kept, or `GET /runs/{id}`.
4. Do **not** add workers to "push through" a rate limit: more workers means more concurrent calls and more 429s.

**Demonstrate it (demo item 8):** with the load-test overlay running, `python scripts/demo_checks.py circuit` makes the stub
provider fail, shows the breaker opening, later claims failing fast, no confident answer returned, and recovery.

---

## Failure mode 2: queue backlog / workers saturated

**Why it is likely:** each worker runs one job at a time and each job is ~10 s of waiting on the model (see `docs/SLO.md`).
Throughput is `workers / run time`; beyond that, claims queue.

**Symptoms**
- `GET /runs/{id}` stays `queued` (or `running`) for a long time; end-to-end latency passes the SLO.
- Worker CPU is low while latency rises (they are waiting, not computing).

**Confirm**
```bash
docker compose exec -T redis redis-cli llen rq:queue:claims    # jobs waiting; steadily growing = backlog
docker compose ps                                              # how many worker containers are healthy
docker stats --no-stream                                       # confirms workers are idle, not CPU-bound
python scripts/report.py --window 5m                           # run latency, runs by status
```

**Recover**
1. Add capacity: `docker compose up -d --scale worker=6` (adjust). Check the provider's rate limit first (failure mode 1).
2. If one run is stuck: RQ stops a job that exceeds `JOB_TIMEOUT_S` (180 s). Check that run's status with `GET /runs/{id}`; if it never reached a terminal status, re-submit the claim.
3. If the cause is one client looping against the API: the per-client rate limit (`RATE_LIMIT_PER_MIN`) returns 429 to that
   client; confirm with `rate_limit_exceeded_total` in `python scripts/report.py`.
4. Longer term: let one worker run several jobs concurrently, or reject new work when the queue is deeper than ~2x workers
   (neither is implemented; see `docs/LOAD_TEST.md`).

---

## Failure mode 3: Redis is down

**Symptoms:** `GET /health` fails or reports the store unreachable; submissions return 5xx; workers log connection errors.

**Confirm:** `docker compose ps redis` and `docker compose logs --tail=50 redis`; `docker compose exec -T redis redis-cli ping`.

**Recover:** `docker compose restart redis`, then check `curl localhost:8000/health`. `docker-compose.yml` gives Redis no volume, so
anything not yet snapshotted is lost if the container is *recreated*: treat in-flight runs as lost and have claimants re-submit.
If the API or workers do not recover on their own, `docker compose restart api worker`. Also clear the shared breaker if it
was left open: `docker compose exec -T redis redis-cli del cb:llm`.

---

After any real incident, add an entry to `docs/INCIDENT_LOG.md`.
