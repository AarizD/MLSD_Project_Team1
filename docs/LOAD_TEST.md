# Load test (D6, A7): how to run it and where your results go

This file contains **no results**. Run the commands below on your own machine; the scripts print the numbers and write
them to `docs/loadtest_results/`. Paste them into the tables at the bottom and into the report.

Before you start: `docs/SLO.md` must already be committed (the target has to exist before the measurement).

## 0. One-time setup

```bash
pip install -r requirements-loadtest.txt              # locust, psutil, httpx (on the host, not in a container)
cp .env.example .env                                  # the base stack reads this; the load test overrides the model with the stub
cp .env.loadtest.example .env.loadtest                # stub delay (1.5 s +/- 0.3 s), stub endpoint, reference prices
```

## 1. Start the stack with the stub in place of the model (A7: `--scale worker=3`)

```bash
docker compose --env-file .env.loadtest \
  -f docker-compose.yml -f docker-compose.loadtest.yml up --build --scale worker=3
```

Wait until `docker compose ps` shows every service `healthy` (api, redis, jaeger, llm-stub, 3 x worker). The overlay
forces the workers onto the stub regardless of what `.env` says, so this cannot spend money on a real provider.

Quick check: `curl localhost:8000/health` and `curl localhost:9000/health` (stub).

## 2. Sweep concurrency against the declared SLO (D6)

In a second terminal:

```bash
python scripts/load_steps.py --docker --workers 3 --label w3 \
  --compose-file docker-compose.yml --compose-file docker-compose.loadtest.yml
```

Default steps are 1, 3, 6, 7, 9, 12, 18 concurrent users, each held until >= 30 runs completed (warm-up excluded).
Expect roughly 15-20 minutes. For each step it prints one line while running, then a table at the end:

| column | meaning |
| --- | --- |
| In flight (users) | concurrent claims, each submitting and polling to completion |
| p50 / p95 s | submission -> terminal status (the SLI in `docs/SLO.md`), warm-up excluded |
| predicted p95 s | `ceil(N/W) x 10.5 s`, the hypothesis from `docs/SLO.md` section 4 |
| Throughput runs/s | completed runs per second over the step |
| Failures % | `failed` runs plus runs with no terminal status within 120 s |
| SLO met | p95 <= 30 s **and** failures < 1 % |
| worker / api / stub / redis CPU% | summed `docker stats` CPU over containers of that service (100% = one core) |
| queue depth avg/max | RQ `claims` queue length, read with `docker compose exec redis redis-cli llen` |

Files written: `docs/loadtest_results/w3_u<N>.json` (one per step), `w3_u<N>.csv` (one row per run), `w3_summary.md`.

**Reading it:**
- *The concurrency at which the target fails* = the first row with SLO met = NO.
- *The component responsible* = whichever resource is saturated at that row. If worker/api/redis/stub CPU is low, memory is
  flat, and the queue depth is growing, the limit is **worker slots** (hypothesis in `docs/SLO.md`). If a container sits
  near 100% CPU, that is your bottleneck: name it. You can also look at Grafana (http://localhost:3000, admin/admin).
- Throughput that stops rising while latency keeps rising is the signature of saturation.

If you see `WARNING: --workers 3 but [n] worker container(s) were running`, your `--scale` and `--workers` disagree; fix
one so the warm-up exclusion and the prediction column are right.

## 3. Effect of `--scale` (A7)

Re-run with a different worker count, restarting the stack each time (Ctrl-C, then up again with the new `--scale`):

```bash
# stack started with --scale worker=1 (then 2, then 6)
python scripts/load_steps.py --docker --workers 1 --users 9 --min-runs 20 --label scale_w1 \
  --compose-file docker-compose.yml --compose-file docker-compose.loadtest.yml
```

Compare throughput and p95 at the same `--users` across `scale_w1`, `scale_w2`, `w3`, `scale_w6`. If scaling does not
help, A7 asks you to identify the actual bottleneck: use the CPU columns and `docker stats` again. To see whether the
breaking point moves to `2W + 1` in flight, run W=6 with `--users 9 13`.

## 4. Other numbers you can generate from the same running stack

```bash
python scripts/report.py --window 1h          # success rate, steps/run, latency, retries, cost per 100 runs, drivers, uptime
python scripts/demo_checks.py all             # demo items 7 and 8 (guardrails, circuit breaker); prints PASS/FAIL
```

## 5. Notes that affect how you quote results

- The stub answers after a fixed delay, so this measures **your system**, not a provider. With a real provider the limit is
  usually the provider's rate limit: at 7 calls per run, R requests/min caps you at R/7 runs/min whatever the worker count.
- Locust gives every simulated user a fresh client id and policy number, so the rate limiter and the run cache are **not**
  exercised (that is deliberate; `scripts/benchmark_cache.py` measures caching).
- Run it on an otherwise idle machine; Docker Desktop's VM CPU/memory limits (Settings -> Resources) change the result.
- Without `--docker` the script samples host processes instead (for running the API and workers by hand).

## 6. Your results (fill in from `docs/loadtest_results/w3_summary.md`)

**Environment:** host CPU/RAM: ____ ; Docker resources: ____ ; date: ____ ; commit: ____

| In flight | Runs counted | p50 s | p95 s | predicted p95 s | Throughput runs/s | Failures % | SLO met | worker CPU% | api CPU% | queue depth avg/max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | | | | 10.5 | | | | | | |
| 3 | | | | 10.5 | | | | | | |
| 6 | | | | 21.0 | | | | | | |
| 7 | | | | 31.5 | | | | | | |
| 9 | | | | 31.5 | | | | | | |
| 12 | | | | 42.0 | | | | | | |
| 18 | | | | 63.0 | | | | | | |

**Verdict against the declared SLO:** ____ (met up to ____ in flight; fails at ____)
**Component responsible, with evidence:** ____

**Scaling (same load, different `--scale worker=N`):**

| Workers | In flight | Throughput runs/s | p95 s | SLO met |
| --- | --- | --- | --- | --- |
| 1 | 9 | | | |
| 2 | 9 | | | |
| 3 | 9 | | | |
| 6 | 9 | | | |

**Did my hypotheses in `docs/SLO.md` section 4 hold?** ____
