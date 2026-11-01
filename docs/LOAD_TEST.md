# Load test results (D6, A7)

Objective, SLI and **predictions** were fixed in `docs/SLO.md` (commit `5b30dbc`) before any measurement. This file
reports the measurements against them. Raw per-run data: `docs/loadtest_results/*.csv`; per-step JSON alongside.

## Verdict against the declared SLO (p95 <= 30 s, <1 % failures, valid claims, 3 workers)

* **Met up to 6 claims in flight** (p95 22.2 s). **Fails at 7 in flight** (p95 31.7 s), exactly where `docs/SLO.md`
  predicted (31.5 s). 0 % failed or timed-out runs at every step; the SLO fails on latency, not errors.
* **Component responsible: worker slots** (one job per worker, each blocked on sequential model calls).

## Method and environment (read before quoting any number)

Locust 2.47, `scripts/load_steps.py`, stub LLM service at 1.5 s +/- 0.3 s per call, 7 LLM calls per run. Measured
end to end: `POST /claims` to first poll returning a terminal status (0.5 s poll interval included). Warm-up
`ceil(N/W)` runs excluded and >= 30 counted per step, as declared. Slow runs are recorded as failures, never dropped.

**Not Docker.** The sandbox had no Docker, so the stack ran as real processes (`scripts/local_stack.sh`: Redis 7.0.15, the
API, N RQ workers, the stub service) on **1 vCPU / 4 GB** with no container limits. Same code and queue semantics as
the containers, but CPU-bound limits and per-container resource numbers do **not** transfer. The I/O-bound queueing
result does. Re-run with Docker before the demo:
`docker compose --env-file .env.loadtest -f docker-compose.yml -f docker-compose.loadtest.yml up --build --scale worker=3`
then `python scripts/load_steps.py --workers 3 --label w3` (Redis is not published by the compose file, so queue-depth
and per-process CPU sampling will be empty; use Grafana / `docker stats`).

A first attempt at the sweep was lost when the sandbox restarted mid-step; it was discarded and the whole sweep re-run
from scratch. No partial data is used here.

## Sweep: 3 workers
| In flight (users) | Runs counted | p50 s | p95 s | predicted p95 s | Throughput runs/s | Failures % | SLO met | worker CPU% | api CPU% | stub CPU% | redis CPU% | queue depth avg/max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 31 | 10.6 | 11.6 | 10.5 | 0.09 | 0.0 | yes | 0.3 | 0.4 | 0.2 | 0.1 | 0.0/0 |
| 3 | 33 | 10.6 | 11.6 | 10.5 | 0.2658 | 0.0 | yes | 0.6 | 0.6 | 0.2 | 0.1 | 0.0/0 |
| 6 | 36 | 21.1 | 22.2 | 21.0 | 0.2731 | 0.0 | yes | 0.8 | 1.1 | 0.2 | 0.2 | 2.23/3 |
| 7 | 37 | 22.2 | 31.7 | 31.5 | 0.28 | 0.0 | **NO** | 0.8 | 1.3 | 0.3 | 0.2 | 3.0/4 |
| 9 | 39 | 31.7 | 32.9 | 31.5 | 0.2902 | 0.0 | **NO** | 0.7 | 1.7 | 0.3 | 0.2 | 4.49/6 |
| 12 | 43 | 42.9 | 43.7 | 42.0 | 0.2782 | 0.0 | **NO** | 0.6 | 1.7 | 0.2 | 0.2 | 6.69/9 |
| 18 | 48 | 63.6 | 65.0 | 63.0 | 0.2882 | 0.0 | **NO** | 0.6 | 2.5 | 0.3 | 0.2 | 10.99/15 |

`predicted p95` = `ceil(N/W) x 10.5 s` from `docs/SLO.md`. Measured is within ~2 s at every step.

## Bottleneck evidence

1. **Throughput plateaus at ~0.28 runs/s regardless of load** = 3 workers / 10.6 s. Adding users beyond 3 only
   lengthens the queue (average depth rises 2.2 -> 11.0).
2. **Nothing else is busy.** At the worst step every component is below 3 % CPU (api 2.5 %, workers 0.6 %, stub 0.3 %,
   redis 0.2 %).
3. **Workers are idle, not slow:** a direct measurement showed 16 ms CPU in a 10.4 s run (0.2 %).
4. Predicted and measured p95 agree at every step, which a different bottleneck (CPU, Redis, API) would not produce.

## Scaling (A7): partial, honestly

Only the **1-worker** step finished before the sandbox restarted: 9 in flight, p50 96.8 s, p95 98.8 s,
throughput 0.0931 runs/s (predicted 94.5 s and 0.095/s; SLO not met, 0 % failures). Compared with
the 3-worker sweep at the same load (p95 32.9 s, 0.29 runs/s), tripling workers tripled throughput and cut p95 about 3x,
consistent with linear scaling. **The 2- and 6-worker steps did not run, so scaling beyond that, and the predicted
breaking point at 2W+1, is untested.** Re-run: `scripts/local_stack.sh start 6` then
`python scripts/load_steps.py --workers 6 --users 9 13 --min-runs 20 --label scale_w6`.

## What this means

* Each worker is >99 % idle, waiting on the model. Throughput is bought with worker count (about 90 MB each here), or by
  letting one worker run several jobs concurrently (threads/async). Neither is implemented.
* Admission control (reject or defer when queue depth exceeds `2W`) would keep accepted runs inside the SLO under
  overload, and cap spend. Not implemented.
* **With a real provider the limit will likely move to the provider's rate limit:** at 7 calls per run, a limit of R
  requests/min caps throughput at R/7 runs/min whatever the worker count. That is why the stub is used here, and why
  this result says nothing about real-provider latency (heavier tails than the stub's +/-0.3 s jitter).
