# Service Level Objective for a complete run (D6)

**Declared:** 2026-11-01T03:31:32Z, *before any load test was run.* The commit that adds this file is the evidence;
no Locust run, no multi-user measurement and no scaling measurement existed when it was written. Only a 3-claim smoke test
(correctness, not timing) had been executed. Do not edit the numbers in section 2 after measuring; if reality
disagrees, record that in `docs/LOAD_TEST.md` and leave this file as declared.

## 1. What is measured

**SLI:** wall-clock time from the client sending `POST /claims` to the first poll of `GET /runs/{id}` that returns a
terminal status (`completed`, `escalated`, `degraded`, `failed`). Polling interval 0.5 s, so the SLI includes up to
0.5 s of polling granularity. This is submission to final result, not the submission call alone (the submit call is
~5 ms and is reported separately, because it is the number that would flatter us).

Only **valid** claims count toward the latency SLI. Malformed claims are rejected in ~ms at the boundary and are
reported separately; mixing them in would drag p95 down artificially.

## 2. The objective (fixed)

| | Target |
| --- | --- |
| **p95 end-to-end latency, valid claims** | **<= 30 s** |
| Failed or timed-out runs (status `failed`, or no terminal status within 120 s) | **< 1 %** |
| Load at which the above must hold | **up to 6 claims in flight, with 3 workers** (2 per worker) |

Stub LLM latency used for the test: **1.5 s per call, +/-0.3 s uniform jitter** (fixed here, before measuring).

## 3. Why 30 s (justified against the brief's section 1)

Section 1 calls this pipeline *high-volume, latency-sensitive and correctness-critical.* It sits behind a claim form
or a call-centre screen: the claimant or the agent on the phone is waiting for the triage result. The target comes from
that, not from what we expect to achieve:

* **A person is waiting.** 30 s is about the longest an interactive screen can show "working on it" before people
  abandon or resubmit. A resubmission is worse than slow: it is another multi-call agent run, so latency failure
  turns directly into cost (see the rate-limit and cache rationale in D7).
* **Headroom over the unloaded cost.** A happy-path run makes **7 LLM calls and 4 tool calls** (verified by
  counting the stub's script, no load involved: 1 triage + 5 coverage turns + 1 verifier). At 1.5 s per call the
  unloaded service time is **S = 7 x 1.5 = 10.5 s** (tool time is milliseconds). 30 s is 2.9x S: enough for one
  queued run ahead of you on a worker, not more.
* **Inside our own bounds.** Well under `RUN_DEADLINE_S=90` and `JOB_TIMEOUT_S=180`, so the SLO is stricter than the
  failure bounds and a breach is a latency problem, not a timeout.
* Correctness is not traded for speed: the verifier step is **inside** the measured path. Skipping it to hit the
  target would be a different (non-compliant) system, not a faster one.

## 4. Predictions recorded before measuring

These are hypotheses from design arithmetic (queueing on a fixed worker pool), written down so the test can refute them.

* Each RQ worker runs **one job at a time** and spends nearly all of it waiting on sequential LLM calls (CPU idle).
  So the worker pool, not CPU, should be the limiting resource.
* With `W` workers and `N` runs in flight, the slowest run waits for `ceil(N/W)` service times:
  **p95 ~ ceil(N/W) x 10.5 s.** At W=3: N<=3 -> ~10.5 s; N=4..6 -> ~21 s; **N=7..9 -> ~31.5 s (SLO breached).**
* **Predicted breaking point: 7 claims in flight with 3 workers.** Component responsible: **worker concurrency**
  (jobs are processed one-per-worker, and each is I/O-bound on the model), not the API, Redis or the stub.
* **Predicted scaling:** capacity = W / S = 0.095 runs/s per worker, so 1 worker ~ 0.095/s, 3 workers ~ 0.29/s (~17
  runs/min); throughput should scale ~linearly with `--scale worker` until a different component saturates, and the
  breaking point should move to 2W+1 in flight.
* **Predicted non-bottlenecks:** API (submit ~ms, CPU-light), Redis (a handful of ops per run), stub (sleeps).

## 5. Method (so it is repeatable)

Locust (`scripts/locustfile.py`) step-loads concurrent users; each user submits a valid claim, polls to a terminal
status, records the end-to-end time as a separate `claim_full_lifecycle` metric, and repeats. A user that times out is
**recorded as a failure** (a harness that silently drops slow runs would hide exactly the breaches this SLO is about).
Concurrency steps: 1, 3, 6, 7, 9, 12, 18. Each step is held until at least 30 runs have completed; the first
`ceil(N/W)` run times are warm-up and are excluded from p95. Results, bottleneck evidence and the verdict against this
table go in `docs/LOAD_TEST.md`.
