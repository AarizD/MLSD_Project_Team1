# Service Level Objective for a complete run (D6)

**Status:** the targets in section 2 are proposed from design arithmetic (section 3), not from measurement. They become
your declared SLO when this file is **committed before your first load run**; the commit timestamp is the evidence the brief
asks for ("a target set after seeing the results does not satisfy this requirement"). After you measure, do not edit
section 2. If reality disagrees, say so in `docs/LOAD_TEST.md` and leave this file as declared.

## 1. What is measured

**SLI:** wall-clock time from the client sending `POST /claims` to the first poll of `GET /runs/{id}` that returns a
terminal status (`completed`, `escalated`, `degraded`, `failed`). Polling interval 0.5 s, so the SLI includes up to
0.5 s of polling granularity. This is submission to final result, not the submission call alone (the submit call is
a few ms and is reported separately, because it is the number that would flatter us).

Only **valid** claims count toward the latency SLI. Malformed claims are rejected in milliseconds at the boundary;
mixing them in would drag p95 down artificially (`LOAD_MALFORMED_PCT` defaults to 0 for this reason).

## 2. The objective

| | Target |
| --- | --- |
| **p95 end-to-end latency, valid claims** | **<= 30 s** |
| Failed or timed-out runs (status `failed`, or no terminal status within 120 s) | **< 1 %** |
| Load at which the above must hold | **up to 6 claims in flight, with 3 workers** (2 per worker) |

Stub LLM latency for the test: **1.5 s per call, +/-0.3 s uniform jitter** (`.env.loadtest.example`).

## 3. Why 30 s (justified against the brief's section 1)

Section 1 calls this pipeline *high-volume, latency-sensitive and correctness-critical.* It sits behind a claim form
or a call-centre screen: the claimant, or the agent on the phone, is waiting for the triage result.

* **A person is waiting.** 30 s is about the longest an interactive screen can show "working on it" before people
  abandon or resubmit. A resubmission is worse than slow: it is another multi-call agent run, so a latency failure turns
  directly into cost (the reason for rate limiting and caching in D7).
* **Headroom over the unloaded cost.** A happy-path run makes **7 LLM calls and 4 tool calls** (1 triage + 5 coverage
  turns + 1 verifier; visible as `llm_calls` and `tool_calls` in any run record). At 1.5 s per call the unloaded service
  time is **S = 7 x 1.5 = 10.5 s** (tool time is milliseconds). 30 s is 2.9x S: enough for one queued run ahead of you on
  a worker, not more.
* **Inside our own bounds.** Well under `RUN_DEADLINE_S=90` and `JOB_TIMEOUT_S=180`, so a breach is a latency problem,
  not a timeout.
* Correctness is not traded for speed: the verifier step is **inside** the measured path.

## 4. Hypotheses to test (from queueing arithmetic; confirm or refute them in `docs/LOAD_TEST.md`)

* Each RQ worker runs **one job at a time** and spends nearly all of it waiting on sequential model calls, so the
  **worker pool**, not CPU, should be the limiting resource.
* With `W` workers and `N` runs in flight the slowest run waits for `ceil(N/W)` service times:
  **p95 ~ ceil(N/W) x 10.5 s.** At W=3: N<=3 -> ~10.5 s; N=4..6 -> ~21 s; **N=7..9 -> ~31.5 s (SLO breached).**
* **Predicted breaking point: 7 claims in flight with 3 workers**, component responsible: **worker concurrency**.
* **Predicted scaling:** capacity = W / S = 0.095 runs/s per worker; throughput should scale ~linearly with
  `--scale worker=N` until another component saturates, and the breaking point should move to 2W+1 in flight.
* **Predicted non-bottlenecks:** API (submit in ms, CPU-light), Redis (a handful of ops per run), the stub (it sleeps).

If your measurements disagree (Docker CPU limits, a slower host, a different stub delay), the *disagreement* is the
finding: say which component saturated first and show the evidence (`docker stats`, queue depth, Grafana).

## 5. Method

`scripts/load_steps.py` (Locust underneath) step-loads concurrent users; each user submits a valid claim, polls to a
terminal status, records the end-to-end time, and repeats. A user that times out is **recorded as a failure** (a harness
that silently drops slow runs would hide exactly the breaches this SLO is about). Steps: 1, 3, 6, 7, 9, 12, 18 users;
each is held until at least 30 runs have completed after excluding the first `ceil(N/W)` as warm-up. Exact commands in
`docs/LOAD_TEST.md`.
