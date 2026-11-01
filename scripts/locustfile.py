"""Locust load test for Motor Insurance Claim Intake & Triage (D6, A7).

Measures the SLI declared in docs/SLO.md: wall-clock time from POST /claims to the first poll that returns a terminal
status. Recorded as the custom metric `claim_full_lifecycle` (request_type E2E) and, per run, as a CSV row.

Design rules (each fixes a way a load test can quietly lie):
  * A run that does not finish within LOAD_POLL_TIMEOUT_S is RECORDED AS A FAILURE with its elapsed time. The earlier
    version returned silently, so the slowest runs, i.e. the SLO breaches, disappeared from the statistics.
  * A run that ends `failed` counts as a failure even though HTTP was 200.
  * Valid-claim latency is kept apart from malformed-claim traffic (LOAD_MALFORMED_PCT, default 0), because instant
    422s would drag percentiles down and change how many runs are really in flight.
  * Every user gets a unique client id and policy number, so neither the rate limiter (D7) nor the run cache is hit:
    this measures the agents, which is what the SLO is about. (Cache effect is measured by scripts/benchmark_cache.py.)

Env:
  LOAD_RUNS_CSV          append one row per finished run here (used by scripts/load_steps.py)
  LOAD_STOP_FILE         when this file exists, users finish their current run and submit no more (clean shutdown)
  LOAD_POLL_TIMEOUT_S    default 120
  LOAD_POLL_INTERVAL_S   default 0.5 (included in the SLI as polling granularity)
  LOAD_MALFORMED_PCT     default 0
  LOAD_DESCRIPTION_MIX   "plain" (default) | "mixed" (adds the section 4.1 delivery case and theft)
"""

import csv
import datetime as dt
import os
import random
import threading
import time
import uuid

from locust import HttpUser, between, task

RUNS_CSV = os.getenv("LOAD_RUNS_CSV")
STOP_FILE = os.getenv("LOAD_STOP_FILE")
POLL_TIMEOUT_S = float(os.getenv("LOAD_POLL_TIMEOUT_S", "120"))
POLL_INTERVAL_S = float(os.getenv("LOAD_POLL_INTERVAL_S", "0.5"))
MALFORMED_PCT = float(os.getenv("LOAD_MALFORMED_PCT", "0"))
MIX = os.getenv("LOAD_DESCRIPTION_MIX", "plain")
TERMINAL = {"completed", "escalated", "failed", "degraded"}
_csv_lock = threading.Lock()

DESCRIPTIONS = {
    "plain": ["Front bumper cracked after hitting a bollard while parking."],
    "mixed": ["Front bumper cracked after hitting a bollard while parking.",
              "I was making a food delivery run in my own car when the accident happened.",
              "My car was stolen from outside my house last night."],
}


def _record_row(row: dict):
    if not RUNS_CSV:
        return
    with _csv_lock:
        new = not os.path.exists(RUNS_CSV)
        with open(RUNS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_end", "e2e_ms", "submit_ms", "status", "timed_out", "run_id", "users"])
            if new:
                w.writeheader()
            w.writerow(row)


def valid_payload():
    today, uid = dt.date.today(), uuid.uuid4().hex[:8]
    return uid, {
        "policy_number": f"POL-LOCUST-{uid}", "insured_name": f"Locust User {uid}", "vehicle_registration": "KA-01-AB-1234",
        "channel": "app", "date_of_loss": str(today - dt.timedelta(days=1)), "date_of_report": str(today),
        "place_of_loss": "Bangalore", "estimated_loss_inr": 25000.0, "description": random.choice(DESCRIPTIONS[MIX]),
        "garage_name": "Rapid Care Motors", "cashless_requested": True, "third_party_involved": False,
        "injuries": False, "police_report_filed": False,
    }


class ClaimIntakeUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def one_claim(self):
        if STOP_FILE and os.path.exists(STOP_FILE):
            time.sleep(1)               # drain: finish nothing new, let in-flight runs complete
            return
        if MALFORMED_PCT and random.random() * 100 < MALFORMED_PCT:
            return self.malformed()
        uid, payload = valid_payload()
        headers = {"Content-Type": "application/json", "X-Client-Id": f"client_{uid}"}
        t0 = time.perf_counter()
        with self.client.post("/claims", json=payload, headers=headers, name="POST /claims [valid]",
                              catch_response=True) as resp:
            submit_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 202:
                resp.failure(f"submit failed: {resp.status_code}")
                return
            resp.success()
            poll_url = resp.json()["poll_url"]

        if RUNS_CSV:   # one line per submission: the runner drains until finished rows == submitted lines
            with _csv_lock, open(RUNS_CSV + ".starts", "a") as sf:
                sf.write(f"{time.time()}\n")
        status, timed_out, run_id = None, False, poll_url.rsplit("/", 1)[-1]
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed > POLL_TIMEOUT_S:
                timed_out = True
                break
            time.sleep(POLL_INTERVAL_S)
            with self.client.get(poll_url, name="GET /runs/{id} [poll]", catch_response=True) as p:
                if p.status_code != 200:
                    p.failure(f"poll returned {p.status_code}")
                    status = f"poll_{p.status_code}"
                    break
                p.success()
                status = p.json().get("status")
                if status in TERMINAL:
                    break

        e2e_ms = (time.perf_counter() - t0) * 1000
        failed = timed_out or status == "failed" or (status or "").startswith("poll_")
        exc = None
        if timed_out:
            exc = TimeoutError(f"no terminal status within {POLL_TIMEOUT_S:.0f}s")
        elif failed:
            exc = RuntimeError(f"run ended with status {status}")
        self.environment.events.request.fire(
            request_type="E2E", name="claim_full_lifecycle", response_time=e2e_ms, response_length=0,
            exception=exc, context={})
        _record_row({"t_end": time.time(), "e2e_ms": round(e2e_ms, 1), "submit_ms": round(submit_ms, 1),
                     "status": "timeout" if timed_out else status, "timed_out": int(timed_out), "run_id": run_id,
                     "users": getattr(self.environment.runner, "user_count", "")})

    def malformed(self):
        payload = {"policy_number": "BAD", "insured_name": "Test", "vehicle_registration": "NOT-A-PLATE",
                   "channel": "invalid_channel", "date_of_loss": "2030-01-01", "date_of_report": "2030-01-01",
                   "place_of_loss": "Nowhere", "estimated_loss_inr": -100.0, "description": "Short"}
        with self.client.post("/claims", json=payload, name="POST /claims [malformed]", catch_response=True) as resp:
            if resp.status_code == 422:
                resp.success()
            else:
                resp.failure(f"expected 422 for malformed claim, got {resp.status_code}")
