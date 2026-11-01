"""Runs one claim through the agent graph and stores the result.

In Docker, RQ workers call process_run() for each job taken from the Redis queue.
Locally (QUEUE_BACKEND=inline) the API calls it on a background thread instead.
"""

from __future__ import annotations

import datetime as dt
import logging
import traceback

from app.store import RUN_TTL_S, STORE

log = logging.getLogger("worker")


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def process_run(run_id: str, fnol: dict) -> str:
    from app.agents import run_claim  # imported here so the API process starts fast

    run = STORE.get_json(f"run:{run_id}") or {"run_id": run_id}
    run.update(status="running", started_at=_now())
    STORE.set_json(f"run:{run_id}", run, ttl=RUN_TTL_S)
    log.info("run started", extra={"run_id": run_id})

    try:
        record = run_claim(fnol, run_id=run_id)
        run.update(status=record["status"], steps=record["steps"], result=record["result"],
                   config=record.get("config"), finished_at=_now())
    except Exception as e:  # never leave a run stuck in "running"
        run.update(status="failed", finished_at=_now(),
                   result={"escalated": True, "route": "human_review",
                           "escalation_reason": f"Worker error: {type(e).__name__}: {e}"},
                   error=traceback.format_exc()[-2000:])
    STORE.set_json(f"run:{run_id}", run, ttl=RUN_TTL_S)
    try:   # metrics must never turn a finished run into a failed one
        from app.metrics import emit_run_metrics
        emit_run_metrics({"status": run["status"], "steps": run.get("steps"), "result": run.get("result")})
    except Exception:
        log.exception("metrics emission failed", extra={"run_id": run_id})
    log.info("run finished", extra={"run_id": run_id, "status": run["status"]})
    return run["status"]