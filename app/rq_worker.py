"""RQ worker entry point used in Docker.

Imports the agent graph once at startup, so each job (which RQ runs in a forked child
process) starts with LangGraph, the prompts and, later, B's embedding model already loaded.

Metrics (D5/A6/L5): each job runs in a forked child, so counters are collected with prometheus_client's
multiprocess mode and served from this long-lived parent on WORKER_METRICS_PORT. Prometheus discovers every
scaled replica through DNS (see observability/prometheus.yml).
"""

import logging
import os
import shutil

# must happen before anything imports prometheus_client
MULTIPROC_DIR = os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", "/tmp/prom_multiproc")
shutil.rmtree(MULTIPROC_DIR, ignore_errors=True)   # stale files from a previous container life would double-count
os.makedirs(MULTIPROC_DIR, exist_ok=True)

from redis import Redis  # noqa: E402
from rq import Queue, Worker  # noqa: E402

import app.agents  # noqa: E402,F401  preload

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

if __name__ == "__main__":
    port = int(os.getenv("WORKER_METRICS_PORT", "9101"))
    if port:
        from prometheus_client import CollectorRegistry, multiprocess, start_http_server
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        start_http_server(port, registry=reg)
        logging.getLogger("worker").info("worker metrics on :%s", port)
    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    Worker([Queue("claims", connection=conn)], connection=conn).work(with_scheduler=False)
