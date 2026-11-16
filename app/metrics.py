"""Prometheus metrics definition for the Claim Intake & Triage service (D5, A6)."""

from __future__ import annotations

import os

from fastapi import Response

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _DummyMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def observe(self, amount):
            pass

        def set(self, value):
            pass

    Counter = Histogram = Gauge = _DummyMetric

    def generate_latest():
        return b"# prometheus_client not installed in this environment\nclaims_http_requests_total 0\n"


# HTTP metrics
HTTP_REQUESTS_TOTAL = Counter(
    "claims_http_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "claims_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Ingestion & Cache metrics
CLAIMS_SUBMITTED_TOTAL = Counter(
    "claims_submitted_total",
    "Total FNOL claims submitted at the ingestion boundary",
    ["channel", "status"],  # status: queued, quarantined, cached
)
RUN_CACHE_HITS_TOTAL = Counter(
    "claims_cache_hits_total",
    "Total claim submissions resolved via the run cache (D7)",
)
RUN_CACHE_MISSES_TOTAL = Counter(
    "claims_cache_misses_total",
    "Total claim submissions requiring execution (cache miss)",
)
RATE_LIMIT_EXCEEDED_TOTAL = Counter(
    "claims_rate_limit_exceeded_total",
    "Total requests rejected by source-IP rate limiting (D7)",
)

# Agent performance & execution metrics (A6)
AGENT_STEP_DURATION_SECONDS = Histogram(
    "agent_step_duration_seconds",
    "Latency of individual agent steps and tool calls",
    ["agent", "action"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)
AGENT_STEP_COST_USD = Counter(
    "agent_step_cost_usd_total",
    "Attributed cost of agent executions in USD (L5)",
    ["agent"],
)
CLAIM_RUN_DURATION_SECONDS = Histogram(
    "claim_run_duration_seconds",
    "End-to-end duration of a claim triage run from start to finish",
    ["outcome"],  # outcome: completed, escalated, failed
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0),
)


# L5: tokens and cost attributed per agent AND per step kind (the thing you need to find the cost driver)
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total", "LLM tokens consumed, per agent and prompt version", ["agent", "prompt_version", "kind"],
)  # kind: prompt | completion
LLM_STEP_COST_USD = Counter(
    "llm_step_cost_usd_total", "LLM cost in USD attributed per agent and step action (L5)", ["agent", "action"],
)
LLM_CALLS_TOTAL = Counter("llm_calls_total", "LLM calls made", ["agent", "prompt_version"])
LLM_RETRIES_TOTAL = Counter("llm_retries_total", "LLM call retries (A6 retry rate = retries / calls)", ["agent"])
TOOL_CALLS_TOTAL = Counter("agent_tool_calls_total", "Agent tool calls", ["tool", "outcome"])  # outcome: ok | error
RUNS_TOTAL = Counter("claim_runs_total", "Finished agent runs by terminal status (A6 task success rate)", ["status"])
RUN_STEPS = Histogram("claim_run_steps", "Steps per run", buckets=(3, 5, 8, 10, 12, 15, 20, 30))
RUN_TOOL_CALLS = Histogram("claim_run_tool_calls", "Tool calls per run", buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10))
RUN_COST_USD = Histogram("claim_run_cost_usd", "Cost of one run in USD",
                         buckets=(0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
GUARDRAIL_TRIGGERS_TOTAL = Counter("guardrail_triggers_total", "Guardrail firings (L4)", ["check"])
# check: prompt_injection | toxicity_severe | toxicity_mild | unauthorised_determination_blocked | toxic_output_blocked ...


def emit_run_metrics(record: dict) -> None:
    """Turn one finished run record into Prometheus samples. Called by the worker after every run.

    Cost/tokens come from the per-step values the gateway computed AT CALL TIME (priced with the env
    prices in force then); this only exports them, it never re-prices.
    """
    steps = record.get("steps") or []
    result = record.get("result") or {}
    known_agents = {"guardrail", "triage", "coverage", "verifier", "router", "finalize", "system"}
    decision_actions = {"decide:finish", "decide:search_corpus", "decide:get_clause", "decide:get_fnol_field",
                        "decide:other"}
    known_actions = {"check_input", "classify_and_extract", "verify", "provider_unavailable", "tool_call",
                     "route", "assemble_result"} | decision_actions
    for s in steps:
        agent, action = s.get("agent", "unknown"), s.get("action", "unknown")
        agent = agent if isinstance(agent, str) and agent in known_agents else "other"
        # The model's proposed action is included in step text; never use that free text as a metric label.
        if not isinstance(action, str):
            action = "other"
        elif action.startswith("decide:") and action not in decision_actions:
            action = "decide:other"
        elif action not in known_actions:
            action = "other"
        AGENT_STEP_DURATION_SECONDS.labels(agent=agent, action=action).observe((s.get("latency_ms") or 0) / 1000)
        if s.get("prompt_version"):
            pv = s["prompt_version"]
            LLM_CALLS_TOTAL.labels(agent=agent, prompt_version=pv).inc()
            LLM_TOKENS_TOTAL.labels(agent=agent, prompt_version=pv, kind="prompt").inc(s.get("prompt_tokens") or 0)
            LLM_TOKENS_TOTAL.labels(agent=agent, prompt_version=pv, kind="completion").inc(s.get("completion_tokens") or 0)
            LLM_STEP_COST_USD.labels(agent=agent, action=action).inc(s.get("cost_usd") or 0)
            AGENT_STEP_COST_USD.labels(agent=agent).inc(s.get("cost_usd") or 0)
            if s.get("retries"):
                LLM_RETRIES_TOTAL.labels(agent=agent).inc(s["retries"])
        if s.get("tool"):
            TOOL_CALLS_TOTAL.labels(tool=s["tool"], outcome="error" if s.get("error") else "ok").inc()
    status = record.get("status", "failed")
    RUNS_TOTAL.labels(status=status).inc()
    RUN_STEPS.observe(len(steps))
    RUN_TOOL_CALLS.observe(result.get("tool_calls") or 0)
    RUN_COST_USD.observe(result.get("total_cost_usd") or 0)
    CLAIM_RUN_DURATION_SECONDS.labels(outcome=status).observe((result.get("latency_ms") or 0) / 1000)
    for f in result.get("guardrail_flags") or []:
        GUARDRAIL_TRIGGERS_TOTAL.labels(check=f.split(":")[0]).inc()


def _registry():
    """In a worker, RQ forks a child per job and the child does the counting, so a plain registry in the
    scraped parent would stay empty. PROMETHEUS_MULTIPROC_DIR switches on prometheus_client's multiprocess
    mode (child writes mmap files, the parent aggregates). Unset (the API container, local runs) = default."""
    if PROMETHEUS_AVAILABLE and os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import CollectorRegistry, multiprocess
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        return reg
    return None


def render_metrics() -> bytes:
    reg = _registry()
    return generate_latest(reg) if reg is not None else generate_latest()


def metrics_endpoint() -> Response:
    """FastAPI endpoint handler returning Prometheus text exposition format."""
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)


class IndexFreshnessCollector:
    """Exports the corpus index's freshness and the last build's throughput (D2), read from
    index/manifest.json (written by `python corpus.py index`) at every scrape, so a rebuild shows up without a
    restart. Registered on the worker's metrics server, the container that mounts the index."""

    def __init__(self, index_dir=None):
        self.index_dir = index_dir or os.getenv("CORPUS_INDEX_DIR", "index")

    def collect(self):
        import json
        import datetime as dt
        from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
        try:
            with open(os.path.join(self.index_dir, "manifest.json"), encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            return
        built = dt.datetime.fromisoformat(m["built_at_utc"]).timestamp()
        for name, doc, value in (
            ("corpus_index_built_timestamp_seconds", "When the index was last built (Unix time)", built),
            ("corpus_index_chunks", "Chunks in the index", m.get("chunks")),
            ("corpus_index_documents", "Source documents in the index", m.get("documents")),
            ("corpus_index_last_build_embedded_chunks", "Chunks embedded by the last build", m.get("embedded_this_run")),
            ("corpus_index_last_build_seconds", "Embedding time of the last build", m.get("embedding_seconds")),
            ("corpus_index_build_throughput_chunks_per_second", "Embedding throughput of the last build",
             m.get("throughput_chunks_per_second")),
        ):
            if value is not None:
                yield GaugeMetricFamily(name, doc, value=float(value))
        info = InfoMetricFamily("corpus_index", "Index model, build mode and source document versions")
        info.add_metric([], {"model": str(m.get("model")), "mode": str(m.get("mode")),
                             **{f"doc_{i}": f"{k}@{v}" for i, (k, v) in enumerate(sorted(m.get("versions", {}).items()))}})
        yield info
