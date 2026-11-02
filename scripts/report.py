"""Print the system's run parameters, straight from Prometheus (A6, L5, D5).

    python scripts/report.py                          # last 15 minutes, Prometheus on localhost:9090
    python scripts/report.py --window 1h --json out.json
    python scripts/report.py --prom http://localhost:9090

Nothing here is stored or invented: every number is a PromQL query over the metrics the API and the workers export, so
it reports whatever the system did in the window. Run it after you have sent traffic (a load test, the demo scripts,
or `python eval/run_eval.py` against a running stack is NOT enough: eval runs in-process and does not touch Prometheus).

Reported:
  runs by terminal status and task success rate        (A6)
  steps and tool calls per run                          (A6)
  end-to-end run latency p50 / p95, per-step p95        (A6)
  LLM retry rate                                        (A6)
  tokens and cost per agent and per step, cost per 100 runs, largest driver   (L5)
  guardrail triggers                                    (L4)
  cache hits, rate-limit rejections                     (D7)
  scrape targets up, and uptime per instance            (A6 container uptime)

Definitions: "task success rate" = runs that ended `completed` / all finished runs. `escalated` is a correct outcome
for some inputs (out-of-corpus, injection) so it is shown separately rather than counted as a failure; `failed` and
`degraded` are the ones that mean the system did not do its job.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request


class Prom:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def q(self, expr: str) -> list[tuple[dict, float]]:
        url = f"{self.base}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
        with urllib.request.urlopen(url, timeout=15) as r:
            body = json.load(r)
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus error for {expr!r}: {body}")
        out = []
        for s in body["data"]["result"]:
            v = float(s["value"][1])
            if v == v:                                    # drop NaN (0/0 when there is no traffic)
                out.append((s["metric"], v))
        return out

    def scalar(self, expr: str):
        r = self.q(expr)
        return r[0][1] if r else None


def collect(q, w: str) -> dict:
    """q: callable(expr) -> list[(labels, value)]. Pure function of the query results, so it is unit-testable."""
    sc = lambda e: (lambda r: r[0][1] if r else None)(q(e))
    d: dict = {"window": w}

    runs = {m.get("status", "?"): v for m, v in q(f"sum by (status) (increase(claim_runs_total[{w}]))")}
    total = sum(runs.values())
    d["runs_by_status"], d["runs_total"] = runs, total
    d["task_success_rate"] = (runs.get("completed", 0) / total) if total else None
    d["failed_or_degraded_rate"] = ((runs.get("failed", 0) + runs.get("degraded", 0)) / total) if total else None

    d["steps_per_run"] = sc(f"sum(increase(claim_run_steps_sum[{w}])) / sum(increase(claim_run_steps_count[{w}]))")
    d["tool_calls_per_run"] = sc(f"sum(increase(claim_run_tool_calls_sum[{w}])) / sum(increase(claim_run_tool_calls_count[{w}]))")
    for name, quant in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        d[f"run_latency_{name}_s"] = sc(f"histogram_quantile({quant}, sum by (le) (increase(claim_run_duration_seconds_bucket[{w}])))")
    d["step_latency_p95_s"] = {f"{m.get('agent')}:{m.get('action')}": v for m, v in
                               q(f"histogram_quantile(0.95, sum by (le, agent, action) (increase(agent_step_duration_seconds_bucket[{w}])))")}

    calls = sc(f"sum(increase(llm_calls_total[{w}]))")
    retries = sc(f"sum(increase(llm_retries_total[{w}]))")
    d["llm_calls"], d["llm_retries"] = calls, retries
    d["llm_retry_rate"] = (retries or 0) / calls if calls else None
    d["llm_calls_per_run"] = calls / total if calls and total else None

    d["tokens_by_agent"] = {f"{m.get('agent')}:{m.get('kind')}": v for m, v in
                            q(f"sum by (agent, kind) (increase(llm_tokens_total[{w}]))")}
    cost_steps = {f"{m.get('agent')}:{m.get('action')}": v for m, v in
                  q(f"sum by (agent, action) (increase(llm_step_cost_usd_total[{w}]))")}
    d["cost_by_step_usd"] = cost_steps
    cost_total = sum(cost_steps.values())
    d["cost_total_usd"] = cost_total
    d["cost_per_100_runs_usd"] = 100 * cost_total / total if total else None
    by_agent: dict[str, float] = {}
    for k, v in cost_steps.items():
        by_agent[k.split(":")[0]] = by_agent.get(k.split(":")[0], 0) + v
    d["cost_by_agent_usd"] = by_agent
    d["largest_cost_driver_agent"] = max(by_agent, key=by_agent.get) if cost_total > 0 else None
    d["largest_cost_driver_step"] = max(cost_steps, key=cost_steps.get) if cost_total > 0 else None

    d["guardrail_triggers"] = {m.get("check"): v for m, v in q(f"sum by (check) (increase(guardrail_triggers_total[{w}]))")}
    d["cache_hits"] = sc(f"sum(increase(run_cache_hits_total[{w}]))")
    d["cache_misses"] = sc(f"sum(increase(run_cache_misses_total[{w}]))")
    d["rate_limit_rejections"] = sc(f"sum(increase(rate_limit_exceeded_total[{w}]))")
    d["targets_up"] = {f"{m.get('job')}@{m.get('instance')}": v for m, v in q("up")}
    d["uptime_s"] = {f"{m.get('job')}@{m.get('instance')}": v for m, v in q("time() - process_start_time_seconds")}
    return d


def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None else fmt.format(v)


def render(d: dict) -> str:
    L = [f"# Run parameters (Prometheus, last {d['window']})", ""]
    L.append(f"Runs finished: {_f(d['runs_total'], '{:.0f}')}   by status: "
             + (", ".join(f"{k}={v:.0f}" for k, v in sorted(d['runs_by_status'].items())) or "none"))
    L.append(f"Task success rate (completed / all): {_f(d['task_success_rate'], '{:.1%}')}   "
             f"failed or degraded: {_f(d['failed_or_degraded_rate'], '{:.1%}')}")
    L.append(f"Steps per run: {_f(d['steps_per_run'], '{:.1f}')}   tool calls per run: {_f(d['tool_calls_per_run'], '{:.1f}')}   "
             f"LLM calls per run: {_f(d['llm_calls_per_run'], '{:.1f}')}")
    L.append(f"End-to-end run latency (worker side) p50 / p95 / p99: {_f(d['run_latency_p50_s'], '{:.1f}')} / "
             f"{_f(d['run_latency_p95_s'], '{:.1f}')} / {_f(d['run_latency_p99_s'], '{:.1f}')} s")
    L.append("  (This is processing time inside the worker. Submission-to-result latency, which includes queueing, is what "
             "scripts/load_steps.py measures against the SLO.)")
    L.append(f"LLM retry rate: {_f(d['llm_retry_rate'], '{:.1%}')} ({_f(d['llm_retries'], '{:.0f}')} retries / {_f(d['llm_calls'], '{:.0f}')} calls)")
    if d["step_latency_p95_s"]:
        L += ["", "Per-step p95 latency (s):"] + [f"  {k:34s} {v:8.2f}" for k, v in sorted(d["step_latency_p95_s"].items(), key=lambda kv: -kv[1])]
    if d["tokens_by_agent"]:
        L += ["", "Tokens in window:"] + [f"  {k:34s} {v:12.0f}" for k, v in sorted(d["tokens_by_agent"].items())]
    L += ["", f"Cost in window: ${_f(d['cost_total_usd'], '{:.6f}')}   COST PER 100 RUNS: ${_f(d['cost_per_100_runs_usd'], '{:.4f}')}"]
    if d["largest_cost_driver_agent"]:
        tot = d["cost_total_usd"]
        L.append(f"Largest cost driver: agent `{d['largest_cost_driver_agent']}` ({100 * d['cost_by_agent_usd'][d['largest_cost_driver_agent']] / tot:.1f}%), "
                 f"step `{d['largest_cost_driver_step']}` ({100 * d['cost_by_step_usd'][d['largest_cost_driver_step']] / tot:.1f}%)")
        L += ["  by step:"] + [f"    {k:32s} ${v:.6f}" for k, v in sorted(d["cost_by_step_usd"].items(), key=lambda kv: -kv[1])]
    else:
        L.append("  (cost is 0: LLM_PRICE_IN_PER_1K / LLM_PRICE_OUT_PER_1K are unset, or no model calls were made)")
    if d["guardrail_triggers"]:
        L += ["", "Guardrail triggers: " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(d["guardrail_triggers"].items()))]
    L += ["", f"Cache hits/misses: {_f(d['cache_hits'], '{:.0f}')}/{_f(d['cache_misses'], '{:.0f}')}   "
              f"rate-limit rejections: {_f(d['rate_limit_rejections'], '{:.0f}')}"]
    if d["targets_up"]:
        L += ["", "Scrape targets (up = 1) and uptime:"]
        for k in sorted(d["targets_up"]):
            up = d["uptime_s"].get(k)
            L.append(f"  {k:36s} up={d['targets_up'][k]:.0f}  uptime={_f(up, '{:.0f}')} s")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prom", default="http://localhost:9090")
    ap.add_argument("--window", default="15m", help="PromQL range, e.g. 5m, 1h")
    ap.add_argument("--json", help="also write the raw numbers here")
    a = ap.parse_args()
    try:
        d = collect(Prom(a.prom).q, a.window)
    except Exception as e:
        sys.exit(f"could not query Prometheus at {a.prom}: {e}\n(is the stack up? `docker compose up`, then send some traffic)")
    print(render(d))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(d, f, indent=2)


if __name__ == "__main__":
    main()
