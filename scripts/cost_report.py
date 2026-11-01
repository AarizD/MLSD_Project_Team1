"""Cost report (L5): per-step attribution, cost per 100 runs, largest single cost driver.

Reads finished run records from either a runs directory (what the worker writes) or Redis (what the API serves):

    python scripts/cost_report.py --runs-dir runs
    python scripts/cost_report.py --redis-url redis://localhost:6379/0 --out docs/cost_report.md

Each LLM step in a record already carries prompt_tokens, completion_tokens and cost_usd, priced AT CALL TIME
by the gateway from LLM_PRICE_IN_PER_1K / LLM_PRICE_OUT_PER_1K. If those were 0 (e.g. a stub run), pass
--price-in/--price-out (USD per 1k tokens) to re-price the recorded token counts; the report then says so.

"Largest single cost driver" is reported at two levels, because the answer differs:
  * by agent                (triage / coverage / verifier)
  * by agent + step action  (e.g. coverage decide:search_corpus), which is the thing you can actually change
and each is split into the prompt-token share and the completion-token share.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_records(runs_dir: str | None, redis_url: str | None, limit: int | None) -> list[dict]:
    recs = []
    if runs_dir:
        for p in sorted(Path(runs_dir).glob("run_*.json")):
            try:
                recs.append(json.loads(p.read_text()))
            except Exception:
                continue
    if redis_url:
        import redis
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        for k in r.scan_iter("run:run_*"):
            try:
                v = json.loads(r.get(k) or "null")
            except Exception:
                v = None
            if v and v.get("steps"):
                recs.append(v)
    recs = [r for r in recs if r.get("steps")]
    return recs[-limit:] if limit else recs


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def analyse(records: list[dict], price_in: float | None, price_out: float | None) -> dict:
    def step_cost(s):
        if price_in is not None or price_out is not None:
            return ((s.get("prompt_tokens") or 0) * (price_in or 0) + (s.get("completion_tokens") or 0) * (price_out or 0)) / 1000
        return s.get("cost_usd") or 0.0

    run_costs, run_tokens, llm_calls = [], [], []
    by_agent = defaultdict(lambda: dict(cost=0.0, pt=0, ct=0, calls=0, cost_in=0.0, cost_out=0.0))
    by_step = defaultdict(lambda: dict(cost=0.0, pt=0, ct=0, calls=0, cost_in=0.0, cost_out=0.0))
    free_runs = 0
    for rec in records:
        rc = rt = n = 0
        for s in rec["steps"]:
            if not s.get("prompt_version"):
                continue
            c = step_cost(s)
            ci = (s.get("prompt_tokens") or 0) * ((price_in if price_in is not None else 0)) / 1000
            co = (s.get("completion_tokens") or 0) * ((price_out if price_out is not None else 0)) / 1000
            for d in (by_agent[s["agent"]], by_step[(s["agent"], s["action"])]):
                d["cost"] += c
                d["pt"] += s.get("prompt_tokens") or 0
                d["ct"] += s.get("completion_tokens") or 0
                d["calls"] += 1
                d["cost_in"] += ci
                d["cost_out"] += co
            rc += c
            rt += (s.get("prompt_tokens") or 0) + (s.get("completion_tokens") or 0)
            n += 1
        run_costs.append(rc)
        run_tokens.append(rt)
        llm_calls.append(n)
        free_runs += n == 0                      # e.g. stopped by a guardrail before any model call

    total = sum(run_costs)
    total_tokens = sum(run_tokens)
    return dict(
        runs=len(records), repriced=price_in is not None or price_out is not None,
        price_in_per_1k=price_in, price_out_per_1k=price_out,
        total_cost_usd=total, mean_cost_usd=total / max(len(records), 1),
        p50_cost_usd=pct(run_costs, 0.5), p95_cost_usd=pct(run_costs, 0.95), max_cost_usd=max(run_costs, default=0.0),
        cost_per_100_runs_usd=100 * total / max(len(records), 1),
        mean_tokens_per_run=total_tokens / max(len(records), 1),
        mean_llm_calls_per_run=statistics.mean(llm_calls) if llm_calls else 0.0,
        runs_with_no_model_call=free_runs,
        by_agent={k: v for k, v in sorted(by_agent.items(), key=lambda kv: -kv[1]["cost"])},
        by_step={f"{a}:{act}": v for (a, act), v in sorted(by_step.items(), key=lambda kv: -kv[1]["cost"])},
    )


def render(a: dict) -> str:
    n = max(a["runs"], 1)
    tok_total = sum(v["pt"] + v["ct"] for v in a["by_agent"].values()) or 1
    L = [f"# Cost report ({a['runs']} runs)", ""]
    if a["repriced"]:
        L += [f"Token counts are as recorded; **re-priced** at ${a['price_in_per_1k']}/1k prompt and "
              f"${a['price_out_per_1k']}/1k completion tokens (`--price-in/--price-out`).", ""]
    else:
        L += ["Costs are as recorded at call time by the gateway.", ""]
    L += [f"| Metric | Value |", "| --- | --- |",
          f"| **Cost per 100 runs** | **${a['cost_per_100_runs_usd']:.4f}** |",
          f"| Mean cost per run | ${a['mean_cost_usd']:.6f} |",
          f"| p50 / p95 / max per run | ${a['p50_cost_usd']:.6f} / ${a['p95_cost_usd']:.6f} / ${a['max_cost_usd']:.6f} |",
          f"| Mean tokens per run | {a['mean_tokens_per_run']:.0f} |",
          f"| Mean LLM calls per run | {a['mean_llm_calls_per_run']:.2f} |",
          f"| Runs that made no model call (guardrail / validation stop) | {a['runs_with_no_model_call']} |", ""]
    L += ["## By agent", "", "| Agent | LLM calls/run | Prompt tok/run | Completion tok/run | Cost/100 runs | Share of cost | Share of tokens |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    tc = a["total_cost_usd"] or 1
    for k, v in a["by_agent"].items():
        L.append(f"| {k} | {v['calls'] / n:.2f} | {v['pt'] / n:.0f} | {v['ct'] / n:.0f} | ${100 * v['cost'] / n:.4f} | "
                 f"{100 * v['cost'] / tc:.1f}% | {100 * (v['pt'] + v['ct']) / tok_total:.1f}% |")
    L += ["", "## By agent and step", "", "| Step | Calls/run | Prompt tok/call | Completion tok/call | Cost/100 runs | Share of cost |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k, v in a["by_step"].items():
        c = max(v["calls"], 1)
        L.append(f"| {k} | {v['calls'] / n:.2f} | {v['pt'] / c:.0f} | {v['ct'] / c:.0f} | ${100 * v['cost'] / n:.4f} | "
                 f"{100 * v['cost'] / tc:.1f}% |")
    if a["by_agent"] and a["total_cost_usd"] > 0:
        top_a = next(iter(a["by_agent"]))
        top_s, ts = next(iter(a["by_step"].items()))
        L += ["", f"**Largest cost driver by agent: `{top_a}`** ({100 * a['by_agent'][top_a]['cost'] / tc:.1f}% of cost).  ",
              f"**Largest single step: `{top_s}`** ({100 * ts['cost'] / tc:.1f}% of cost)."]
        pt_share = 100 * sum(v["pt"] for v in a["by_agent"].values()) / tok_total
        L.append(f"Prompt tokens are {pt_share:.1f}% of all tokens: cost is driven by input size, not by output length.")
    elif a["total_cost_usd"] == 0:
        L += ["", "_All recorded costs are 0 (prices unset). Re-run with `--price-in/--price-out`._"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir")
    ap.add_argument("--redis-url")
    ap.add_argument("--limit", type=int, help="only the most recent N records")
    ap.add_argument("--price-in", type=float, help="USD per 1k prompt tokens (re-price recorded tokens)")
    ap.add_argument("--price-out", type=float, help="USD per 1k completion tokens")
    ap.add_argument("--out", help="write markdown here (also prints)")
    ap.add_argument("--json", help="write the raw numbers here")
    args = ap.parse_args()
    if not (args.runs_dir or args.redis_url):
        ap.error("give --runs-dir and/or --redis-url")
    recs = load_records(args.runs_dir, args.redis_url, args.limit)
    if not recs:
        sys.exit("no run records found")
    a = analyse(recs, args.price_in, args.price_out)
    md = render(a)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps(a, indent=2))


if __name__ == "__main__":
    main()
