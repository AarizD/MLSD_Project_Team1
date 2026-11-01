"""Batch evaluation from the command line (L3, and the harness for L2 prompt comparisons).

    # run the fixed scenario set once and record metrics (+ TruLens records)
    python eval/run_eval.py run --label baseline

    # same set, one deliberate change, applied as environment overrides before the agents load
    python eval/run_eval.py run --label after --set MAX_VERIFY_ATTEMPTS=2 --set STUB_MODE=hallucinate

    # prompt A/B (L2): identical set, only the prompt version differs
    python eval/run_eval.py run --label cov-v1 --set COVERAGE_PROMPT=coverage-v1
    python eval/run_eval.py run --label cov-v2 --set COVERAGE_PROMPT=coverage-v2

    python eval/run_eval.py compare eval/results/before.json eval/results/after.json

Provider: whatever the environment configures (LLM_PROVIDER=stub by default, so it runs offline and in CI). With a real
provider set LLM_PROVIDER/LLM_BASE_URL/LLM_API_KEY/LLM_MODEL and the price variables, exactly as for the service.

TruLens: every scenario run is recorded as a TruLens record in eval/results/trulens.sqlite and the four metrics are
registered as custom feedback functions (see eval/metrics.py for the definitions), so `trulens` leaderboards and the
dashboard work on them. Pass --no-trulens to skip; the JSON/markdown results are produced either way and are
computed by the same functions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

METRIC_KEYS = ["context_relevance", "context_recall", "groundedness", "faithfulness",
               "hallucination_rate_draft", "hallucination_rate_returned"]
OUTCOME_KEYS = ["outcome_accuracy", "answer_rate", "escalation_precision", "escalation_recall", "support_hit_rate",
                "mean_cost_usd", "mean_latency_ms", "mean_llm_calls"]


def build_fnol(base: dict, case: dict) -> dict:
    f = {**base, **case["fnol"]}
    today = dt.date.today()
    f["date_of_loss"] = str(today - dt.timedelta(days=f.pop("loss_days_ago")))
    f["date_of_report"] = str(today - dt.timedelta(days=f.pop("report_days_ago")))
    return {k: v for k, v in f.items() if v is not None}


def apply_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if not k or not _:
            sys.exit(f"--set expects KEY=VALUE, got {p!r}")
        os.environ[k] = v
        out[k] = v
    return out


def load_agents():
    os.environ.setdefault("RUNS_DIR", tempfile.mkdtemp(prefix="eval_runs_"))
    os.environ.setdefault("LLM_PROVIDER", "stub")
    from app import agents
    return agents


# --------------------------------------------------------------------------- TruLens (optional)
class TruRecorder:
    """Wraps each run as a TruLens record and attaches the custom metric feedbacks. Never allowed to break the eval."""

    def __init__(self, label: str, db_path: Path, scorer):
        self.ok, self.err = False, None
        try:
            from trulens.apps.app import TruApp, instrument
            from trulens.core import Feedback, TruSession
            self.db_path = str(db_path)
            self.counts = {}
            self.session = TruSession(database_url=f"sqlite:///{db_path}")

            class ClaimApp:
                def __init__(self, fn):
                    self.fn = fn

                @instrument
                def run(self, case_json: str) -> str:
                    return self.fn(case_json)

            def feedback(name):
                def fn(inp: str, out: str):
                    v = scorer(name, inp, out)
                    return float(v) if v is not None else float("nan")
                fn.__name__ = name
                return Feedback(fn, name=name).on_input().on_output()

            names = ["groundedness", "context_relevance", "faithfulness", "hallucination_returned"]
            self.app = ClaimApp(self._call)
            self.tru = TruApp(self.app, app_name="motor-claim-triage", app_version=label,
                              feedbacks=[feedback(n) for n in names])
            self._payload = None
            self.ok = True
        except Exception as e:                       # TruLens missing or API drift: keep evaluating
            self.err = f"{type(e).__name__}: {e}"

    def _call(self, case_json: str) -> str:
        return self._payload

    def record(self, case_json: str, output_json: str):
        if not self.ok:
            return
        self._payload = output_json
        with self.tru:
            self.app.run(case_json)

    def finish(self):
        """Verify what actually landed in the TruLens database. 'Constructed without error' is not 'recorded'."""
        if not self.ok:
            return
        try:
            self.session.wait_for_feedback_results(timeout=120)
        except Exception as e:
            self.err = f"feedback wait: {e}"
        try:
            import sqlite3
            con = sqlite3.connect(self.db_path)
            n = lambda t: con.execute(f"select count(*) from {t}").fetchone()[0]
            self.counts = {t: n(t) for t in ("trulens_records", "trulens_events", "trulens_feedbacks")}
            if not any(self.counts.values()):
                self.ok = False
                self.err = (f"TruLens registered {n('trulens_feedback_defs')} feedback definitions but persisted no records or "
                            f"feedback results ({self.counts}); wiring incomplete. Metrics below come from eval/metrics.py.")
        except Exception as e:
            self.err = f"could not verify TruLens database: {e}"


def cmd_run(a):
    overrides = apply_overrides(a.set)
    agents = load_agents()
    import metrics as M
    from app.schemas import validate_fnol

    doc = json.loads(Path(a.scenarios).read_text(encoding="utf-8"))
    cases = doc["scenarios"]
    if a.only:
        cases = [c for c in cases if c["id"] in a.only]
    ptext = lambda cid: (agents.get_clause(cid) or {}).get("text")
    ptype = lambda cid: (agents.get_clause(cid) or {}).get("section_type")

    def scorer(name, inp, out):
        try:
            case, record = json.loads(inp), json.loads(out)
            s = M.score_run(case, record, ptext, ptype)
            return {"groundedness": s.groundedness, "context_relevance": s.context_relevance, "faithfulness": s.faithfulness,
                    "hallucination_returned": None if s.hallucinated_returned is None else float(s.hallucinated_returned)}[name]
        except Exception:
            return None

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tru = None if a.no_trulens else TruRecorder(a.label, outdir / "trulens.sqlite", scorer)
    t0 = time.time()
    scores, per_case = [], []
    for case in cases:
        fnol = build_fnol(doc["base"], case)
        valid, errs = validate_fnol(fnol)
        if valid is None:
            sys.exit(f"scenario {case['id']} is not a valid FNOL (the eval set must pass the ingestion contract): {errs}")
        record = agents.run_claim(fnol)
        sc = M.score_run(case, record, ptext, ptype)
        scores.append(sc)
        per_case.append({**asdict(sc), "title": case["title"], "claim_text": case["fnol"]["description"][:140]})
        if tru and tru.ok:
            tru.record(json.dumps(case), json.dumps(record, default=str))
        print(f"{'PASS' if sc.outcome_ok else 'FAIL'} {case['id']} [{case['tier']}] {case['title'][:58]:58s} "
              f"{sc.status}/{sc.route}" + ("" if sc.outcome_ok else "  <- " + "; ".join(sc.failures)), flush=True)
    if tru:
        tru.finish()

    cfg = agents.CFG
    result = {
        "label": a.label, "ran_at": dt.datetime.now(dt.timezone.utc).isoformat(), "seconds": round(time.time() - t0, 1),
        "scenarios_file": a.scenarios, "n_scenarios": len(cases), "overrides": overrides,
        "config": {k: cfg[k] for k in ("LLM_PROVIDER", "LLM_MODEL", "TRIAGE_PROMPT", "COVERAGE_PROMPT", "VERIFIER_PROMPT",
                                       "STUB_MODE", "MAX_VERIFY_ATTEMPTS", "ENFORCE_MODIFIER_SEARCH", "WEAK_RETRIEVAL_THRESHOLD")},
        "trulens": {"enabled": bool(tru and tru.ok), "error": tru.err if tru else "disabled (--no-trulens)"},
        "metrics": M.aggregate(scores), "scenarios": per_case,
    }
    (outdir / f"{a.label}.json").write_text(json.dumps(result, indent=2, default=str))
    (outdir / f"{a.label}.md").write_text(render_run(result))
    print("\n" + render_run(result))
    if tru and tru.ok:
        try:
            print(tru.session.get_leaderboard().to_string())
        except Exception:
            pass
    elif tru:
        print(f"[trulens unavailable: {tru.err}]")


def pct(v):
    return "n/a" if v is None else (f"{v:.3f}")


def render_run(r: dict) -> str:
    m = r["metrics"]
    L = [f"# Evaluation `{r['label']}`", "",
         f"Scenarios: {r['n_scenarios']} ({m['core']['n_scenarios']} core, {m['stretch']['n_scenarios']} stretch). "
         f"Provider: `{r['config']['LLM_PROVIDER']}`. Overrides: `{r['overrides'] or 'none'}`. "
         f"Prompts: {r['config']['TRIAGE_PROMPT']} / {r['config']['COVERAGE_PROMPT']} / {r['config']['VERIFIER_PROMPT']}. "
         f"TruLens: {'recorded' if r['trulens']['enabled'] else 'not used (' + str(r['trulens']['error']) + ')'}.", "",
         "| Metric | all | core | stretch |", "| --- | --- | --- | --- |"]
    for k in OUTCOME_KEYS + METRIC_KEYS:
        L.append(f"| {k} | {pct(m['all'][k])} | {pct(m['core'][k])} | {pct(m['stretch'][k])} |")
    L.append(f"| n_answered | {m['all']['n_answered']} | {m['core']['n_answered']} | {m['stretch']['n_answered']} |")
    bad = [s for s in r["scenarios"] if not s["outcome_ok"]]
    L += ["", f"Failed scenarios ({len(bad)}):"] + [f"- {s['id']} [{s['tier']}] {s['title']}: {'; '.join(s['failures'])}" for s in bad]
    s41 = next((s for s in r["scenarios"] if "section_4_1" in s["tags"]), None)
    if s41:
        L += ["", f"Section 4.1 case ({s41['id']}): {'PASS' if s41['outcome_ok'] else 'FAIL'}; cited {s41['cited_ids']}; "
                  f"context recall {pct(s41['context_recall'])}."]
    return "\n".join(L) + "\n"


def cmd_compare(a):
    b, c = (json.loads(Path(p).read_text()) for p in (a.before, a.after))
    scope = a.scope
    L = [f"# Before / after: `{b['label']}` -> `{c['label']}` ({scope} scenarios)", "",
         f"Change: before overrides `{b['overrides'] or 'none'}`, after overrides `{c['overrides'] or 'none'}`.", "",
         "| Metric | Before | After | Delta |", "| --- | --- | --- | --- |"]
    for k in METRIC_KEYS + OUTCOME_KEYS:
        x, y = b["metrics"][scope][k], c["metrics"][scope][k]
        d = "n/a" if x is None or y is None else f"{y - x:+.3f}"
        L.append(f"| {k} | {pct(x)} | {pct(y)} | {d} |")
    L.append(f"| n_answered | {b['metrics'][scope]['n_answered']} | {c['metrics'][scope]['n_answered']} | "
             f"{c['metrics'][scope]['n_answered'] - b['metrics'][scope]['n_answered']:+d} |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.out:
        Path(a.out).write_text(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--scenarios", default=str(ROOT / "eval" / "scenarios.json"))
    r.add_argument("--set", action="append", help="KEY=VALUE environment override applied before the agents load")
    r.add_argument("--only", nargs="+", help="run only these scenario ids")
    r.add_argument("--out", default=str(ROOT / "eval" / "results"))
    r.add_argument("--no-trulens", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--scope", default="all", choices=["all", "core", "stretch"])
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
