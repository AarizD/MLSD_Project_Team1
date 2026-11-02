"""Evaluation metrics (L3). Deterministic, reference-based, no model calls, so the numbers are reproducible and the
before/after comparison measures the system and not judge noise.

HOW EACH METRIC IS COMPUTED (this docstring is the definition quoted in docs/EVALUATION.md)

  Scope: metrics are computed per run over the runs that RETURNED an answer (status completed). Escalated runs have no
  answer to score; they are covered by outcome accuracy and escalation precision/recall instead. `n_answered` is
  always reported next to the averages so a metric can't look good by answering almost nothing.

  context_relevance   Precision of retrieval against the labelled support.
                      = |retrieved ∩ relevant| / |retrieved| where `retrieved` is every distinct passage the agent
                      pulled (search + get_clause) and `relevant` is the scenario's labelled support clauses plus the
                      clauses its expected answer must cite. Companion `context_recall` = |retrieved ∩ relevant| /
                      |relevant|: the share of the needed passages (e.g. the use restriction in the section 4.1 case)
                      that retrieval actually surfaced.

  groundedness        Are the answer's statements supported by the passages it cites?
                      The coverage_position is split into sentences. A sentence is SUPPORTED if at least
                      GROUND_TAU (0.5) of its content-word stems occur in the concatenated text of the cited passages.
                      groundedness = supported sentences / sentences. (A lexical-entailment proxy for TruLens'
                      LLM/NLI groundedness. Strict on invented specifics, lenient on paraphrase.)

  faithfulness        Do the citations and the verdict match the evidence?
                      = quote_fidelity x verdict_consistent, where quote_fidelity is the share of citations whose
                      `support` quote appears verbatim (after whitespace/case normalisation) in the text of the cited
                      passage, and verdict_consistent is 1 unless the verdict contradicts the cited clause types:
                      `outside_cover` needs a cited limitation/exclusion clause; `within_cover` must not rest on an
                      exclusion alone. (Groundedness checks the prose; faithfulness checks the evidence trail.)

  hallucination_rate  Share of answered runs containing at least one fabricated citation: a clause id that is not among
                      the retrieved passages, or a `support` quote that is not in the cited passage. Reported twice:
                        draft     the generator's first finish attempt, BEFORE verification (what the model did)
                        returned  what reached the caller, AFTER verification (what the user would see)
                      returned > 0 means the verifier let something through; draft - returned is what it caught.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GROUND_TAU = 0.5
_STOP = set("a an the of to in on at for and or but is are was were be been being by with from as that this these those it its "
            "than then which who whom whose under over into upon any all each such not no nor if so do does did has have had "
            "will would shall should may might can could their there they them his her he she we you your our i my me".split())


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%]+", " ", (s or "").lower())).strip()


def stems(s: str) -> set[str]:
    out = set()
    for w in norm(s).split():
        if w in _STOP or len(w) < 3:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.;!?])\s+", text or "") if len(p.strip()) > 3]


# --------------------------------------------------------------------------- per-answer metrics
def groundedness(position: str, cited_texts: list[str]) -> float | None:
    sents = sentences(position)
    if not sents:
        return None
    pool = set().union(*(stems(t) for t in cited_texts)) if cited_texts else set()
    ok = 0
    for s in sents:
        st = stems(s)
        ok += bool(st) and len(st & pool) / len(st) >= GROUND_TAU
    return ok / len(sents)


def citation_issues(citations: list[dict], retrieved: dict[str, str]) -> tuple[int, int, int]:
    """(n_citations, n_unresolved_ids, n_non_verbatim_quotes)"""
    unresolved = bad_quote = 0
    for c in citations:
        text = retrieved.get(c.get("clause_id"))
        if text is None:
            unresolved += 1
        elif norm(c.get("support", "")) and norm(c["support"]) not in norm(text):
            bad_quote += 1
    return len(citations), unresolved, bad_quote


def faithfulness(applies: str | None, citations: list[dict], retrieved: dict[str, str], types: dict[str, str]) -> float | None:
    if not citations:
        return None
    n, unresolved, bad_quote = citation_issues(citations, retrieved)
    fidelity = (n - unresolved - bad_quote) / n
    kinds = {types.get(c["clause_id"]) for c in citations}
    consistent = 1.0
    if applies == "outside_cover" and not kinds & {"limitation", "exclusion"}:
        consistent = 0.0
    if applies == "within_cover" and kinds and kinds <= {"exclusion"}:
        consistent = 0.0
    return fidelity * consistent


def is_hallucinated(answer: dict | None, retrieved: dict[str, str]) -> bool | None:
    if not answer or not answer.get("citations"):
        return None
    _, unresolved, bad_quote = citation_issues(answer["citations"], retrieved)
    return (unresolved + bad_quote) > 0


# --------------------------------------------------------------------------- per-run scoring
@dataclass
class RunScore:
    id: str
    tier: str
    tags: list[str]
    status: str
    route: str | None
    outcome_ok: bool                       # every field the scenario pins down matches
    failures: list[str] = field(default_factory=list)
    answered: bool = False
    expected_escalation: bool = False
    escalated: bool = False
    context_relevance: float | None = None
    context_recall: float | None = None
    groundedness: float | None = None
    faithfulness: float | None = None
    hallucinated_draft: bool | None = None
    hallucinated_returned: bool | None = None
    support_hit: bool | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    retrieved_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)


def score_run(case: dict, record: dict, passage_text, passage_type) -> RunScore:
    """case = scenario dict; record = run_claim() output; passage_text/type = callables clause_id -> str | None."""
    exp, res = case["expect"], record.get("result") or {}
    steps = record.get("steps") or []
    cited = res.get("citations") or []
    answer = {"citations": [{"clause_id": c["clause_id"], "support": c.get("support")} for c in cited]}
    status = record.get("status")

    retrieved_ids: list[str] = []
    for s in steps:
        for pid in (s.get("passages") or s.get("retrieved") or []):
            pid = pid["clause_id"] if isinstance(pid, dict) else pid
            if pid not in retrieved_ids:
                retrieved_ids.append(pid)
    for c in cited:                                        # cited passages were necessarily retrieved
        if c["clause_id"] not in retrieved_ids:
            retrieved_ids.append(c["clause_id"])
    retrieved = {i: passage_text(i) or "" for i in retrieved_ids}
    types = {i: passage_type(i) for i in retrieved_ids}

    sc = RunScore(id=case["id"], tier=case["tier"], tags=case.get("tags", []), status=status, route=res.get("route"),
                  outcome_ok=True, answered=(status == "completed" and bool(cited)),
                  expected_escalation=exp.get("status") == "escalated", escalated=status in ("escalated", "degraded"),
                  cost_usd=res.get("total_cost_usd") or 0.0, latency_ms=res.get("latency_ms") or 0.0,
                  llm_calls=res.get("llm_calls") or 0, tool_calls=res.get("tool_calls") or 0,
                  retrieved_ids=retrieved_ids, cited_ids=[c["clause_id"] for c in cited])

    # ---- outcome: only the fields the scenario pins down
    def check(name, ok):
        if not ok:
            sc.outcome_ok = False
            sc.failures.append(name)

    check(f"status={status} expected {exp.get('status')}", status == exp.get("status"))
    if exp.get("route"):
        check(f"route={res.get('route')} expected {exp['route']}", res.get("route") == exp["route"])
    if exp.get("claim_type"):
        check(f"claim_type={res.get('claim_type')} expected {exp['claim_type']}", res.get("claim_type") == exp["claim_type"])
    if exp.get("applies"):
        check(f"applies={res.get('applies')} expected {exp['applies']}", res.get("applies") == exp["applies"])
    for cid in exp.get("must_cite", []):
        check(f"missing citation {cid}", cid in sc.cited_ids)
    for fl in exp.get("flags", []):
        check(f"missing flag {fl}", fl in (res.get("guardrail_flags") or []))
    if exp.get("escalation_contains"):
        check("escalation reason", exp["escalation_contains"].lower() in (res.get("escalation_reason") or "").lower())
    if "model_calls" in exp:
        check(f"model calls {sc.llm_calls} expected {exp['model_calls']}", sc.llm_calls == exp["model_calls"])
    blob = " ".join(str(res.get(k) or "") for k in ("coverage_position", "route_reason", "escalation_reason", "entities")).lower()
    for bad in exp.get("no_text", []):
        check(f"forbidden text '{bad}' returned", bad.lower() not in blob)

    # ---- supporting passage: the labelled quote must appear in a CITED passage
    if case.get("support"):
        cited_text = " ".join(norm(retrieved.get(i, "")) for i in sc.cited_ids)
        sc.support_hit = all(norm(s["quote"]) in cited_text for s in case["support"]) if sc.answered else False

    # ---- retrieval quality against the labelled support (defined when the scenario has labelled support)
    relevant = {s["clause_id"] for s in case.get("support", [])} | set(exp.get("must_cite", []))
    if relevant and retrieved_ids:
        hit = relevant & set(retrieved_ids)
        sc.context_relevance = len(hit) / len(retrieved_ids)
        sc.context_recall = len(hit) / len(relevant)

    # ---- answer quality (answered runs only)
    if sc.answered:
        texts = [retrieved.get(i, "") for i in sc.cited_ids]
        sc.groundedness = groundedness(res.get("coverage_position"), texts)
        sc.faithfulness = faithfulness(res.get("applies"), answer["citations"], retrieved, types)
        sc.hallucinated_returned = is_hallucinated(answer, retrieved)
    draft = next((s.get("output", {}).get("answer") for s in steps
                  if s.get("agent") == "coverage" and s.get("action") == "decide:finish" and (s.get("output") or {}).get("answer")), None)
    sc.hallucinated_draft = is_hallucinated(draft, retrieved)
    return sc


# --------------------------------------------------------------------------- aggregation
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(bool(x) for x in xs) / len(xs), 4) if xs else None


def aggregate(scores: list[RunScore]) -> dict:
    def block(ss: list[RunScore]) -> dict:
        ans = [s for s in ss if s.answered]
        exp_esc = [s for s in ss if s.expected_escalation]
        act_esc = [s for s in ss if s.escalated]
        tp = [s for s in ss if s.expected_escalation and s.escalated]
        return {
            "n_scenarios": len(ss), "n_answered": len(ans),
            "outcome_accuracy": _rate([s.outcome_ok for s in ss]),
            "answer_rate": round(len(ans) / len(ss), 4) if ss else None,
            "escalation_precision": round(len(tp) / len(act_esc), 4) if act_esc else None,
            "escalation_recall": round(len(tp) / len(exp_esc), 4) if exp_esc else None,
            "support_hit_rate": _rate([s.support_hit for s in ss]),
            "context_relevance": _mean([s.context_relevance for s in ans]),
            "context_recall": _mean([s.context_recall for s in ans]),
            "groundedness": _mean([s.groundedness for s in ans]),
            "faithfulness": _mean([s.faithfulness for s in ans]),
            "hallucination_rate_draft": _rate([s.hallucinated_draft for s in ss]),
            "hallucination_rate_returned": _rate([s.hallucinated_returned for s in ans]),
            "mean_cost_usd": round(sum(s.cost_usd for s in ss) / len(ss), 6) if ss else None,
            "cost_per_100_runs_usd": round(100 * sum(s.cost_usd for s in ss) / len(ss), 4) if ss else None,
            "mean_latency_ms": round(sum(s.latency_ms for s in ss) / len(ss), 1) if ss else None,
            "mean_llm_calls": round(sum(s.llm_calls for s in ss) / len(ss), 2) if ss else None,
        }
    return {"all": block(scores), "core": block([s for s in scores if s.tier == "core"]),
            "stretch": block([s for s in scores if s.tier == "stretch"])}
