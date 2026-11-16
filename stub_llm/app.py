"""Stub LLM provider for load tests (D6) and failure demos (A5).

An OpenAI-compatible /v1/chat/completions endpoint that answers with fixed, valid responses after a realistic
delay, so a load test measures OUR system and not a provider's rate limits. It is async, so it can hold thousands
of concurrent "in-flight model calls" on one process: it cannot be the bottleneck being studied.

It is stateless and deterministic apart from the delay. The agent loop is multi-turn, so the stub decides what to say
by reading the prompt it is given (how many tool calls the history already shows), exactly like a scripted actor:
  triage    -> classify from keywords in the claim text
  coverage  -> search cover, search limitation, search exclusion, follow one cross-reference, then finish
  verifier  -> supported: true

Config (env):   STUB_LLM_DELAY_S=1.5   STUB_LLM_JITTER_S=0.3   (declared in docs/SLO.md)
Fault toggle:   POST /admin/fault {"mode": "error"|"timeout"|"slow"|"ok", "rate": 1.0}   (demo item 8)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

DELAY = float(os.getenv("STUB_LLM_DELAY_S", "1.5"))
JITTER = float(os.getenv("STUB_LLM_JITTER_S", "0.3"))

app = FastAPI(title="Stub LLM provider")
_fault = {"mode": "ok", "rate": 1.0}
_stats = {"requests": 0, "errors": 0, "in_flight": 0, "peak_in_flight": 0}


# ----------------------------------------------------------------------------- scripted behaviour
def _sentence(text: str) -> str:
    sents = re.split(r"(?<=[.;])\s+", text.strip())
    return sents[0].strip()


def triage_reply(user: str) -> dict:
    m = re.search(r"<claim_text>(.*?)</claim_text>", user, re.S)
    d = (m.group(1) if m else user).lower()
    fm = re.search(r"Structured fields: (\{.*?\})\n", user, re.S)
    try:
        fields = json.loads(fm.group(1)) if fm else {}
    except Exception:
        fields = {}
    commercial = bool(re.search(r"deliver|swiggy|zomato|uber|\bola\b|taxi|fare|passenger|for hire|reward|courier", d))
    if re.search(r"stolen|theft|was gone|it was gone", d):
        ct = "theft"
    elif re.search(r"total loss|burnt out|written off|beyond repair|fully submerged", d):
        ct = "total_loss"
    elif fields.get("third_party_involved") and fields.get("injuries"):
        ct = "third_party"
    else:
        ct = "own_damage"
    use = "food delivery (carrying goods for reward)" if "deliver" in d else (
        "carrying passengers for hire" if commercial else "private use")
    return {"claim_type": ct, "confidence": 0.8,
            "entities": {"vehicle_use_at_time_of_loss": use, "commercial_use": commercial, "damaged_parts": [],
                         "third_party_involved": bool(fields.get("third_party_involved")),
                         "injuries": bool(fields.get("injuries")), "intoxication_mentioned": False,
                         "driver": "insured", "cause": "theft" if ct == "theft" else "accident"},
            "summary": d[:120]}


def _parse_history(user: str):
    hist = user.split("History of this run:", 1)[-1]
    n_actions = len(re.findall(r"^\[attempt \d+\] Action:", hist, re.M))
    clauses: dict[str, dict] = {}
    for m in re.finditer(r"^  - ([A-Z0-9-]+) \| (.*?) \| type=(\w+)(?: \| score=[^|]*)? \| refs=(\[.*?\])\n    (.*)$", hist, re.M):
        cid, section, stype, refs, text = m.groups()
        clauses[cid] = {"section": section, "type": stype, "refs": re.findall(r"'([^']+)'", refs), "text": text}
    return n_actions, clauses


# Clauses are chosen by what their text says, not by hard-coded IDs, so the stub keeps working when the
# corpus is re-parsed and clause IDs change.
CLAUSE_ID_RE = re.compile(r"[A-Z0-9][A-Z0-9-]{1,80}")
OWN_DAMAGE_COVER = ["indemnify the insured against loss or damage to the vehicle"]
THIRD_PARTY_COVER = ["indemnify", "death of or bodily injury to any person"]
USE_LIMITATION = ["social domestic and pleasure", "hire or reward"]
USE_EXCEPTION = ["otherwise than in accordance with the limitations as to use"]


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def _find(clauses: dict, phrases: list[str]):
    """First clause whose text contains every phrase; returns (clause_id, sentence holding the first phrase)."""
    for cid, c in clauses.items():
        plain = _plain(c["text"])
        if all(_plain(p) in plain for p in phrases):
            key = _plain(phrases[0])
            sent = next((x for x in re.split(r"(?<=[.;])\s+", c["text"]) if key in _plain(x)), _sentence(c["text"]))
            return cid, sent.strip()
    return None, None


def coverage_reply(user: str) -> dict:
    n, clauses = _parse_history(user)
    m = re.search(r"<claim_text>(.*?)</claim_text>", user, re.S)
    d = (m.group(1) if m else "").lower()
    commercial = bool(re.search(r"deliver|swiggy|zomato|uber|\bola\b|taxi|fare|passenger|for hire|reward|courier", d))
    search = lambda q, st: {"thought": "search", "action": "search_corpus", "args": {"query": q, "k": 5, "section_type": st}}
    if n == 0:
        return search("loss of or damage to the vehicle insured by accidental external means", "cover")
    if n == 1:
        return search("vehicle used for " + ("hire or reward delivery" if commercial else "private use") +
                      "; limitations as to use, permitted purposes", "limitation")
    if n == 2:
        return search("exclusions general exceptions used otherwise than in accordance with limitations as to use", "exclusion")
    if n == 3:
        for c in clauses.values():
            for ref in c["refs"]:
                if CLAUSE_ID_RE.fullmatch(ref) and ref not in clauses:
                    return {"thought": f"follow ref {ref}", "action": "get_clause", "args": {"clause_id": ref}}
    cits, applies = [], "within_cover"
    pos = "The loss appears to fall within the own-damage cover."
    cid, sup = _find(clauses, OWN_DAMAGE_COVER)
    if cid is None:
        cid, sup = _find(clauses, THIRD_PARTY_COVER)
    if cid:
        cits.append({"clause_id": cid, "support": sup})
    lim_id, lim_sup = _find(clauses, USE_LIMITATION)
    if commercial and lim_id:
        applies = "outside_cover"
        cits.append({"clause_id": lim_id, "support": lim_sup})
        exc_id, exc_sup = _find(clauses, USE_EXCEPTION)
        if exc_id:
            cits.append({"clause_id": exc_id, "support": exc_sup})
        pos = ("Own-damage cover would apply to accidental damage, but the vehicle appears to have been used for a "
               "purpose outside the Limitations as to Use, and the General Exceptions exclude loss while so used.")
    return {"thought": "have cover and modifying provisions", "action": "finish",
            "answer": {"applies": applies, "conflict": False, "coverage_position": pos, "citations": cits}}


def respond(messages: list[dict]) -> dict:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if "Triage Agent" in system:
        return triage_reply(user)
    if "Coverage Agent" in system:
        return coverage_reply(user)
    if "verify a coverage statement" in system:
        return {"supported": True, "unsupported_claims": []}
    return {}


# ----------------------------------------------------------------------------- HTTP surface
@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    _stats["requests"] += 1
    _stats["in_flight"] += 1
    _stats["peak_in_flight"] = max(_stats["peak_in_flight"], _stats["in_flight"])
    try:
        if _fault["mode"] != "ok" and random.random() < _fault["rate"]:
            if _fault["mode"] == "error":
                _stats["errors"] += 1
                return JSONResponse({"error": {"message": "stub: injected provider failure"}}, status_code=503)
            if _fault["mode"] == "timeout":
                await asyncio.sleep(3600)   # never answers; the gateway's own timeout must save us
            if _fault["mode"] == "slow":
                await asyncio.sleep(DELAY * 6)
        await asyncio.sleep(max(0.0, DELAY + random.uniform(-JITTER, JITTER)))
        messages = body.get("messages") or []
        text = json.dumps(respond(messages))
        pt = sum(len(m.get("content", "")) for m in messages) // 4
        return {"id": "stub-" + str(int(time.time() * 1000)), "object": "chat.completion", "model": body.get("model", "stub"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": len(text) // 4, "total_tokens": pt + len(text) // 4}}
    finally:
        _stats["in_flight"] -= 1


@app.post("/admin/fault")
async def set_fault(request: Request):
    b = await request.json()
    mode = b.get("mode", "ok")
    if mode not in ("ok", "error", "timeout", "slow"):
        return JSONResponse({"detail": "mode must be ok|error|timeout|slow"}, status_code=422)
    _fault.update(mode=mode, rate=float(b.get("rate", 1.0)))
    return _fault


@app.get("/health")
def health():
    return {"status": "ok", "delay_s": DELAY, "jitter_s": JITTER, "fault": _fault}


@app.get("/metrics")
def metrics():
    lines = [f"stub_llm_requests_total {_stats['requests']}", f"stub_llm_errors_total {_stats['errors']}",
             f"stub_llm_in_flight {_stats['in_flight']}", f"stub_llm_peak_in_flight {_stats['peak_in_flight']}"]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
