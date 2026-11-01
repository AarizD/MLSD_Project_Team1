"""Guardrails (L4): prompt-injection defence, toxicity detection, output filtering.

Three layers, all local and dependency-free so they work offline and add ~no latency:

1. INPUT  - `inspect_input(fnol)` scans EVERY string field that can reach a model (not only the
            obvious free-text ones: policy_number / fir_number are free strings that the agents
            serialise into their prompts). Injection is scored, not matched on one regex.
2. PROMPT - `sanitize_for_prompt(text)` neutralises delimiter tokens (e.g. a literal </claim_text>)
            so text that scores below the injection threshold still cannot break out of the data
            tags the prompts wrap it in.
3. OUTPUT - `filter_output(text)` blocks anything that reads as a claim determination or a payment
            promise (the system is advisory only), and anything toxic.

Why agents are more exposed than pipelines: injected text can reach a model that holds tools. Two
rules below (`tool_call_json`, `tool_abuse`) target that specifically: a claimant who writes a
JSON `{"action": "finish", ...}` into the description is trying to make the coverage agent emit a
forged answer.

Limits (stated honestly in docs/GUARDRAILS.md): regex scoring catches known attack shapes, not
novel semantic ones; the toxicity lexicon is deliberately small. Both have a plug-in point
(`set_toxicity_scorer`) for a classifier, and the verifier + output filter remain the backstop.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

INJECTION_THRESHOLD = float(os.getenv("INJECTION_THRESHOLD", "0.6"))
TOXICITY_MODEL_THRESHOLD = float(os.getenv("TOXICITY_MODEL_THRESHOLD", "0.8"))

# Every FNOL string field that can end up inside a model prompt.
MODEL_REACHABLE_FIELDS = ("description", "place_of_loss", "garage_name", "insured_name",
                          "policy_number", "fir_number", "vehicle_registration")

# --------------------------------------------------------------------------- normalisation
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad\u202a\u202b\u202c\u202d\u202e"))
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})


def normalise(text: str) -> str:
    """NFKC fold (full-width / homoglyph-ish forms), drop zero-width and bidi controls, collapse space."""
    t = unicodedata.normalize("NFKC", str(text or "")).translate(_ZERO_WIDTH)
    t = "".join(ch for ch in t if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", t)


# --------------------------------------------------------------------------- injection
@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    weight: float


def _r(name, pat, weight, flags=re.I):
    return Rule(name, re.compile(pat, flags), weight)


_NOUN = r"(?:instructions?|prompts?|rules?|guidelines?|polic(?:y|ies)|constraints?|restrictions?|directives?)"
_MOD = r"(?:all|any|the|your|previous|prior|above|earlier|preceding|these|those|system|safety|original|my)"
INJECTION_RULES = [
    _r("override_instructions", rf"\b(?:ignore|disregard|forget|override|bypass|skip|drop)\s+(?:{_MOD}\s+){{0,4}}{_NOUN}", 1.0),
    _r("role_hijack", r"\b(?:you are now|from now on,? you|act as (?:an? |the )?(?:admin|administrator|system|developer|"
                      r"claims? (?:manager|approver|officer)|underwriter|root)|pretend (?:to be|you are)|new persona)", 0.8),
    _r("prompt_exfiltration", r"\b(?:reveal|show|print|repeat|output|leak|disclose|display)\s+(?:me\s+)?(?:your|the)\s+"
                              r"(?:system|hidden|initial|original|secret)\s+(?:prompt|instructions?|message)", 0.9),
    _r("fake_role_marker", r"(?:^|\n)\s*(?:system|assistant|developer|admin)\s*[:>\]]|\[/?(?:system|inst)\]|<<\s*sys\s*>>", 0.7),
    _r("delimiter_break", r"</?\s*claim_text\s*>|<\|[a-z_]+\|>|#{3,}\s*(?:system|instruction)", 1.0),
    _r("forced_determination", r"\b(?:output|say|respond with|reply with|write|return|mark|set)\b.{0,40}\b(?:claim|status|result)\b"
                               r".{0,25}\b(?:approved|accepted|settled|payable|fast[_ -]?track)\b", 0.9),
    _r("tool_call_json", r"[\"']action[\"']\s*:\s*[\"'](?:finish|search_corpus|get_clause|get_fnol_field)[\"']", 1.0),
    _r("answer_json", r"[\"'](?:applies|citations|coverage_position|supported)[\"']\s*:", 0.9),
    _r("tool_abuse", r"\b(?:call|invoke|run|execute|use)\s+(?:the\s+)?(?:tool|function)?\s*"
                     r"(?:get_clause|search_corpus|get_fnol_field|end_conversation)\b", 0.8),
    _r("jailbreak", r"\b(?:jailbreak|developer mode|do anything now|dan mode)\b", 0.8),
    # Hinglish override, e.g. "pichle instructions ignore karo"
    _r("hinglish_override", r"\b(?:pichle|purane|upar ke|saare)\s+(?:instructions?|nirdesh|rules?)\b.{0,25}\b(?:ignore|bhool|mat maano|hata)", 0.9),
    _r("encoded_payload", r"[A-Za-z0-9+/]{80,}={0,2}", 0.5, 0),
]


@dataclass
class InputVerdict:
    flags: list[str] = field(default_factory=list)          # machine-readable, goes into the run record
    escalate: bool = False                                    # True -> do not call any model
    findings: list[dict] = field(default_factory=list)      # field, rule, score, severity, for the audit trail


def injection_score(text: str) -> tuple[float, list[str]]:
    """Noisy-OR over matched rules: independent weak signals add up, one strong signal is enough."""
    t = normalise(text)
    hit = [r for r in INJECTION_RULES if r.pattern.search(t)]
    p = 1.0
    for r in hit:
        p *= 1.0 - r.weight
    return round(1.0 - p, 3), [r.name for r in hit]


def sanitize_for_prompt(text: str) -> str:
    """Neutralise tokens that could close the data tag or open a chat turn. Content is otherwise kept."""
    t = normalise(text)
    t = re.sub(r"</?\s*claim_text\s*>", "[tag removed]", t, flags=re.I)
    t = re.sub(r"<\|[a-z_]+\|>", "[token removed]", t, flags=re.I)
    return t


# --------------------------------------------------------------------------- toxicity
# Deliberately small. Threat patterns are "severe" (a human must read it: possible safety issue);
# plain profanity / insults are "mild" (claimants are often angry; do not block their claim).
_PROFANITY = ("fuck", "shit", "bastard", "bitch", "asshole", "bullshit", "crap", "damn", "piss", "dick", "wtf")
_INSULTS = ("idiot", "stupid", "moron", "useless", "incompetent", "scumbag", "jerk", "loser")
_PROFANITY_RE = re.compile(r"\b(?:" + "|".join(_PROFANITY + _INSULTS) + r")(?:s|ing|ed|er|ers)?\b", re.I)
def _obfuscated(word: str) -> str:
    """'fuck' -> f, then each later letter may be replaced by a symbol (f*ck, f@ck, s#it), separators allowed."""
    return word[0] + "".join(rf"[\W_]{{0,2}}(?:{c}|[*@#$])" for c in word[1:])


_STAR_RE = re.compile(r"\b(?:" + "|".join(_obfuscated(w) for w in ("fuck", "shit", "bitch")) + r")\w*", re.I)
_THREAT_RE = re.compile(
    r"\b(?:i|we)(?:'ll|\s+will|\s+am going to|'m going to|\s+gonna|\s+shall)\s+(?:kill|burn|bomb|shoot|stab|hurt|"
    r"destroy|murder|blow up|beat)\b|"
    r"\b(?:kill|murder|shoot|stab|beat up|burn down|blow up)\s+(?:you|them|him|her|your|the\s+(?:agent|staff|office|"
    r"surveyor|handler|garage))\b|"
    r"\bdeath threat\b|\byou(?:'ll|\s+will)\s+regret\b", re.I)

_toxicity_scorer: Callable[[str], float] | None = None


def set_toxicity_scorer(fn: Callable[[str], float] | None) -> None:
    """Plug a classifier in (returns 0..1). Used in addition to the lexicon, never instead of the threat rules."""
    global _toxicity_scorer
    _toxicity_scorer = fn


def toxicity(text: str) -> tuple[str | None, str | None]:
    """Returns (severity, reason) where severity is None | 'mild' | 'severe'."""
    t = normalise(text)
    if _THREAT_RE.search(t):
        return "severe", "threat"
    if _toxicity_scorer is not None:
        try:
            if _toxicity_scorer(t) >= TOXICITY_MODEL_THRESHOLD:
                return "severe", "model"
        except Exception:
            pass  # a broken classifier must never take the pipeline down; lexicon below still runs
    if _PROFANITY_RE.search(t) or _STAR_RE.search(t):
        return "mild", "profanity_or_insult"
    return None, None


# --------------------------------------------------------------------------- input entry point
def inspect_input(fnol: dict) -> InputVerdict:
    v = InputVerdict()
    for f in MODEL_REACHABLE_FIELDS:
        val = fnol.get(f)
        if not val:
            continue
        score, rules = injection_score(str(val))
        if score >= INJECTION_THRESHOLD:
            v.flags.append(f"prompt_injection:{f}")
            v.findings.append({"field": f, "check": "injection", "score": score, "rules": rules, "severity": "severe"})
            v.escalate = True
        sev, why = toxicity(str(val))
        if sev:
            v.flags.append(f"toxicity_{sev}:{f}")
            v.findings.append({"field": f, "check": "toxicity", "reason": why, "severity": sev})
            if sev == "severe":
                v.escalate = True
    return v


def check_input(fnol: dict) -> list[str]:
    """Backwards-compatible: just the flags."""
    return inspect_input(fnol).flags


# --------------------------------------------------------------------------- output filter
_ACT = r"(?:approv|reject|deny|denie|declin|repudiat|settl|sanction|accept|admit)"
DETERMINATION_RE = re.compile(
    r"\b(?:"
    rf"(?:claim|request|application)\s+(?:is|has been|was|will be|stands|shall be)\s+(?:hereby\s+)?{_ACT}\w*|"
    rf"(?:is|are)\s+hereby\s+{_ACT}\w*|"
    rf"{_ACT}(?:e|es|ed|ing)?\s+(?:the|this|your)\s+claim|"
    r"(?:we|the company|the insurer|insurer|company)\s+(?:will|shall|hereby|are going to)\s+"
    r"(?:pay|reimburse|settle|compensate|indemnify you|approve|reject|deny|decline)|"
    r"(?:you|the insured|the claimant)\s+(?:will|shall|are entitled to|are eligible to)\s+"
    r"(?:receive|get|be paid|be reimbursed|be compensated)|"
    r"(?:payable|settlement)\s+amount|amount\s+(?:payable|to be paid)|"
    r"will be\s+(?:paid|credited|reimbursed|settled)|"
    r"guarantee[sd]?\s+(?:cover|coverage|payment|settlement|reimbursement)|"
    r"(?:cover|coverage|payment|settlement|reimbursement)\s+(?:is|are|will be)\s+guarantee[sd]?|"
    r"pay\s+(?:the\s+)?full\s+amount|"
    r"no need (?:for|to)\s+(?:a\s+|the\s+)?(?:human|surveyor|handler|investigation)|"
    r"skip\s+(?:the\s+)?(?:surveyor|investigation|human review)"
    r")\b", re.I)

BLOCKED_DETERMINATION = "[Removed: the draft contained a claim determination the system is not authorised to make.]"
BLOCKED_TOXIC = "[Removed: the draft contained abusive or threatening language.]"


def filter_output(text):
    """Returns (safe_text, flags). Applied to every human-readable string the service returns."""
    if not text or not isinstance(text, str):
        return text, []
    t = normalise(text)
    if DETERMINATION_RE.search(t):
        return BLOCKED_DETERMINATION, ["unauthorised_determination_blocked"]
    sev, _ = toxicity(t)
    if sev:
        return BLOCKED_TOXIC, ["toxic_output_blocked"]
    return text, []


def filter_tree(obj, source_text: str = ""):
    """filter_output over every string inside a dict/list (e.g. the triage entities the model wrote).

    `source_text` is what the claimant wrote. A flagged string that is a verbatim fragment of it is the claimant's own
    wording echoed back (the triage summary quotes the description): it is still redacted, but its flag gets an
    `_echo` suffix so the caller can treat it as informational instead of as the system misbehaving. Text the model
    wrote itself keeps the plain flag and is escalated.
    """
    flags: list[str] = []
    src = norm_text(source_text)

    def walk(x):
        if isinstance(x, str):
            safe, f = filter_output(x)
            if f and src and norm_text(x) and norm_text(x) in src:
                f = [name + "_echo" for name in f]
            flags.extend(f)
            return safe
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(obj), flags


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%]+", " ", normalise(t).lower())).strip()


def sanitize_fields(fields: dict) -> dict:
    """sanitize_for_prompt over the string values of the structured-fields dict sent to the models."""
    return {k: sanitize_for_prompt(v) if isinstance(v, str) else v for k, v in fields.items()}
