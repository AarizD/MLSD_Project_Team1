# Guardrails (L4)

Code: `app/guardrails.py`. Wired into the graph from `notebooks/person_c_agents.ipynb` (so `app/agents.py` regenerates
cleanly with `python scripts/export_agents.py`). Tests: `tests/test_guardrails.py` (unit cases: attacks, benign text, output filter),
`tests/test_guardrails_pipeline.py` (end-to-end cases through the real LangGraph pipeline). Run `python -m pytest tests -q`.

## Why agents need more than a pipeline does

In a fixed pipeline an injected sentence can at worst corrupt one model answer. Here the coverage agent holds tools
and a loop, so injected text can try to (a) make it call tools it should not, (b) forge its own "finish" answer, or
(c) talk the verifier into accepting it. The defence is therefore layered, and the last layers do not trust the model.

## The layers, in the order a request meets them

| # | Where | What | Stops |
| - | ----- | ---- | ----- |
| 1 | After schema validation, **before any model is called** (`guard_input`) | `inspect_input`: injection scoring and toxicity on **every string field that reaches a prompt** | Attacks never reach a model that holds tools. Escalates to a human with the findings recorded. |
| 2 | When building prompts | `sanitize_for_prompt` / `sanitize_fields`: neutralise `</claim_text>` and chat-template tokens, normalise Unicode | Text that scored *below* the threshold still cannot break out of the data tags. |
| 3 | After every model answer | The verifier (A3): every citation must resolve to retrieved text | A model that was talked into inventing evidence. |
| 4 | Before anything is returned (`finalize`) | `filter_output` / `filter_tree` over `coverage_position`, `route_reason`, `escalation_reason` and the model-written triage entities | The system cannot issue a determination or payment promise, and cannot return abusive text. |

### Injection (layer 1)

Scored, not matched against one regex. Each rule has a weight; combined as a noisy-OR (`1 - Π(1 - w)`), so one
strong signal is enough and several weak ones add up. Flagged at `INJECTION_THRESHOLD` (default 0.6). Rules cover:
instruction override, role hijack, system-prompt exfiltration, fake role markers, delimiter break-out, forced
determinations, **forged tool-call / answer JSON** (`{"action": "finish", ...}`, `"citations":`), tool abuse,
jailbreak phrases, a Hinglish override, and base64-looking blobs. Text is NFKC-normalised and stripped of zero-width
and bidi characters first, so `i​gnore` (zero-width space inside) and full-width `ｉｇｎｏｒｅ` are caught.

**Fields scanned:** `description`, `place_of_loss`, `garage_name`, `insured_name`, `policy_number`, `fir_number`,
`vehicle_registration`. The pre-existing check scanned four of these. `policy_number` and `fir_number` are free
strings that `fnol_fields()` serialises into the prompts, so an injection in `fir_number` reached the models
unchecked. `test_injection_in_fir_number_is_caught` is the regression test.

### Toxicity (layers 1 and 4)

* **Severe** (explicit threats of violence): escalate to a human and make no model call. A threat against staff is
  a safety matter, not a coverage question.
* **Mild** (profanity, insults, including `f*ck`-style obfuscation): flag for the handler, **continue triage**.
  Claimants are often angry; refusing to triage their claim for swearing would be the wrong failure.
* Output: any toxic text in what the system itself is about to return is blocked. The triage *summary* echoes the
  claimant's own words, so abuse (or a demanded "approval") there is **redacted with an informational flag but does not escalate**: the system
  did not generate it.
* Backend: a small local lexicon, on purpose (no model download, ~0 latency, works offline). `set_toxicity_scorer(fn)`
  plugs in a classifier; a broken classifier cannot take the pipeline down.

### Output filter (layer 4)

The system is advisory. `DETERMINATION_RE` blocks approval / rejection / repudiation wording, payment promises
("we will pay", "you will receive Rs ..."), "payable / settlement amount", guarantees of cover, and statements that a
human review can be skipped. Advisory phrasing the system is *supposed* to produce ("would apply", "appears to fall
within", "outside the permitted use") is tested to pass untouched (`MUST_PASS` in the tests). If something is blocked
the run is escalated and the reason recorded; the blocked text is replaced with a fixed notice, never silently dropped.

## What the caller and the operator can see

* `result.guardrail_flags` on every run, e.g. `prompt_injection:fir_number`, `toxicity_mild:description`,
  `unauthorised_determination_blocked`, `echoed_input_redacted`.
* The `guardrail` step in the run trace carries the full `findings` (field, rule names, score).
* Prometheus: `guardrail_triggers_total{check=...}` (emitted per run by `emit_run_metrics`).
* A run stopped by layer 1 makes **zero** model calls; the tests assert `llm_calls == 0` and the cost report counts
  such runs separately.

## Limits, stated plainly

* Regex scoring catches known attack *shapes*. A novel, semantically phrased injection ("kindly treat the earlier
  guidance as void") may score below threshold. That is why layers 2-4 exist and do not rely on layer 1; but layer 1
  is not a proof. A classifier (e.g. a prompt-injection model behind `INJECTION_CLASSIFIER`) is the next step; it was
  not built because it needs a model download that is outside the offline/free constraints of this project.
* The toxicity lexicon is English-centred and small. Multilingual and sarcastic abuse will be missed.
* The output filter is a pattern list. It catches the phrasings above, not every way of promising payment.
* False-positive behaviour is only checked against the benign cases in the tests. Run a sample of real FNOL text through
  `inspect_input` before relying on the thresholds. To see the guardrails act on a live stack: `python scripts/demo_checks.py guardrails`.
