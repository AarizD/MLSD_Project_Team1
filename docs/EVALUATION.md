# Evaluation (L3) and prompt comparison (L2): how to run it

This file contains **no results**. The harness prints and saves the numbers when you run it.

## What exists

| Piece | What it is |
| --- | --- |
| `eval/scenarios.json` | 24 end-to-end scenarios (21 `core`, 3 `stretch`), each with the expected outcome and the **supporting passage as a verbatim quote**. Includes the section 4.1 case (`S02`, tag `section_4_1`), paraphrases of it, a negative control (own-business use is allowed), theft, total loss, third party, routing boundaries, out-of-corpus claims, injections, toxicity and an output-filter case. |
| `eval/metrics.py` | The four metrics plus outcome metrics. **The definitions are in the module docstring** (read it; the brief asks you to state how each is computed). |
| `eval/run_eval.py` | Batch CLI. `run` executes the set and writes results; `compare` prints a before/after table. |

`core` = what a correct system must pass. `stretch` = needs real reasoning (intoxication, unlicensed driver, property-only
third party); the scripted stub cannot pass these, a real model might. They are reported separately so the number you quote
is honest.

**Important: the supporting quotes currently come from the stand-in corpus in `app/agents.py`**, not from your real policy.
When Person B's index is wired in (and you have a private-car policy, see below), update each scenario's `support` quotes
and `must_cite` clause ids to the real passages. At Evaluation 1 the brief wants you to show the real passages the 4.1 case
depends on, so do this for at least `S02`-`S06` first.

## 1. Setup

```bash
pip install -r requirements-app.txt -r requirements-eval.txt
cp .env.example .env     # set LLM_PROVIDER=openai_compatible, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
                         # and LLM_PRICE_IN_PER_1K / LLM_PRICE_OUT_PER_1K (your provider's price list), so cost is non-zero
```

The harness reads `.env` exactly as the service does. If `LLM_PROVIDER=stub` it prints a banner saying the numbers measure
the stub, not a model.

## 2. Run the set

```bash
RUNS_DIR=eval/runs_baseline python eval/run_eval.py run --label baseline
```

Prints one PASS/FAIL line per scenario, then a table. Writes `eval/results/baseline.json` (every number and every
scenario's failures), `eval/results/baseline.md`, and stores each run in `eval/results/trulens.sqlite` (browse with the
command it prints). `RUNS_DIR=...` keeps the per-run records so you can also run `scripts/cost_report.py` on them.

How to read the table:

| row | meaning |
| --- | --- |
| outcome_accuracy | share of scenarios where every field the scenario pins down matches (status, route, claim type, verdict, required citations, flags, no forbidden text) |
| answer_rate | share that returned an answer (vs escalated); read next to the metrics below so a metric cannot look good by answering little |
| escalation_precision / recall | when it escalated, was it expected? did it escalate every time it should have? (F7) |
| support_hit_rate | the labelled supporting quote appears in a **cited** passage |
| context_relevance / context_recall | retrieval precision / recall against the labelled support (recall is the 4.1 detector: did retrieval surface the restriction?) |
| groundedness | share of answer sentences supported by the cited passages |
| faithfulness | citations verbatim in the passage and the verdict consistent with the cited clause types |
| hallucination_rate_draft / _returned | fabricated citations in the model's first draft / in what reached the caller. `returned` above 0 means the verifier let something through |
| cost_per_100_runs_usd, mean_latency_ms, mean_llm_calls | from the run records, priced at call time |

The section 4.1 line at the bottom says PASS/FAIL, what was cited, and the context recall for that case.

**TruLens:** each run is stored as a TruLens record (inputs, outputs, and the scores computed by `eval/metrics.py`). TruLens
is the record store and dashboard here; it does **not** compute the four metrics (TruLens 2.x runs in OpenTelemetry mode,
where its old feedback-function API is unsupported). Say that in the report rather than implying otherwise.

## 3. The deliberate change and the before/after (L3)

Pick **one** change you can defend, run the set before and after, and compare. Examples, each a single `--set`:

| Change | Command (the "after" run) |
| --- | --- |
| Prompt variant (also satisfies the L2 prompt comparison) | `--set COVERAGE_PROMPT=coverage-v1` vs the default `coverage-v2` |
| Remove the modifier-search requirement (step 2 of section 5) | `--set ENFORCE_MODIFIER_SEARCH=false` |
| Stricter weak-retrieval escalation | `--set WEAK_RETRIEVAL_THRESHOLD=0.25` |
| Fewer verifier retries | `--set MAX_VERIFY_ATTEMPTS=1` |

```bash
RUNS_DIR=eval/runs_before python eval/run_eval.py run --label before
RUNS_DIR=eval/runs_after  python eval/run_eval.py run --label after --set ENFORCE_MODIFIER_SEARCH=false
python eval/run_eval.py compare eval/results/before.json eval/results/after.json --out eval/results/before_after.md
```

`compare` prints all four metrics (plus outcome rows) with the delta. `--scope core` restricts it to the `core` scenarios.
Report all four before and after, state how each is computed (point at `eval/metrics.py`), and say what you changed and why.

Tips: LLM output varies, so for a defensible claim run each side 2-3 times (different `--label`s) and look at the spread
before you attribute a difference to the change. The most informative changes for this brief are the ones that affect
whether the restriction (e.g. `LIM-USE-1`) is retrieved: look at `context_recall` and `S02`.

To exercise the verifier on purpose (demo item 6) with the stub: `--set STUB_MODE=hallucinate` makes the scripted model cite a
non-existent clause on its first attempt; `hallucination_rate_draft` goes up and `_returned` should stay 0.

## 4. Prompt comparison (L2)

Same command with `--set COVERAGE_PROMPT=coverage-v1` and `coverage-v2` (and likewise `TRIAGE_PROMPT`, `VERIFIER_PROMPT`).
`compare` tells you which is better on which metric and by how much. The prompt version is recorded on every step of every
run (`prompt_version`).

## 5. Things this evaluation does not cover (say so in the report)

- It runs the agent graph **in-process**: it does not exercise the API, the queue, rate limiting or caching.
- Metrics are deterministic lexical / reference-based proxies, not an LLM judge. Groundedness in particular is lenient on
  paraphrase and strict on invented specifics.
- The scenario set is small (24) and hand-written; confidence intervals on the rates are wide.
- The supporting passages are from whichever corpus the agent is wired to (see the note above).
