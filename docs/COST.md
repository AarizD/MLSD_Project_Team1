# Cost per run (L5): how to generate the numbers

This file contains **no results**. Three tools print cost; use whichever matches the traffic you measured.

## 0. Prices first

Cost is computed **at call time** from `LLM_PRICE_IN_PER_1K` and `LLM_PRICE_OUT_PER_1K` (USD per 1,000 prompt / completion
tokens). With the defaults (0) every cost is 0 and the tools say so. Put your provider's real list prices in `.env`
before generating anything you intend to quote. Do not quote the figures from `.env.loadtest.example`: they are
reference placeholders so the load test has non-zero cost, and the stub's token counts are character-based estimates.

For a defensible "cost per 100 runs", generate traffic against your **real provider** (the evaluation set is a good
source: `RUNS_DIR=eval/runs python eval/run_eval.py run --label cost` runs 24 scenarios) or a small real-traffic sample,
not the stub.

## 1. From the live stack (Prometheus): `scripts/report.py`

```bash
python scripts/report.py --window 1h
```

Prints, among the other run parameters: tokens per agent, cost per agent and per step, **COST PER 100 RUNS**, and the
**largest cost driver** (agent and step) with its share. It needs the stack up and traffic in the window.

## 2. From saved run records: `scripts/cost_report.py`

Every run is saved as `run_<id>.json`. In Docker the worker writes them to the `runs` volume:

```bash
mkdir -p runs_export
docker compose cp worker:/srv/runs/. ./runs_export        # add your -f files if you used the load-test overlay
python scripts/cost_report.py --runs-dir runs_export --out docs/cost_report.md
```

For an evaluation run, point it at the directory you gave `RUNS_DIR`:

```bash
python scripts/cost_report.py --runs-dir eval/runs
```

It prints cost per 100 runs, p50/p95/max cost per run, tokens per run, LLM calls per run, a per-agent table and a per-step
table (calls per run, prompt and completion tokens per call, share of cost), and names the largest driver. Runs stopped by a
guardrail before any model call are counted separately (they cost 0 and dilute the mean, as they should). To re-price the
same token counts under different prices: `--price-in 0.0003 --price-out 0.0012`.

## 3. From the evaluation harness

`python eval/run_eval.py run ...` includes `cost_per_100_runs_usd` and `mean_cost_usd` in its table (over the 24 scenarios,
which include runs the guardrails stop for free, so it is not a happy-path figure).

## 4. What to write in the report

- Cost per 100 runs **and the traffic mix it was measured on** (all-valid claims vs the evaluation mix are different numbers).
- The largest single driver (L5 asks for it). Read it from the by-step table, then say *why*: the coverage agent re-sends its
  growing scratchpad on every turn, so its prompt tokens grow each turn; check whether prompt tokens dominate cost.
- What you could change to reduce it (fewer coverage turns via `MAX_COVERAGE_TURNS`, shorter passages, caching: D7), and, if you
  tried one, the measured effect.
