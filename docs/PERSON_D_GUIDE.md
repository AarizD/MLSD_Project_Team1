# Person D guide: what to run to generate the numbers

Everything below prints or saves its own output. Nothing in this repo contains measured results; **you generate them on your
machine** (that is also what a viva expects). Commands run from the repo root.

| Deliverable | What produces it | Where the output lands |
| --- | --- | --- |
| L4 guardrails (injection, output filter, toxicity) | code in `app/guardrails.py`; **demo:** `python scripts/demo_checks.py guardrails` | printed PASS/FAIL per attack; `guardrail_flags` on every run |
| L5 per-step tokens and cost, cost per 100 runs, largest driver | `python scripts/report.py` (live) / `python scripts/cost_report.py` (saved runs) | printed; see `docs/COST.md` |
| L3 evaluation + before/after (and L2 prompt A/B) | `python eval/run_eval.py run` / `compare` | `eval/results/*.json` + `.md`; see `docs/EVALUATION.md` |
| D6 SLO + Locust, A7 `--scale worker=3` + bottleneck | `python scripts/load_steps.py --docker ...` | `docs/loadtest_results/`; see `docs/LOAD_TEST.md` |
| A5 / demo 8 circuit breaker and degradation | `python scripts/demo_checks.py circuit` | printed PASS/FAIL timeline |
| A6 per-agent metrics, uptime | `python scripts/report.py` (and Grafana) | printed |
| D8 runbook, incident log | `docs/RUNBOOK.md` (written), `docs/INCIDENT_LOG.md` (**you** write from real failures) | docs |

## Order to do it in

**1. Sanity (no Docker, 1 minute).** `pip install -r requirements-app.txt pytest && python -m pytest tests -q` should be all green.

**2. Commit the SLO before any measurement.** `git add docs/SLO.md && git commit -m "docs: declare p95 SLO"`. Then do not edit section 2.

**3. Evaluation with your real provider** (`docs/EVALUATION.md`): put provider settings and prices in `.env`; run the baseline,
make one deliberate change, run `compare`. Run `python scripts/cost_report.py --runs-dir eval/runs` for cost per 100 runs.

**4. Bring the stack up with the stub and load-test it** (`docs/LOAD_TEST.md`):
```bash
cp .env.loadtest.example .env.loadtest
docker compose --env-file .env.loadtest -f docker-compose.yml -f docker-compose.loadtest.yml up --build --scale worker=3
python scripts/load_steps.py --docker --workers 3 --label w3 --compose-file docker-compose.yml --compose-file docker-compose.loadtest.yml
```
then the `--scale` comparison and `python scripts/report.py`.

**5. Demo-item checks on the running stack:** `python scripts/demo_checks.py all` (guardrails, then circuit breaker; the
breaker check takes ~1 minute because it waits for the reset window).

**6. Write the incident log** from real failures (`docs/INCIDENT_LOG.md`). The first `docker compose up` from a clean clone is the
most likely place to find one. Capture evidence while it happens (the template says how).

## Things that will look wrong if you do not know them

- **`LLM_PROVIDER=stub` numbers are not model results.** The stub is scripted. The evaluation prints a banner when it is on.
- **The corpus is a stand-in.** `app/agents.py` searches a 12-clause hard-coded private-car corpus, not Person B's index
  (`corpus.py`). The evaluation scenarios' supporting quotes come from that stand-in. The brief wants a real private-car policy
  and the same insurer's claims procedure; `insurance_policy_pdf.pdf` in the repo is a **two-wheeler** policy. Fix the corpus,
  wire it in, then update the scenarios' `support` quotes and `must_cite` ids (see `docs/EVALUATION.md`).
- **TruLens is the record store, not the metric engine** (see `docs/EVALUATION.md`).
- **The circuit breaker state is shared in Redis** (`cb:llm`) when the stack runs under compose, because RQ forks a child per job.
- **Prometheus scrapes workers by DNS** (`observability/prometheus.yml`); each worker serves its metrics on port 9101.
- **The load-test overlay forces the stub**, so it cannot spend money; the base compose file uses whatever `.env` says.
- **There is still no top-level README.md.** The brief says it will be tested: a stranger must be able to run the system from it.
  Link to this guide from it.
