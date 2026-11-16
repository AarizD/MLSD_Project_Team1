# Motor Insurance Claim Intake and Triage Service

A production multi-agent service for motor insurance First Notice of Loss (FNOL) intake, validation, policy-grounded coverage analysis, and queue routing (`fast_track`, `surveyor`, `investigation`, or human escalation).

> **Advisory only:** The system never approves or rejects claims or calculates settlement amounts. It produces a recommendation with clause-level citations for a human claims handler.

---

## Architecture & Key Components

- **Ingestion API & Web UI (`app/main.py`, `app/schemas.py`, `web/`)**: FastAPI service with schema/cross-field validation at the ingestion boundary, malformed record quarantine, per-client rate limiting, run caching, and a feedback endpoint.
- **Async Queue & Workers (`app/rq_worker.py`, `app/worker.py`)**: Redis + RQ decouples submission (`POST /claims` → `202` with `run_id`) from execution (`GET /runs/{run_id}`). Supports horizontal scaling (`--scale worker=3`).
- **Multi-Agent Reasoning (`app/agents.py`)**: LangGraph workflow (`guardrail → triage → coverage ↔ tool_call → verifier → finalize`) with bounded steps/tool calls/wall-clock time, retries with exponential backoff, and a circuit breaker.
- **Policy Corpus & RAG (`corpus.py`, `data/policies/`, `index/`)**: Section-aware PDF chunking (`pypdf`) + local `sentence-transformers/all-MiniLM-L6-v2` embeddings + FAISS index (`199` chunks across 3 ICICI Lombard Private Car documents).
- **Guardrails (`app/guardrails.py`)**: Prompt-injection defense on free-text inputs, toxicity detection on input/output, and output filtering.
- **Observability (`observability/`, `app/metrics.py`, `app/tracing.py`)**: Structured JSON logs with `run_id`, OpenTelemetry traces exported to **Jaeger**, **Prometheus** scraping API + scaled workers, and a pre-provisioned **Grafana** dashboard.

---

## Quick Start (Docker Compose)

### 1. Create `.env` and local secret files

```powershell
Copy-Item .env.example .env
Copy-Item secrets/llm_api_key.example secrets/llm_api_key.txt
Copy-Item secrets/grafana_admin_password.example secrets/grafana_admin_password.txt
```

- Edit `secrets/llm_api_key.txt` and paste your LLM provider API key (leave as-is if running in `stub` mode).
- Edit `secrets/grafana_admin_password.txt` and set your Grafana `admin` password.
- Edit `.env` to configure your LLM provider (for example, Gemini OpenAI-compatible endpoint):
  ```env
  LLM_PROVIDER=openai_compatible
  LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
  LLM_MODEL=gemini-3.5-flash-lite
  ```
  *(Set `LLM_PROVIDER=stub` to run without an external API key.)*

### 2. Start all 6 services

```powershell
docker compose up --build -d
docker compose ps
```

To scale agent workers:

```powershell
docker compose up -d --scale worker=3
```

### 3. Open the UIs

| Service | URL | Credentials / Notes |
|---|---|---|
| **Web Client & API** | http://localhost:8000 | Swagger docs at http://localhost:8000/docs, health at `/health` |
| **Grafana** | http://localhost:3000 | Username: `admin`, password from `secrets/grafana_admin_password.txt` |
| **Prometheus** | http://localhost:9090 | Scrapes `api:8000/metrics` and `worker:9101` |
| **Jaeger UI** | http://localhost:16686 | Select service `motor-claim-triage` |

---

## Corpus & Re-Indexing

The pre-built FAISS index (`index/`) for the ICICI Lombard Private Car corpus (`data/policies/`) is included in the repository. To rebuild after changing PDFs:

```powershell
pip install -r requirements.txt
python corpus.py index
python corpus.py search "Limitations as to Use" --section-type limitation
```

See [docs/CORPUS.md](docs/CORPUS.md) for details.

---

## Testing, Evaluation & Load Testing

```powershell
# Unit & guardrail tests (uses the deterministic sample corpus for stub assertions)
$env:CORPUS_BACKEND="sample"; python -m pytest -q

# Demo checks (guardrails, circuit breaker) against a running stack
python scripts/demo_checks.py all

# Batch evaluation (24 scenarios including Section 4.1)
pip install -r requirements-app.txt -r requirements-eval.txt
python eval/run_eval.py
```

For load testing with Locust against the LLM stub (`D6`, `A7`), see [docs/LOAD_TEST.md](docs/LOAD_TEST.md) and [docs/SLO.md](docs/SLO.md).

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System & agent topology, deployment boundaries
- [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) — External API & FNOL schema contract
- [docs/CORPUS.md](docs/CORPUS.md) — Corpus chunking & FAISS indexing
- [docs/EVALUATION.md](docs/EVALUATION.md) — Prompt versioning & TruLens evaluation
- [docs/GUARDRAILS.md](docs/GUARDRAILS.md) — Prompt injection, toxicity, and output guardrails
- [docs/SLO.md](docs/SLO.md) & [docs/LOAD_TEST.md](docs/LOAD_TEST.md) — Declared p95 SLO and Locust load test guide
- [docs/RUNBOOK.md](docs/RUNBOOK.md) & [docs/INCIDENT_LOG.md](docs/INCIDENT_LOG.md) — Operational runbook and incident log
- [docs/SECRETS.md](docs/SECRETS.md) — Docker Compose secret handling
