# Connect the running agents to the policy corpus

The agent tools now use `corpus.py`'s FAISS index whenever it is available. In `CORPUS_BACKEND=auto` mode, the
app warns and uses the small sample corpus if no index exists. Set `CORPUS_BACKEND=faiss` to require the real index
and stop with a clear error if it is missing or incomplete.

## Build or update the index

The repository includes the **ICICI Lombard Private Car Package Policy** corpus in `data/policies/`:
- `private-car-package-policy.pdf` — ICICI Lombard Private Car Package Policy Wording (Product Code 3001)
- `icici_lombard_policy_schedule_clauses.pdf` — Standard Private Car Policy Schedule clauses (`LIMITATIONS AS TO USE` and `DRIVERS CLAUSE`)
- `icici_lombard_claims_procedure.pdf` — ICICI Lombard Motor Claims Procedure (intimation, cashless/reimbursement, surveyor, document checklist, theft & third-party steps)

The pre-built FAISS index (`199` chunks) lives in `index/` (`corpus.faiss`, `chunks.jsonl`, `documents.json`) and is committed to the repository so `docker compose up` works immediately from a clean clone.

If you add or modify PDFs in `data/policies/`, rebuild the index on the host:

```powershell
python -m pip install -r requirements.txt
python corpus.py index
python corpus.py search "Limitations as to Use" --section-type limitation
```

## Run the app

For Docker Compose, set `CORPUS_BACKEND=faiss` in `.env` after building the index, then start or rebuild the stack:

```powershell
docker compose up --build
```

Compose mounts `index/` read-only into the worker. The worker loads the MiniLM model into a named Docker volume during
startup, so the first start needs internet access and may take a while. Once healthy, jobs reuse the loaded FAISS
index and model. Rebuild the index on the host and recreate the worker to load changed policy files.

For running the Python app directly on the host, set `CORPUS_BACKEND=faiss` in `.env` and run the app from the
project directory. The host needs the dependencies in `requirements.txt` as well as `requirements-app.txt`.

The evaluation labels in `eval/scenarios.json` have been updated to match the indexed ICICI Lombard corpus quotes.
