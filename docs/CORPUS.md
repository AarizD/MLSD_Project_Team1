# Policy corpus and retrieval

The agent tools use `corpus.py`'s FAISS index whenever it is available. In `CORPUS_BACKEND=auto` mode the app warns
and falls back to the small sample corpus if no index exists. Set `CORPUS_BACKEND=faiss` to require the real index and
stop with a clear error if it is missing or incomplete.

## Documents in `data/policies/`

| File | What it is |
|---|---|
| `private-car-package-policy.pdf` | ICICI Lombard Private Car Package Policy wording (insurer-issued, 24 pages, two-column layout) |
| `schedule_clauses_transcription.pdf` | **Team-prepared transcription** of the standard schedule clauses `LIMITATIONS AS TO USE` and `DRIVERS CLAUSE`. Not an insurer-issued document. |
| `claims_procedure_summary.pdf` | **Team-prepared summary** of the motor claims procedure (intimation, survey, documents, theft and third-party steps). Not an insurer-issued document; figures in it have not been checked against the insurer. |

The two team-prepared files were generated with ReportLab so the schedule wording the 4.1 case depends on is in the
corpus. Replace them with insurer-issued documents if you obtain them; nothing else needs to change.

## How a PDF becomes chunks

1. **Text extraction.** `pdfplumber` detects two-column pages (almost no words cross the page midline) and reads the
   left column, then the right, so clauses are not interleaved. Single-column pages fall back to `pypdf`.
2. **Noise removal.** Page numbers, `CIN:` / `UIN:` footers and similar lines are dropped.
3. **Structure-aware chunking.** A new chunk starts at each heading (SECTION, CONDITION, GENERAL EXCEPTIONS, …).
   Each chunk is typed (`cover`, `exclusion`, `limitation`, `condition`, `definition`, `procedure`, …) and records its
   cross-references (`refs`).
4. **Length limit.** Clauses longer than `CHUNK_MAX_CHARS` (default 850 characters, under MiniLM's 256 word-piece
   limit) are split on sentence boundaries into `<clause_id>-PART<n>` with a one-sentence overlap. `get_clause` on the
   parent ID returns all parts joined, so following a reference always returns the whole clause.

Current corpus: **324 chunks** (315 policy wording, 6 claims procedure, 3 schedule clauses).

## Index and retrieval

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-d), L2-normalised, stored in `faiss.IndexIDMap2` over
  `IndexFlatIP` — inner product of unit vectors is **cosine similarity**.
- Query: hybrid score = 0.8 × semantic + 0.2 × keyword overlap (+0.03 exact-phrase bonus), optional `section_type`
  filter, top `k = 5`.
- The agents call retrieval through `corpus.PolicyClauseRetriever`, a LangChain `BaseRetriever` (`corpus.retrieve()`
  wraps it and returns plain dicts).

## Build or update the index (inside Docker)

```cmd
docker compose run --rm indexer
```

This runs `python corpus.py index` in a container with `data/policies/` mounted read-only and `index/` writable.
The build is **incremental**: each PDF is fingerprinted (SHA-256); unchanged documents keep their vectors, and only
chunks of new or changed documents are embedded and added (stale vectors are removed by ID). After a build,
`index/manifest.json` records the build time, model, chunk/document counts, chunks embedded in that run and throughput.
The worker exposes these as Prometheus metrics (`corpus_index_built_timestamp_seconds`, `corpus_index_chunks`,
`corpus_index_last_build_embedded_chunks`, `corpus_index_build_throughput_chunks_per_second`, …).

Restart the worker afterwards so it loads the new index: `docker compose restart worker`.

Running on the host also works (`pip install -r requirements.txt` then `python corpus.py index`), and
`python corpus.py search "Limitations as to Use" --section-type limitation` is a quick check.

## Evaluation labels

`eval/scenarios.json` cites clause IDs from this corpus (for example `SECTION-I-LOSS-OF-OR-DAMAGE-TO-THE-VEHICLE-INSURED`,
`LIMITATIONS-AS-TO-USE`, `GENERAL-EXCEPTIONS`). If you change the chunker or the documents, re-check that every
`must_cite` ID still exists in `index/chunks.jsonl` and every `support` quote is still found in it.
