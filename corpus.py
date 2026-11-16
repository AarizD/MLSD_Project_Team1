"""Person B corpus pipeline: PDF -> clause chunks -> MiniLM -> FAISS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PDF_DIR = Path(os.getenv("POLICY_PDF_DIR", str(ROOT / "data" / "policies")))
INDEX_DIR = Path(os.getenv("CORPUS_INDEX_DIR", str(ROOT / "index")))
META_PATH = INDEX_DIR / "chunks.jsonl"
STATE_PATH = INDEX_DIR / "documents.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
FAISS_PATH = INDEX_DIR / "corpus.faiss"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Basic descriptions shared with Person C's tool contract.
FNOL_FIELDS = {
    "policy_number": "Policy identifier printed on the insurance schedule.",
    "insured_name": "Name of the person or entity insured under the policy.",
    "vehicle_registration": "Registration number of the insured vehicle.",
    "date_of_loss": "Date on which the incident or damage occurred.",
    "date_of_report": "Date on which the claim was reported.",
    "place_of_loss": "Location where the incident occurred.",
    "estimated_loss_inr": "Estimated loss amount in Indian rupees.",
    "description": "Claimant's narrative describing the incident and damage.",
    "third_party_involved": "Whether another person or property was involved.",
    "injuries": "Whether anyone reported an injury.",
    "police_report_filed": "Whether a police report/FIR was filed, if applicable.",
    "cashless_requested": "Whether cashless repair/service was requested.",
}


HEADING_RE = re.compile(
    r"^\s*((?i:SECTION|CHAPTER|PART)\s+[IVX0-9A-Z.-]+[^\n]{0,140}"
    r"|IMT\.?\s*\d+[A-Z.]*\s+[^\n]{2,140}"
    r"|[A-Z][A-Z0-9 &/,()'–-]{4,100})\s*$"
)

# Page furniture: page numbers ("-3-", "-3"), the UIN footer (with or without page number / CIN) and a bare CIN line.
NOISE_LINE_RE = re.compile(
    r"^\s*(?:-?\d{1,3}-?(?:\s+CIN:\s*\S+)?|UIN:\s*IRDAN\S+.*|CIN:\s*\S+)\s*$"
)

CLAUSE_RE = re.compile(
    r"^\s*((?:\d+(?:\.\d+){0,4}|[a-zA-Z][.)]|\([a-zA-Z0-9ivx]+\)))[.)]?\s+(.{12,})$"
)

REF_PATTERNS = [
    re.compile(
        r"\bsubject to\s+(?:the\s+)?([^.;\n]{3,100})",
        re.I,
    ),
    re.compile(
        r"\b(?:as provided|as stated|as described|in accordance)"
        r"\s+in\s+([^.;\n]{3,100})",
        re.I,
    ),
    re.compile(
        r"\b(?:section|clause|condition|exception)\s+"
        r"(?:no\.?\s*)?[IVX0-9]+(?:\.\d+)*",
        re.I,
    ),
]


def sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def detect_type(heading: str, text: str) -> str:
    """Assign a basic category to a policy chunk."""
    s = (heading + " " + text[:350]).lower()

    if any(
        x in s
        for x in (
            "exclusion",
            "general exception",
            "not liable",
            "not be liable",
            "will not cover",
            "shall not cover",
            "excluded",
        )
    ):
        return "exclusion"

    if any(
        x in s
        for x in (
            "limitation",
            "limit of",
            "maximum",
            "excess",
            "deductible",
            "depreciation",
        )
    ):
        return "limitation"

    if any(x in s for x in ("definition", "means", "defined as")):
        return "definition"

    if any(
        x in s
        for x in (
            "condition",
            "condition precedent",
            "duties",
            "obligation",
            "notice",
        )
    ):
        return "condition"

    if any(
        x in s
        for x in (
            "claim procedure",
            "intimation",
            "documents required",
            "procedure",
        )
    ):
        return "procedure"

    if any(
        x in s
        for x in (
            "cover",
            "indemnify",
            "loss of or damage",
            "liability",
        )
    ):
        return "cover"

    return "general"


def extract_refs(text: str) -> list[str]:
    """Extract simple textual cross-references."""
    refs: list[str] = []

    for pattern in REF_PATTERNS:
        for match in pattern.finditer(text):
            value = re.sub(
                r"\s+",
                " ",
                match.group(0),
            ).strip(" .;,:\n")

            if value and value not in refs:
                refs.append(value[:180])

    return refs


def _column_text(page) -> str:
    """One pdfplumber page in reading order: the left column, then the right one.

    The policy wording is printed in two columns. Reading straight across the page interleaves them, which put
    General Exception 3(a) inside the Personal Accident chunk. A page counts as two-column when almost no word
    crosses the vertical centre line and both halves hold a real share of the words."""
    words = page.extract_words()
    if not words:
        return ""
    mid = page.width / 2
    crossing = sum(1 for w in words if w["x0"] < mid - 4 and w["x1"] > mid + 4)
    left = sum(1 for w in words if w["x1"] <= mid + 4)
    right = sum(1 for w in words if w["x0"] >= mid - 4)
    if crossing <= 0.03 * len(words) and left > 0.2 * len(words) and right > 0.2 * len(words):
        return "\n".join(
            page.crop(box).extract_text() or ""
            for box in ((0, 0, mid, page.height), (mid, 0, page.width, page.height))
        )
    return page.extract_text() or ""


def _open_plumber(path: Path):
    """pdfplumber handle for column-aware reading, or None (then plain pypdf text is used)."""
    try:
        import pdfplumber
    except ImportError:
        print("WARNING: pdfplumber not installed; two-column pages will be read straight across")
        return None
    return pdfplumber.open(str(path))


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    """Extract PDF text into section-aware chunks."""
    reader = PdfReader(str(path))
    plumber = _open_plumber(path)

    chunks: list[dict[str, Any]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []
    current_page = 1
    current_id = "INTRO"
    sequence = 0

    def flush() -> None:
        nonlocal sequence, current_lines

        text = re.sub(
            r"\s+",
            " ",
            " ".join(current_lines),
        ).strip()

        if len(text) >= 35:
            sequence += 1

            if current_id != "INTRO":
                clause_id = current_id
            else:
                clause_id = (
                    f"{path.stem.upper()}-P{current_page:03d}"
                    f"-C{sequence:03d}"
                )

            chunks.append(
                {
                    "clause_id": clause_id,
                    "section": current_heading,
                    "section_type": detect_type(current_heading, text),
                    "source_doc": path.name,
                    "page": current_page,
                    "refs": extract_refs(text),
                    "text": text,
                    "score": 0.0,
                    "document_hash": sha256(path),
                }
            )

        current_lines = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw = (_column_text(plumber.pages[page_num - 1]) if plumber else "") or page.extract_text() or ""

        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in raw.splitlines()
        ]
        lines = [
            line for line in lines
            if line and not NOISE_LINE_RE.match(line)
        ]

        for line in lines:
            heading = HEADING_RE.match(line)
            clause = CLAUSE_RE.match(line)

            if heading:
                # Merge consecutive heading lines (e.g., multi-line ALL-CAPS headings)
                if not current_lines and current_heading != "Introduction" and len(current_heading) < 120:
                    current_heading = f"{current_heading} {line.strip()}".strip()
                    current_id = re.sub(
                        r"[^A-Za-z0-9]+",
                        "-",
                        current_heading,
                    ).strip("-").upper()[:70]
                    continue

                flush()

                current_heading = line.strip()
                current_page = page_num

                current_id = re.sub(
                    r"[^A-Za-z0-9]+",
                    "-",
                    current_heading,
                ).strip("-").upper()[:70]

                current_lines = []

            elif clause and len(" ".join(current_lines)) > 900:
                flush()

                current_page = page_num

                current_id = re.sub(
                    r"[^A-Za-z0-9]+",
                    "-",
                    clause.group(1),
                ).strip("-").upper()

                current_lines = [line]

            else:
                if not current_lines:
                    current_page = page_num

                current_lines.append(line)

    flush()
    if plumber:
        plumber.close()

    # Avoid duplicate IDs caused by repeated headings.
    seen: dict[str, int] = {}

    for chunk in chunks:
        base = chunk["clause_id"]
        seen[base] = seen.get(base, 0) + 1

        if seen[base] > 1:
            chunk["clause_id"] = f"{base}-{seen[base]}"

    return [part for chunk in chunks for part in _split_long(chunk)]


# all-MiniLM-L6-v2 reads at most 256 word pieces (roughly 1,000-1,200 characters of policy text); anything after
# that is silently cut from the embedding. Clauses longer than this are split into parts at sentence boundaries,
# each part repeating the previous part's last sentence (a small overlap) and keeping the clause's heading and type.
# Bump when parsing/chunking changes: documents indexed by an older parser are re-parsed even if the PDF is unchanged.
PARSER_VERSION = "2-columns-850"
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "850"))
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[(\[A-Za-z0-9])")


def _split_long(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one over-long clause into parts of at most CHUNK_MAX_CHARS. IDs: <id>, <id>-PART2, <id>-PART3 ..."""
    text = chunk["text"]
    if len(text) <= CHUNK_MAX_CHARS:
        return [chunk]

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > CHUNK_MAX_CHARS:  # a single very long sentence: cut at a space
            cut = sentence.rfind(" ", 0, CHUNK_MAX_CHARS) or CHUNK_MAX_CHARS
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        pieces.append(sentence)

    parts: list[list[str]] = [[]]
    for piece in pieces:
        if parts[-1] and len(" ".join(parts[-1] + [piece])) > CHUNK_MAX_CHARS:
            overlap = parts[-1][-1] if len(parts[-1][-1]) <= 200 else ""
            parts.append([overlap] if overlap else [])
        parts[-1].append(piece)

    out = []
    for i, words in enumerate(parts, start=1):
        part = dict(chunk)
        part["text"] = " ".join(words).strip()
        part["clause_id"] = chunk["clause_id"] if i == 1 else f"{chunk['clause_id']}-PART{i}"
        part["part"], part["parts"], part["parent_id"] = i, len(parts), chunk["clause_id"]
        out.append(part)
    return out


def load_chunks() -> list[dict[str, Any]]:
    """Load saved chunks."""
    if not META_PATH.exists():
        return []

    return [
        json.loads(line)
        for line in META_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    """Save chunks as JSON Lines."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    META_PATH.write_text(
        "".join(
            json.dumps(chunk, ensure_ascii=False) + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )


def load_state() -> dict[str, Any]:
    """Load document fingerprint metadata."""
    if not STATE_PATH.exists():
        return {}

    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def build_index(document: Path | None = None) -> None:
    """
    Build or update the FAISS index.

    A full indexing run also removes deleted PDFs.
    A document-specific run updates only the selected PDF.
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run: pip install -r requirements.txt"
        ) from exc

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_chunks()
    previous_vec_ids = {c["vec_id"] for c in existing if "vec_id" in c}
    state = load_state()

    if document is not None:
        selected_pdf = document.resolve()

        if not selected_pdf.exists():
            raise SystemExit(f"PDF not found: {selected_pdf}")

        pdfs = [selected_pdf]
        full_scan = False

    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        full_scan = True

    if not pdfs and not full_scan:
        raise SystemExit("No PDF selected.")

    changed_docs: list[str] = []
    index_changed = False

    # Remove documents that no longer exist.
    # Only do this during a full corpus scan.
    if full_scan:
        current_names = {pdf.name for pdf in pdfs}
        old_names = set(state.keys())

        deleted_names = old_names - current_names

        if deleted_names:
            for name in sorted(deleted_names):
                print(f"REMOVE deleted document: {name}")

            existing = [
                chunk
                for chunk in existing
                if chunk["source_doc"] not in deleted_names
            ]

            for name in deleted_names:
                state.pop(name, None)

            index_changed = True

    # Process new or modified PDFs.
    for pdf in pdfs:
        digest = sha256(pdf)

        previous = state.get(pdf.name, {})

        if previous.get("sha256") == digest and previous.get("parser_version") == PARSER_VERSION:
            print(f"SKIP unchanged: {pdf.name}")
            continue

        print(f"Parsing: {pdf.name}")

        # Replace only this document's existing chunks.
        existing = [
            chunk
            for chunk in existing
            if chunk["source_doc"] != pdf.name
        ]

        parsed = parse_pdf(pdf)

        if not parsed:
            raise SystemExit(
                f"No extractable text found in {pdf.name}. "
                "The PDF may be scanned and need OCR."
            )

        existing.extend(parsed)

        state[pdf.name] = {
            "sha256": digest,
            "chunk_count": len(parsed),
            "build_time_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model": MODEL_NAME,
            "parser_version": PARSER_VERSION,
        }

        changed_docs.append(pdf.name)
        index_changed = True

        print(f"  extracted {len(parsed)} chunks")

    # If nothing changed, retain the existing index.
    if not index_changed:
        if FAISS_PATH.exists():
            print("Nothing changed; existing index is up to date.")
            return

        if not existing:
            raise SystemExit(
                f"No indexed chunks or PDFs found in {PDF_DIR}"
            )

    # Handle the case where every indexed document was deleted.
    if not existing:
        META_PATH.unlink(missing_ok=True)
        FAISS_PATH.unlink(missing_ok=True)
        STATE_PATH.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )

        print("No documents remain. The index has been cleared.")
        return

    # Incremental re-embedding (L1). Vectors sit in a FAISS IndexIDMap2 keyed by each chunk's vec_id, so when a
    # document changes only its old vectors are removed and only its new chunks are embedded. Unchanged documents
    # keep their vectors. An index written by an older version (no IDs) is rebuilt in full once.
    index, mode = None, "full"
    kept = [c for c in existing if "vec_id" in c]
    if FAISS_PATH.exists() and kept:
        try:
            candidate = faiss.read_index(str(FAISS_PATH))
        except Exception:
            candidate = None
        if candidate is not None and type(candidate).__name__.startswith("IndexIDMap"):
            index, mode = candidate, "incremental"
            stale = sorted(previous_vec_ids - {c["vec_id"] for c in kept})
            if stale:
                index.remove_ids(np.asarray(stale, dtype="int64"))
    if index is None:
        for chunk in existing:
            chunk.pop("vec_id", None)

    to_embed = [c for c in existing if "vec_id" not in c]
    next_id = max((c["vec_id"] for c in existing if "vec_id" in c), default=-1) + 1
    for offset, chunk in enumerate(to_embed):
        chunk["vec_id"] = next_id + offset

    started = time.perf_counter()
    model = None
    if to_embed or index is None:
        print(f"Generating embeddings for {len(to_embed)} chunk(s) ({mode})...")
        model = SentenceTransformer(MODEL_NAME)
    if index is None:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(model.get_sentence_embedding_dimension()))
    if to_embed:
        vectors = model.encode(
            [f"{chunk['section']}\n{chunk['text']}" for chunk in to_embed],
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        index.add_with_ids(np.asarray(vectors, dtype="float32"),
                           np.asarray([c["vec_id"] for c in to_embed], dtype="int64"))
    elapsed = time.perf_counter() - started

    if index.ntotal != len(existing):
        raise SystemExit(f"Index has {index.ntotal} vectors but {len(existing)} chunks; "
                         "delete index/ and run `python corpus.py index` to rebuild.")

    # Data freshness and pipeline throughput (D2): exported as Prometheus metrics by app/metrics.py.
    MANIFEST_PATH.write_text(json.dumps({
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "mode": mode,
        "documents": len(state),
        "chunks": len(existing),
        "embedded_this_run": len(to_embed),
        "embedding_seconds": round(elapsed, 3),
        "throughput_chunks_per_second": round(len(to_embed) / elapsed, 2) if to_embed and elapsed > 0 else None,
        "versions": {name: meta.get("sha256", "")[:12] for name, meta in sorted(state.items())},
    }, indent=2), encoding="utf-8")

    faiss.write_index(index, str(FAISS_PATH))
    save_chunks(existing)

    STATE_PATH.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )

    print(
        f"Indexed {len(existing)} chunks from "
        f"{len(state)} documents."
    )

    if changed_docs:
        print(f"Updated: {', '.join(changed_docs)}")

    if full_scan and deleted_names if "deleted_names" in locals() else False:
        print(f"Removed: {', '.join(sorted(deleted_names))}")


@lru_cache(maxsize=1)
def _load_search_assets():
    """Load FAISS, chunks and the embedding model."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: pip install -r requirements.txt"
        ) from exc

    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            "Index missing. Run `python corpus.py index` first."
        )

    index = faiss.read_index(str(FAISS_PATH))
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            "FAISS index and chunk metadata have different sizes. "
            "Run `python corpus.py index` to rebuild."
        )

    # FAISS returns vec_ids (IndexIDMap2) or, for an index built by an older version, row positions.
    by_id = {c["vec_id"]: c for c in chunks if "vec_id" in c} or dict(enumerate(chunks))
    return index, chunks, model, by_id


def warm_search_assets() -> None:
    """Load and cache the FAISS index and embedding model before handling jobs."""
    _load_search_assets()


def search_corpus(
    query: str,
    k: int = 5,
    section_type: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval using semantic similarity and weighted keywords."""
    if not query.strip() or k < 1:
        return []

    index, chunks, model, by_id = _load_search_assets()

    if not chunks:
        return []

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    # Retrieve a broad semantic candidate set before reranking.
    search_count = min(max(k * 10, 30), len(chunks))
    scores, ids = index.search(query_vector, search_count)

    stop_words = {
        "what", "when", "where", "which", "who", "why", "how",
        "the", "and", "for", "from", "with", "under", "into",
        "are", "is", "was", "were", "does", "do", "can", "a",
        "an", "of", "to", "in", "on", "or", "be", "being",
        "this", "that", "their", "there", "it",
    }

    terms = [
        term.lower()
        for term in re.findall(r"\b[a-zA-Z0-9]+\b", query)
        if len(term) > 2 and term.lower() not in stop_words
    ]

    # Give distinctive policy terms more weight than generic terms.
    weights = {}
    for term in terms:
        if term in {"exclusion", "exclusions", "excluded", "theft",
                    "constructive", "depreciation", "deductible",
                    "indemnity", "idv"}:
            weights[term] = 2.5
        else:
            weights[term] = 1.0

    candidates = {}
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id < 0:
            continue

        if int(chunk_id) not in by_id:
            continue
        item = dict(by_id[int(chunk_id)])

        if section_type and item.get("section_type") != section_type:
            continue

        heading = str(item.get("section", "")).lower()
        text = str(item.get("text", "")).lower()
        combined = heading + " " + text

        matched = [term for term in terms if term in combined]
        total_weight = sum(weights.values()) or 1.0
        keyword_score = sum(weights[t] for t in matched) / total_weight

        # Keep semantic similarity primary; keywords provide a modest rerank.
        semantic_score = max(0.0, min(1.0, float(score)))
        final_score = 0.80 * semantic_score + 0.20 * keyword_score

        # Exact phrase matches provide a small additional signal.
        normalized_query = " ".join(terms)
        if normalized_query and normalized_query in combined:
            final_score += 0.03

        item["semantic_score"] = round(semantic_score, 4)
        item["keyword_score"] = round(keyword_score, 4)
        item["score"] = round(min(1.0, final_score), 4)
        item["matched_terms"] = matched
        candidates[item["clause_id"]] = item

    results = sorted(
        candidates.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:k]


def get_clause(clause_id: str) -> dict[str, Any] | None:
    """Retrieve a clause by exact ID or an unambiguous heading/number."""
    chunks = load_chunks()
    query = clause_id.strip().casefold()

    # Exact clause-ID lookup. A clause split into parts is returned whole (all parts joined), so following a
    # reference always yields the complete clause.
    for item in chunks:
        if item.get("clause_id", "").casefold() == query:
            parent = item.get("parent_id", item["clause_id"])
            parts = [c for c in chunks if c.get("parent_id", c["clause_id"]) == parent]
            if len(parts) > 1:
                item = dict(parts[0], clause_id=parent,
                            text=" ".join(c["text"] for c in sorted(parts, key=lambda c: c.get("part", 1))))
            item["score"] = 1.0
            return item

    # Fall back to matching the visible heading, e.g. "Condition 3".
    import re
    match = re.fullmatch(
        r"(section|condition|clause)\s+(?:no\.?\s*)?([ivxlcdm]+|\d+(?:\.\d+)*)",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None

    kind, number = match.groups()
    number = number.upper() if number.isalpha() else number

    candidates = []
    for item in chunks:
        heading = str(item.get("section", "")).strip()
        heading_match = re.match(
            r"^(section|condition|clause)\s+(?:no\.?\s*)?([ivxlcdm]+|\d+(?:\.\d+)*)\b",
            heading,
            re.IGNORECASE,
        )
        if heading_match:
            heading_kind, heading_number = heading_match.groups()
            if (
                heading_kind.casefold() == kind.casefold()
                and heading_number.casefold() == number.casefold()
            ):
                candidates.append(item)

    # Never silently choose between multiple clauses with the same number.
    if len(candidates) != 1:
        return None

    candidates[0]["score"] = 1.0
    return candidates[0]


# ---- L1: the same retrieval exposed as a LangChain retriever ----
# The agents call retrieval through this class, so the RAG layer runs on a framework abstraction
# (langchain-core's BaseRetriever) while ranking stays in search_corpus above.
try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class PolicyClauseRetriever(BaseRetriever):
        """LangChain retriever over the FAISS policy index. Each Document is one clause chunk;
        its metadata carries clause_id, section, section_type, source_doc, page, refs and score."""

        k: int = 5
        section_type: str | None = None

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            hits = search_corpus(query, k=self.k, section_type=self.section_type)
            return [
                Document(page_content=h["text"], metadata={key: v for key, v in h.items() if key != "text"})
                for h in hits
            ]
except ImportError:  # corpus.py still works standalone without langchain-core
    PolicyClauseRetriever = None


def retrieve(query: str, k: int = 5, section_type: str | None = None) -> list[dict[str, Any]]:
    """Run a query through the LangChain retriever and return plain dicts (same shape as search_corpus)."""
    if PolicyClauseRetriever is None:
        return search_corpus(query, k=k, section_type=section_type)
    docs = PolicyClauseRetriever(k=k, section_type=section_type).invoke(query)
    return [{**d.metadata, "text": d.page_content} for d in docs]


def get_fnol_field(field_name: str) -> dict[str, str] | None:
    """Return the description of a supported FNOL field."""
    if field_name not in FNOL_FIELDS:
        return None

    return {
        "field": field_name,
        "description": FNOL_FIELDS[field_name],
    }



def resolve_references() -> list[dict[str, Any]]:
    """Report explicit cross-references without guessing ambiguous targets."""
    chunks = load_chunks()
    report: list[dict[str, Any]] = []

    explicit_ref = re.compile(
        r"^\s*(section|condition|clause)\s+"
        r"(?:no\.?\s*)?([ivxlcdm]+|\d+(?:\.\d+)*)\s*$",
        re.I,
    )

    for source in chunks:
        for raw_ref in source.get("refs", []):
            match = explicit_ref.fullmatch(raw_ref)

            entry = {
                "reference": raw_ref,
                "source_clause_id": source.get("clause_id"),
                "source_page": source.get("page"),
                "status": "TEXTUAL_ONLY",
                "candidates": [],
            }

            if not match:
                report.append(entry)
                continue

            kind, number = match.groups()
            kind = kind.casefold()
            number = number.casefold()

            candidates = []

            for target in chunks:
                heading = str(target.get("section", "")).strip()

                if kind == "section":
                    # Match explicit section headings only.
                    heading_match = re.match(
                        r"^section\s+([ivxlcdm]+|\d+(?:\.\d+)*)\b",
                        heading,
                        re.I,
                    )
                    if heading_match and (
                        heading_match.group(1).casefold() == number
                    ):
                        candidates.append(target)

                elif kind == "condition":
                    # Numbered passages are candidates, not proof of identity.
                    heading_match = re.match(
                        r"^(\d+(?:\.\d+)*|[ivxlcdm]+)[.)]?\s+",
                        heading,
                        re.I,
                    )
                    if heading_match and (
                        heading_match.group(1).casefold() == number
                    ):
                        candidates.append(target)

                elif kind == "clause":
                    # Only accept a heading that explicitly starts with
                    # "Clause <number>"; do not guess subclause locations.
                    heading_match = re.match(
                        r"^clause\s+([ivxlcdm]+|\d+(?:\.\d+)*)\b",
                        heading,
                        re.I,
                    )
                    if heading_match and (
                        heading_match.group(1).casefold() == number
                    ):
                        candidates.append(target)

            entry["candidates"] = [
                {
                    "clause_id": target.get("clause_id"),
                    "page": target.get("page"),
                    "heading": target.get("section", ""),
                }
                for target in candidates
            ]

            if len(candidates) == 1:
                entry["status"] = "CANDIDATE"
            elif len(candidates) > 1:
                entry["status"] = "AMBIGUOUS"
            else:
                entry["status"] = "UNRESOLVED"

            report.append(entry)

    return report


def cli() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Parse PDFs and build/update the FAISS index",
    )

    index_parser.add_argument(
        "--document",
        type=Path,
        help="Only re-index this PDF",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search policy passages",
    )

    search_parser.add_argument("query")
    search_parser.add_argument("-k", type=int, default=5)

    search_parser.add_argument(
        "--section-type",
        choices=[
            "cover",
            "exclusion",
            "limitation",
            "condition",
            "definition",
            "procedure",
            "general",
        ],
    )

    clause_parser = subparsers.add_parser(
        "clause",
        help="Fetch an exact clause by ID",
    )

    clause_parser.add_argument("clause_id")

    find_clause_parser = subparsers.add_parser(
        "find-clause",
        help="Find a numbered clause and report all matching candidates",
    )
    find_clause_parser.add_argument("reference")

    field_parser = subparsers.add_parser(
        "field",
        help="Describe an FNOL field",
    )

    field_parser.add_argument("field_name")

    references_parser = subparsers.add_parser(
        "references",
        help="Report explicit policy cross-references and candidate targets",
    )

    args = parser.parse_args()

    if args.command == "references":
        print(
            json.dumps(
                resolve_references(),
                indent=2,
                ensure_ascii=False,
            )
        )

    elif args.command == "index":
        build_index(args.document)

    elif args.command == "search":
        print(
            json.dumps(
                search_corpus(
                    args.query,
                    args.k,
                    args.section_type,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )

    elif args.command == "clause":
        print(
            json.dumps(
                get_clause(args.clause_id),
                indent=2,
                ensure_ascii=False,
            )
        )

    elif args.command == "find-clause":
        import re

        match = re.fullmatch(
            r"\s*(section|condition|clause)\s+(?:no\.?\s*)?([ivxlcdm]+|\d+(?:\.\d+)*)\s*",
            args.reference,
            re.IGNORECASE,
        )
        candidates = []

        if match:
            kind, number = match.groups()
            for item in load_chunks():
                heading = str(item.get("section", "")).strip()
                hm = re.match(
                    r"^(?:(section|condition|clause)\s+(?:no\.?\s*)?)?([ivxlcdm]+|\d+(?:\.\d+)*)[.)]?\s+",
                    heading,
                    re.IGNORECASE,
                )
                if hm:
                    heading_kind, heading_number = hm.groups()
                    if (
                        heading_number.casefold() == number.casefold()
                        and (
                            heading_kind is None
                            or heading_kind.casefold() == kind.casefold()
                        )
                    ):
                        candidates.append(item)

        result = {
            "reference": args.reference,
            "status": "NOT_FOUND" if not candidates else "RESOLVED" if len(candidates) == 1 else "AMBIGUOUS",
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "field":
        print(
            json.dumps(
                get_fnol_field(args.field_name),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    cli()
