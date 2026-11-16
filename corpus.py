"""Person B corpus pipeline: PDF -> clause chunks -> MiniLM -> FAISS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

NOISE_LINE_RE = re.compile(
    r"^\s*(?:-\d+-|UIN:\s*IRDAN\S+(?:\s+-\d+-\s+CIN:\s*\S+|\s+CIN:\s*\S+)?)\s*$"
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
            "not liable",
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


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    """Extract PDF text into section-aware chunks."""
    reader = PdfReader(str(path))

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
        raw = page.extract_text() or ""

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

    # Avoid duplicate IDs caused by repeated headings.
    seen: dict[str, int] = {}

    for chunk in chunks:
        base = chunk["clause_id"]
        seen[base] = seen.get(base, 0) + 1

        if seen[base] > 1:
            chunk["clause_id"] = f"{base}-{seen[base]}"

    return chunks


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

        if previous.get("sha256") == digest:
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

    print("Generating embeddings...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        f"{chunk['section']}\n{chunk['text']}"
        for chunk in existing
    ]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

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

    return index, chunks, model


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

    index, chunks, model = _load_search_assets()

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

        item = dict(chunks[int(chunk_id)])

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

    # Preserve exact clause-ID lookup.
    for item in chunks:
        if item.get("clause_id", "").casefold() == query:
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
