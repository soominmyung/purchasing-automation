"""
Vector stores: history, analysis_examples, request_examples, email_examples.
Corresponds to n8n Vector Store In-Memory + OpenAI Embeddings.
"""
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# Chroma storage path (local directory)
_CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_stores: dict[str, Chroma] = {}
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            # Return None gracefully if API key is not set (will error on actual use)
            return None
        _embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    return _embeddings


def _get_or_create_store(collection_name: str) -> Chroma:
    if collection_name not in _stores:
        emb = _get_embeddings()
        if emb is None:
            # Skip store creation if API key is missing (can retry later)
            return None
        _stores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=emb,
            persist_directory=str(_CHROMA_DIR),
        )
    return _stores.get(collection_name)


def get_vector_stores() -> dict[str, Chroma]:
    """history, analysis_examples, request_examples, email_examples."""
    for name in (HISTORY_COLLECTION, "analysis_examples", "request_examples", "email_examples"):
        _get_or_create_store(name)
    return _stores


def _add_docs(collection_name: str, documents: list[Document]) -> None:
    store = _get_or_create_store(collection_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    if splits:
        store.add_documents(splits)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_DATE_LINE_RE = re.compile(r"Date:\s*([^\n\r]+)")
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y")


def _extract_event_date(text: str) -> str:
    """Parse the document's own 'Date: ...' line (existing report-writing convention) into an ISO date.

    Returns "" if no such line is found or it doesn't match a known format; callers fall back
    to ingested_at in that case (see _most_recent) rather than mistaking an undated document for old.
    """
    match = _DATE_LINE_RE.search(text)
    if not match:
        return ""
    raw = match.group(1).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _most_recent(collection_name: str, k: int, filter: dict[str, Any] | None = None, sort_key: str = "event_date") -> list[Document]:
    """Return the k most recent chunks (optionally metadata-filtered), ranked by sort_key (not similarity).

    Falls back to ingested_at when sort_key is missing/empty (e.g. event_date couldn't be
    parsed from the document), so undated documents still rank by upload recency instead of
    being buried behind every dated document.
    """
    store = _get_or_create_store(collection_name)
    if store is None:
        return []
    raw = store._collection.get(where=filter, include=["metadatas", "documents"])
    pairs = list(zip(raw.get("documents") or [], raw.get("metadatas") or []))
    pairs.sort(key=lambda dm: dm[1].get(sort_key) or dm[1].get("ingested_at", ""), reverse=True)
    return [Document(page_content=text, metadata=meta) for text, meta in pairs[:k]]


# ----------------------------------------------------------------------------- history
# One collection for all purchasing history. Whether a document is about a supplier, an
# item, or an item from a supplier is read from its header, not from where it was uploaded:
#   Supplier: <name>          (absent for an item-only document, e.g. a demand spike)
#   Item: <name>[, <name>]    (absent for a supplier-wide document, e.g. a strike)
#   ItemCode: <code>[, ...]   (optional)
#   Date: <date>
HISTORY_COLLECTION = "history"
HEADER_LINES = 15
_HEADER_RE = {
    "supplier": re.compile(r"^\s*Supplier\s*:\s*(.+?)\s*$", re.IGNORECASE),
    "items": re.compile(r"^\s*Items?\s*:\s*(.+?)\s*$", re.IGNORECASE),
    "item_codes": re.compile(r"^\s*Item\s*Codes?\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE),
}


def history_key(value: str) -> str:
    """Case, spacing and punctuation do not matter when matching names and codes."""
    return re.sub(r"[\W_]+", "", str(value or "").lower())


def _split_list(value: str) -> list[str]:
    return [v.strip() for v in re.split(r"[,;/]", value) if v.strip()]


def parse_history_header(text: str) -> dict[str, Any]:
    """Supplier, items, item codes and date from the first lines of a history document."""
    header = {"supplier": "", "items": [], "item_codes": []}
    for line in (text or "").splitlines()[:HEADER_LINES]:
        for field, rx in _HEADER_RE.items():
            m = rx.match(line)
            if not m:
                continue
            if field == "supplier":
                header["supplier"] = header["supplier"] or m.group(1)
            else:
                header[field] += _split_list(m.group(1))
            break
    header["date"] = _extract_event_date(text)
    return header


def history_metadata(header: dict[str, Any], source: str = "", origin: str = "upload",
                     fingerprint: str = "") -> dict[str, Any]:
    """Chroma metadata for one history document. Lists cannot be filtered in Chroma, so each
    supplier, item name and item code also gets its own key for the retrieval filter.
    origin + source identify the document: "folder" with its path in the shared folder, or
    "upload" with its file name."""
    meta: dict[str, Any] = {
        "doc_type": "history",
        "supplier_name": header.get("supplier") or "",
        "items": ", ".join(header.get("items") or []),
        "item_codes": ", ".join(header.get("item_codes") or []),
        "event_date": header.get("date") or "",
        "ingested_at": _now_iso(),
        "source": source or "",
        "origin": origin,
        "source_id": f"{origin}:{source}",
        "fingerprint": fingerprint or "",
    }
    if meta["supplier_name"]:
        meta[f"s:{history_key(meta['supplier_name'])}"] = True
    for value in (header.get("items") or []) + (header.get("item_codes") or []):
        meta[f"i:{history_key(value)}"] = True
    return meta


def ingest_history(text: str, source: str = "", origin: str = "upload", fingerprint: str = "") -> dict[str, Any]:
    """Store one history document, replacing any earlier version with the same origin and
    source, so a file sent twice is held once. Returns what its header was read as, so the
    uploader can check it. A document that names neither a supplier nor an item is refused."""
    header = parse_history_header(text)
    if not header["supplier"] and not header["items"] and not header["item_codes"]:
        raise ValueError("The header names no supplier and no item (expected 'Supplier:' and/or 'Item:' lines).")
    if source:
        delete_history_source(origin, source)
    _add_docs(HISTORY_COLLECTION, [Document(page_content=text,
                                            metadata=history_metadata(header, source, origin, fingerprint))])
    return header


def delete_history_source(origin: str, source: str) -> int:
    """Remove every chunk of one document. Returns how many chunks were removed."""
    store = _get_or_create_store(HISTORY_COLLECTION)
    ids = store._collection.get(where={"source_id": f"{origin}:{source}"})["ids"]
    if ids:
        store._collection.delete(ids=ids)
    return len(ids)


def history_sources(origin: str) -> dict[str, dict[str, str]]:
    """Documents held from one origin: {source: {fingerprint, supplier, items}}."""
    store = _get_or_create_store(HISTORY_COLLECTION)
    raw = store._collection.get(where={"origin": origin}, include=["metadatas"])
    out: dict[str, dict[str, str]] = {}
    for m in raw.get("metadatas") or []:
        out[m.get("source") or ""] = {"fingerprint": m.get("fingerprint") or "",
                                      "supplier": m.get("supplier_name") or "", "items": m.get("items") or ""}
    return out


def known_suppliers() -> set[str]:
    """Normalised names of every supplier named in a history header."""
    raw = _get_or_create_store(HISTORY_COLLECTION)._collection.get(include=["metadatas"])
    return {history_key(m.get("supplier_name")) for m in raw.get("metadatas") or [] if m.get("supplier_name")}


def search_history(supplier: str, items: list[tuple[str, str]], k: int = 10) -> list[dict[str, Any]]:
    """
    The k most recent documents about this supplier or about any of these items, by the
    document's own date. items: (item_code, item_name) pairs; a document matches an item by
    either. Chunks of one document are joined back together.
    """
    keys = [f"s:{history_key(supplier)}"] if history_key(supplier) else []
    for code, name in items:
        keys += [f"i:{history_key(v)}" for v in (code, name) if history_key(v)]
    keys = list(dict.fromkeys(keys))
    if not keys:
        return []
    where = {keys[0]: True} if len(keys) == 1 else {"$or": [{k_: True} for k_ in keys]}
    chunks = _most_recent(HISTORY_COLLECTION, k=10_000, filter=where)
    docs: dict[tuple, dict[str, Any]] = {}
    for c in chunks:
        m = c.metadata or {}
        doc_key = (m.get("source"), m.get("supplier_name"), m.get("items"), m.get("item_codes"),
                   m.get("event_date"), m.get("ingested_at"))
        if doc_key not in docs:
            if len(docs) == k:
                continue
            docs[doc_key] = {
                "supplier": m.get("supplier_name") or "",
                "items": _split_list(m.get("items") or ""),
                "item_codes": _split_list(m.get("item_codes") or ""),
                "date": m.get("event_date") or "",  # undated: shown as such, still ranked by upload time
                "source": m.get("source") or "",
                "text": c.page_content,
            }
        else:
            docs[doc_key]["text"] += "\n" + c.page_content
    return list(docs.values())


def ingest_analysis_examples(text: str) -> None:
    """Ingest purchasing analysis reference examples."""
    _add_docs(
        "analysis_examples",
        [Document(page_content=text, metadata={
            "doc_type": "analysis_examples",
            "ingested_at": _now_iso(),
            "event_date": _extract_event_date(text),
        })],
    )


def ingest_request_examples(text: str) -> None:
    """Ingest purchase request reference examples."""
    _add_docs(
        "request_examples",
        [Document(page_content=text, metadata={
            "doc_type": "request_examples",
            "ingested_at": _now_iso(),
            "event_date": _extract_event_date(text),
        })],
    )


def ingest_email_examples(text: str) -> None:
    """Ingest email draft reference examples."""
    _add_docs(
        "email_examples",
        [Document(page_content=text, metadata={
            "doc_type": "email_examples",
            "ingested_at": _now_iso(),
            "event_date": _extract_event_date(text),
        })],
    )


def search_analysis_examples(k: int = 3) -> list[Document]:
    """Most recent analysis-report examples, ranked by the document's own date (style/tone reference, not fact retrieval)."""
    return _most_recent("analysis_examples", k)


def search_request_examples(k: int = 3) -> list[Document]:
    """Most recent purchase-request examples, ranked by the document's own date (style/tone reference, not fact retrieval)."""
    return _most_recent("request_examples", k)


def search_email_examples(k: int = 3) -> list[Document]:
    """Most recent supplier-email examples, ranked by the document's own date (style/tone reference, not fact retrieval)."""
    return _most_recent("email_examples", k)
