"""
Vector stores: supplier_history, item_history, analysis_examples, request_examples, email_examples.
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
    """supplier_history, item_history, analysis_examples, request_examples, email_examples."""
    for name in ("supplier_history", "item_history", "analysis_examples", "request_examples", "email_examples"):
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


def ingest_supplier_history(text: str, supplier_name: str) -> None:
    """Ingest supplier history text with supplier_name metadata."""
    _add_docs(
        "supplier_history",
        [Document(page_content=text, metadata={
            "supplier_name": supplier_name,
            "doc_type": "supplier_history",
            "ingested_at": _now_iso(),
            "event_date": _extract_event_date(text),
        })],
    )


def ingest_item_history(text: str, item_code: str | None) -> None:
    """Ingest item history text with item_code metadata."""
    _add_docs(
        "item_history",
        [Document(page_content=text, metadata={
            "item_code": item_code or "",
            "doc_type": "item_history",
            "ingested_at": _now_iso(),
            "event_date": _extract_event_date(text),
        })],
    )


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


def search_supplier_history(k: int = 5, filter: dict[str, Any] | None = None) -> list[Document]:
    """Most recent supplier-history chunks for the filtered supplier, ranked by the document's own event date (fact retrieval, not similarity)."""
    return _most_recent("supplier_history", k, filter=filter)


def search_item_history(k: int = 5, filter: dict[str, Any] | None = None) -> list[Document]:
    """Most recent item-history chunks for the filtered item(s), ranked by the document's own event date (fact retrieval, not similarity)."""
    return _most_recent("item_history", k, filter=filter)


def search_analysis_examples(k: int = 3) -> list[Document]:
    """Most recent analysis-report examples, ranked by the document's own date (style/tone reference, not fact retrieval)."""
    return _most_recent("analysis_examples", k)


def search_request_examples(k: int = 3) -> list[Document]:
    """Most recent purchase-request examples, ranked by the document's own date (style/tone reference, not fact retrieval)."""
    return _most_recent("request_examples", k)


def search_email_examples(k: int = 3) -> list[Document]:
    """Most recent supplier-email examples, ranked by the document's own date (style/tone reference, not fact retrieval)."""
    return _most_recent("email_examples", k)
