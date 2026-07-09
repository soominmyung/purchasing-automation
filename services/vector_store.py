"""
Vector stores: supplier_history, item_history, analysis_examples, request_examples, email_examples.
Corresponds to n8n Vector Store In-Memory + OpenAI Embeddings.
"""
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


def _most_recent(collection_name: str, k: int) -> list[Document]:
    """Return the k most recently ingested chunks, ranked by ingested_at (not similarity)."""
    store = _get_or_create_store(collection_name)
    if store is None:
        return []
    raw = store._collection.get(include=["metadatas", "documents"])
    pairs = list(zip(raw.get("documents") or [], raw.get("metadatas") or []))
    pairs.sort(key=lambda dm: dm[1].get("ingested_at", ""), reverse=True)
    return [Document(page_content=text, metadata=meta) for text, meta in pairs[:k]]


def ingest_supplier_history(text: str, supplier_name: str) -> None:
    """Ingest supplier history text with supplier_name metadata."""
    _add_docs(
        "supplier_history",
        [Document(page_content=text, metadata={"supplier_name": supplier_name, "doc_type": "supplier_history"})],
    )


def ingest_item_history(text: str, item_code: str | None) -> None:
    """Ingest item history text with item_code metadata."""
    _add_docs(
        "item_history",
        [Document(page_content=text, metadata={"item_code": item_code or "", "doc_type": "item_history"})],
    )


def ingest_analysis_examples(text: str) -> None:
    """Ingest purchasing analysis reference examples."""
    _add_docs(
        "analysis_examples",
        [Document(page_content=text, metadata={"doc_type": "analysis_examples", "ingested_at": _now_iso()})],
    )


def ingest_request_examples(text: str) -> None:
    """Ingest purchase request reference examples."""
    _add_docs(
        "request_examples",
        [Document(page_content=text, metadata={"doc_type": "request_examples", "ingested_at": _now_iso()})],
    )


def ingest_email_examples(text: str) -> None:
    """Ingest email draft reference examples."""
    _add_docs(
        "email_examples",
        [Document(page_content=text, metadata={"doc_type": "email_examples", "ingested_at": _now_iso()})],
    )


def search_supplier_history(query: str, k: int = 5, filter: dict[str, Any] | None = None) -> list[Document]:
    store = _get_or_create_store("supplier_history")
    return store.similarity_search(query, k=k, filter=filter)


def search_item_history(query: str, k: int = 5, filter: dict[str, Any] | None = None) -> list[Document]:
    store = _get_or_create_store("item_history")
    return store.similarity_search(query, k=k, filter=filter)


def search_analysis_examples(k: int = 3) -> list[Document]:
    """Most recently ingested analysis-report examples (style/tone reference, not fact retrieval)."""
    return _most_recent("analysis_examples", k)


def search_request_examples(k: int = 3) -> list[Document]:
    """Most recently ingested purchase-request examples (style/tone reference, not fact retrieval)."""
    return _most_recent("request_examples", k)


def search_email_examples(k: int = 3) -> list[Document]:
    """Most recently ingested supplier-email examples (style/tone reference, not fact retrieval)."""
    return _most_recent("email_examples", k)
