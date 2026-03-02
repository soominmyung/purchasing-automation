"""
Vector stores: supplier_history, item_history, analysis_examples, request_examples, email_examples.
Corresponds to n8n Vector Store In-Memory + OpenAI Embeddings.
"""
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
        [Document(page_content=text, metadata={"doc_type": "analysis_examples"})],
    )


def ingest_request_examples(text: str) -> None:
    """Ingest purchase request reference examples."""
    _add_docs(
        "request_examples",
        [Document(page_content=text, metadata={"doc_type": "request_examples"})],
    )


def ingest_email_examples(text: str) -> None:
    """Ingest email draft reference examples."""
    _add_docs(
        "email_examples",
        [Document(page_content=text, metadata={"doc_type": "email_examples"})],
    )


def search_supplier_history(query: str, k: int = 5) -> list[Document]:
    store = _get_or_create_store("supplier_history")
    return store.similarity_search(query, k=k)


def search_item_history(query: str, k: int = 5) -> list[Document]:
    store = _get_or_create_store("item_history")
    return store.similarity_search(query, k=k)


def search_analysis_examples(query: str, k: int = 3) -> list[Document]:
    store = _get_or_create_store("analysis_examples")
    return store.similarity_search(query, k=k)


def search_request_examples(query: str, k: int = 3) -> list[Document]:
    store = _get_or_create_store("request_examples")
    return store.similarity_search(query, k=k)


def search_email_examples(query: str, k: int = 3) -> list[Document]:
    store = _get_or_create_store("email_examples")
    return store.similarity_search(query, k=k)
