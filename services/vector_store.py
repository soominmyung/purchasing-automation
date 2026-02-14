"""
벡터스토어: supplier_history, item_history, analysis_examples, request_examples, email_examples.
n8n의 Vector Store In-Memory + Embeddings OpenAI에 대응.
"""
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# Chroma 저장 경로 (로컬 디렉터리)
_CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_stores: dict[str, Chroma] = {}
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            # 설정되지 않은 경우 오류를 내지 않고 None 반환 (나중에 호출 시 에러 발생)
            return None
        _embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    return _embeddings


def _get_or_create_store(collection_name: str) -> Chroma:
    if collection_name not in _stores:
        emb = _get_embeddings()
        if emb is None:
            # 키가 없으면 스토어 생성을 건너뜀 (나중에 다시 시도 가능)
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
    """Supplier history PDF 텍스트 + metadata supplier_name."""
    _add_docs(
        "supplier_history",
        [Document(page_content=text, metadata={"supplier_name": supplier_name, "doc_type": "supplier_history"})],
    )


def ingest_item_history(text: str, item_code: str | None) -> None:
    """Item history PDF 텍스트 + metadata item_code."""
    _add_docs(
        "item_history",
        [Document(page_content=text, metadata={"item_code": item_code or "", "doc_type": "item_history"})],
    )


def ingest_analysis_examples(text: str) -> None:
    """Purchasing analysis examples (n8n Analysis-examples 폴더)."""
    _add_docs(
        "analysis_examples",
        [Document(page_content=text, metadata={"doc_type": "analysis_examples"})],
    )


def ingest_request_examples(text: str) -> None:
    """PR (purchase request) examples (n8n Request-examples 폴더)."""
    _add_docs(
        "request_examples",
        [Document(page_content=text, metadata={"doc_type": "request_examples"})],
    )


def ingest_email_examples(text: str) -> None:
    """Email draft examples (n8n Email-examples 폴더)."""
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
