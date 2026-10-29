"""
The history store: one Chroma collection of purchasing history documents, each stored whole
and retrieved by its header metadata.
"""
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import settings

# Chroma storage path (local directory)
_CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_stores: dict[str, Chroma] = {}
_embeddings = None
# Opening the Chroma client from several threads at once fails ("Could not connect to tenant
# default_tenant", seen 2026-10-24 with four approval calls in parallel): one opener at a time.
_open_lock = threading.Lock()


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            # Return None gracefully if API key is not set (will error on actual use)
            return None
        _embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    return _embeddings


def _get_or_create_store(collection_name: str) -> Chroma:
    with _open_lock:
        return _open_store(collection_name)


def _open_store(collection_name: str) -> Chroma:
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
    """Open the history store at start-up."""
    _get_or_create_store(HISTORY_COLLECTION)
    return _stores


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_DATE_LINE_RE = re.compile(r"Date:\s*([^\n\r]+)")
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y")


def _extract_event_date(text: str) -> str:
    """Parse the document's own 'Date: ...' line (existing report-writing convention) into an ISO date.

    Returns "" if no such line is found or it doesn't match a known format; callers fall back
    to ingested_at in that case (see _doc_date) rather than mistaking an undated document for old.
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
    # stored whole: a history document is at most two pages and is retrieved by its metadata
    _get_or_create_store(HISTORY_COLLECTION).add_documents(
        [Document(page_content=text, metadata=history_metadata(header, source, origin, fingerprint))])
    return header


def delete_history_source(origin: str, source: str) -> int:
    """Remove one document. Returns how many entries were removed (0 if it was not held)."""
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


# Retrieval rule agreed with the purchasing team: documents from the 12 months before the
# snapshot date, and within that the 5 most recent about the supplier as a whole and the 5 most
# recent about each item. Set in config.py (history_window_months, history_docs_per_scope).
def _doc_date(meta: dict[str, Any]) -> str:
    """The document's own date, or its upload date when the document is undated."""
    return meta.get("event_date") or (meta.get("ingested_at") or "")[:10]


def _months_before(day: str, months: int) -> str:
    y, m, d = (int(x) for x in day[:10].split("-"))
    m -= months
    while m <= 0:
        m += 12
        y -= 1
    return f"{y:04d}-{m:02d}-{min(d, 28):02d}"


def search_history(supplier: str, items: list[tuple[str, str]], as_of: str,
                   months: int, per_scope: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    History documents for one analysis. A document qualifies when it names this supplier or any
    of these items (by code or name) and is dated within `months` before `as_of`. Within that,
    the `per_scope` most recent documents are kept for the supplier (supplier-wide documents and
    documents about its other items) and for each item; a document in several scopes is kept
    once. Returns (documents, counts) where counts says how many qualified and how many were
    left out, for the report.
    """
    keys = [f"s:{history_key(supplier)}"] if history_key(supplier) else []
    for code, name in items:
        keys += [f"i:{history_key(v)}" for v in (code, name) if history_key(v)]
    keys = list(dict.fromkeys(keys))
    counts = {"window_months": months, "from": _months_before(as_of, months), "to": as_of[:10],
              "per_scope": per_scope, "matched": 0, "in_window": 0, "included": 0, "left_out": 0}
    if not keys:
        return [], counts
    where = {keys[0]: True} if len(keys) == 1 else {"$or": [{k_: True} for k_ in keys]}
    store = _get_or_create_store(HISTORY_COLLECTION)
    raw = store._collection.get(where=where, include=["metadatas", "documents"])

    # one entry per document
    docs: dict[tuple, dict[str, Any]] = {}
    for text, m in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        key = (m.get("source_id") or m.get("source"), m.get("supplier_name"), m.get("items"),
               m.get("item_codes"), m.get("event_date"), m.get("ingested_at"))
        if key in docs:
            docs[key]["text"] += "\n" + text
            continue
        docs[key] = {"supplier": m.get("supplier_name") or "",
                     "items": _split_list(m.get("items") or ""),
                     "item_codes": _split_list(m.get("item_codes") or ""),
                     "date": m.get("event_date") or "", "source": m.get("source") or "",
                     "_sort": _doc_date(m), "_keys": {k_ for k_ in m if k_.startswith("i:")},
                     "text": text}
    counts["matched"] = len(docs)
    start, end = counts["from"], counts["to"]
    pool = sorted((d for d in docs.values() if start < d["_sort"] <= end), key=lambda d: d["_sort"], reverse=True)
    counts["in_window"] = len(pool)

    item_keys = [{f"i:{history_key(v)}" for v in (code, name) if history_key(v)} for code, name in items]
    scopes = [[d for d in pool if not (d["_keys"] & set().union(*item_keys)) and history_key(d["supplier"]) == history_key(supplier)]]
    scopes += [[d for d in pool if d["_keys"] & ik] for ik in item_keys]
    chosen: list[dict[str, Any]] = []
    for scope in scopes:
        for d in scope[:per_scope]:
            if not any(d is c for c in chosen):
                chosen.append(d)
    chosen.sort(key=lambda d: d["_sort"], reverse=True)
    counts["included"] = len(chosen)
    counts["left_out"] = counts["in_window"] - len(chosen)
    for d in chosen:
        d.pop("_sort"), d.pop("_keys")
    return chosen, counts

