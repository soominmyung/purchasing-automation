"""
Keeping the history store in step with a shared folder of history documents.

The folder is the company's existing one (organised by year, by supplier, or however the
team keeps it). Its layout is not used to decide what a document is about; the document's
header does that. The path is used to identify the document, to show where the original
is, and to warn when the header and the folder disagree (a supplier's folder holding a
document whose header names another supplier), which usually means a misfiled document
or a typo in the header.
"""
import hashlib
from pathlib import PurePosixPath
from typing import Any

from services.vector_store import (
    delete_history_source, history_key, history_sources, ingest_history, known_suppliers,
)

ORIGIN = "folder"


def fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def folder_warning(path: str, supplier: str, suppliers_on_file: set[str]) -> str | None:
    """A warning when a folder on the path is named after a different supplier than the header."""
    header_key = history_key(supplier)
    for folder in PurePosixPath(path).parts[:-1]:
        key = history_key(folder)
        if key in suppliers_on_file and key != header_key:
            named = f"names supplier '{supplier}'" if supplier else "names no supplier"
            return f"Filed under folder '{folder}', but the header {named}."
    return None


def store_folder_document(text: str, path: str, file_fingerprint: str) -> dict[str, Any]:
    """Add or replace one document from the shared folder."""
    header = ingest_history(text, source=path, origin=ORIGIN, fingerprint=file_fingerprint)
    return {
        "path": path, "ok": True, "supplier": header["supplier"] or None, "items": header["items"],
        "item_codes": header["item_codes"], "date": header["date"] or None,
        "warning": folder_warning(path, header["supplier"], known_suppliers()),
    }


def remove_folder_document(path: str) -> int:
    return delete_history_source(ORIGIN, path)


def folder_documents() -> dict[str, dict[str, str]]:
    """Documents currently held from the shared folder, by path, with their fingerprints."""
    return history_sources(ORIGIN)
