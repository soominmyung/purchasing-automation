"""
History document intake: PDF uploads (several files, or one ZIP) and the shared-folder sync.
Each document's header decides what it is about; see services/vector_store.py.
"""
import io

import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from services.security import verify_api_key
from services.history_folder import (
    fingerprint, folder_documents, remove_folder_document, store_folder_document,
)

from utils.pdf_utils import extract_text_from_pdf
from services.vector_store import ingest_history

router = APIRouter()


def _save_upload_to_temp(content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        return Path(tmp.name)


def _extract_pdfs_from_zip(zip_content: bytes) -> list[tuple[bytes, str]]:
    """(content, file name) of every PDF inside a ZIP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zf:
            zf.extractall(root)
        pdfs = [(p.read_bytes(), p.name) for p in root.rglob("*.pdf") if p.is_file()]
    return pdfs


def _ingest_history_pdf(content: bytes, filename: str) -> dict:
    """One history PDF: what its header was read as, or why it was refused."""
    path = _save_upload_to_temp(content)
    try:
        header = ingest_history(extract_text_from_pdf(path), source=filename)
        return {"filename": filename, "ok": True, "supplier": header["supplier"] or None,
                "items": header["items"], "item_codes": header["item_codes"], "date": header["date"] or None}
    except Exception as e:
        return {"filename": filename, "ok": False, "error": str(e)}
    finally:
        path.unlink(missing_ok=True)


@router.post("/history", dependencies=[Depends(verify_api_key)])
async def ingest_history_pdf(
    files: list[UploadFile] = File(..., description="History PDFs"),
):
    """
    Upload history PDFs. What a document is about is read from its first lines: 'Supplier:',
    'Item:' (several separated by commas), 'ItemCode:', 'Date:'. A supplier alone means the
    supplier as a whole; a supplier and items, those items from that supplier; items alone, the
    items whoever supplies them. The response gives each document's header as it was read, so
    the uploader can check it.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file required.")
    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename or "(unknown)", "ok": False, "error": "PDF file required."})
            continue
        results.append(_ingest_history_pdf(await file.read(), file.filename))
    return {"ok": True, "processed": len(results), "results": results}


@router.post("/history/zip", dependencies=[Depends(verify_api_key)])
async def ingest_history_zip(
    file: UploadFile = File(..., description="ZIP of history PDFs (every PDF inside is read)"),
):
    """Upload history PDFs as one ZIP; each is handled as by /history."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="ZIP file required.")
    pdfs = _extract_pdfs_from_zip(await file.read())
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files found inside the ZIP.")
    results = [_ingest_history_pdf(content, name) for content, name in pdfs]
    return {"ok": True, "processed": len(results), "results": results}


# ----- Shared-folder sync (used by scripts/sync_history_folder.py) -----
# The sync job runs next to the company's network drive, compares the folder with what the
# store holds, and sends only what changed. Documents are identified by their path in the
# folder, so a changed file replaces its earlier version and a deleted file is removed.

@router.get("/history/folder", dependencies=[Depends(verify_api_key)])
async def list_folder_history():
    """Documents held from the shared folder: path -> fingerprint, supplier, items."""
    return {"ok": True, "documents": folder_documents()}


@router.put("/history/folder", dependencies=[Depends(verify_api_key)])
async def put_folder_history(
    file: UploadFile = File(..., description="History PDF"),
    path: str = Form(..., description="Path of the file inside the shared folder, with / separators"),
):
    """Add or replace one document from the shared folder."""
    if not path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file required.")
    content = await file.read()
    tmp = _save_upload_to_temp(content)
    try:
        return store_folder_document(extract_text_from_pdf(tmp), path, fingerprint(content))
    except ValueError as e:
        return {"path": path, "ok": False, "error": str(e)}
    finally:
        tmp.unlink(missing_ok=True)


@router.delete("/history/folder", dependencies=[Depends(verify_api_key)])
async def delete_folder_history(path: str):
    """Remove a document that was deleted from the shared folder."""
    return {"ok": True, "path": path, "removed": remove_folder_document(path)}
