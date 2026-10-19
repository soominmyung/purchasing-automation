"""
문서 수집 API: Purchasing History PDF, Analysis/Request/Email 예시 PDF 업로드 → 벡터스토어 저장.
- 여러 파일(files) 또는 ZIP 하나(zip_file)로 대량 업로드 가능. (ZIP = 폴더 압축 후 한 번에 전송)
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
from services.vector_store import (
    ingest_history,
    ingest_analysis_examples,
    ingest_request_examples,
    ingest_email_examples,
)

router = APIRouter()


def _save_upload_to_temp(content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        return Path(tmp.name)


def _extract_pdfs_from_zip(zip_content: bytes) -> list[tuple[bytes, str]]:
    """ZIP 바이트에서 모든 PDF의 (내용 바이트, 원본 파일명) 목록 반환."""
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


async def _history_files(files: list[UploadFile]) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file required.")
    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename or "(unknown)", "ok": False, "error": "PDF file required."})
            continue
        results.append(_ingest_history_pdf(await file.read(), file.filename))
    return {"ok": True, "processed": len(results), "results": results}


async def _history_zip(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="ZIP file required.")
    pdfs = _extract_pdfs_from_zip(await file.read())
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files found inside the ZIP.")
    results = [_ingest_history_pdf(content, name) for content, name in pdfs]
    return {"ok": True, "processed": len(results), "results": results}


@router.post("/history", dependencies=[Depends(verify_api_key)])
async def ingest_history_pdf(
    files: list[UploadFile] = File(..., description="PDF 파일 여러 개"),
):
    """
    Purchasing history PDF 업로드. 공급사 이력과 품목 이력을 나누지 않는다.
    문서 첫머리의 'Supplier:', 'Item:'(여러 개는 쉼표), 'ItemCode:', 'Date:' 줄로 무엇에 관한 문서인지 정한다.
    공급사만 있으면 공급사 전체, 둘 다 있으면 그 공급사의 그 품목, 품목만 있으면 품목 자체에 관한 문서.
    결과에 문서별로 읽힌 헤더를 돌려주므로 올린 사람이 확인할 수 있다.
    """
    return await _history_files(files)


@router.post("/history/zip", dependencies=[Depends(verify_api_key)])
async def ingest_history_zip(
    file: UploadFile = File(..., description="ZIP 파일 (내부 PDF 전부 처리)"),
):
    """History PDF를 ZIP 하나로 업로드. 처리 방식은 /history와 같다."""
    return await _history_zip(file)


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
    return {"ok": True, "path": path, "chunks_removed": remove_folder_document(path)}


# Earlier routes, kept so existing clients keep working. The upload path no longer decides
# what a document is about; the header does, exactly as for /history.
@router.post("/supplier-history", dependencies=[Depends(verify_api_key)], deprecated=True)
async def ingest_supplier_history_pdf(files: list[UploadFile] = File(...)):
    return await _history_files(files)


@router.post("/supplier-history/zip", dependencies=[Depends(verify_api_key)], deprecated=True)
async def ingest_supplier_history_zip(file: UploadFile = File(...)):
    return await _history_zip(file)


@router.post("/item-history", dependencies=[Depends(verify_api_key)], deprecated=True)
async def ingest_item_history_pdf(files: list[UploadFile] = File(...)):
    return await _history_files(files)


@router.post("/item-history/zip", dependencies=[Depends(verify_api_key)], deprecated=True)
async def ingest_item_history_zip(file: UploadFile = File(...)):
    return await _history_zip(file)


@router.post("/analysis-examples", dependencies=[Depends(verify_api_key)])
async def ingest_analysis_examples_pdf(
    files: list[UploadFile] = File(..., description="PDF 파일 여러 개 (폴더에서 다 선택 후 업로드 가능)"),
):
    """Purchasing analysis examples PDF 업로드. 여러 파일 한 번에 가능."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file required.")
    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename or "(unknown)", "ok": False, "error": "PDF file required."})
            continue
        content = await file.read()
        path = _save_upload_to_temp(content)
        try:
            text = extract_text_from_pdf(path)
            ingest_analysis_examples(text)
            results.append({"filename": file.filename, "ok": True})
        except Exception as e:
            results.append({"filename": file.filename, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "store": "analysis_examples", "processed": len(results), "results": results}


@router.post("/analysis-examples/zip", dependencies=[Depends(verify_api_key)])
async def ingest_analysis_examples_zip(
    file: UploadFile = File(..., description="ZIP 파일 (폴더 압축, 내부 PDF 전부 처리). 100개면 1번 업로드."),
):
    """Analysis examples PDF를 ZIP 하나로 업로드. (스케일링용)"""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="ZIP file required.")
    content = await file.read()
    pdfs = _extract_pdfs_from_zip(content)
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files found inside the ZIP.")
    results = []
    for pdf_bytes, name in pdfs:
        path = _save_upload_to_temp(pdf_bytes)
        try:
            text = extract_text_from_pdf(path)
            ingest_analysis_examples(text)
            results.append({"filename": name, "ok": True})
        except Exception as e:
            results.append({"filename": name, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "store": "analysis_examples", "processed": len(results), "results": results}


@router.post("/request-examples", dependencies=[Depends(verify_api_key)])
async def ingest_request_examples_pdf(
    files: list[UploadFile] = File(..., description="PDF 파일 여러 개 (폴더에서 다 선택 후 업로드 가능)"),
):
    """PR (purchase request) examples PDF 업로드. 여러 파일 한 번에 가능."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file required.")
    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename or "(unknown)", "ok": False, "error": "PDF file required."})
            continue
        content = await file.read()
        path = _save_upload_to_temp(content)
        try:
            text = extract_text_from_pdf(path)
            ingest_request_examples(text)
            results.append({"filename": file.filename, "ok": True})
        except Exception as e:
            results.append({"filename": file.filename, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "store": "request_examples", "processed": len(results), "results": results}


@router.post("/request-examples/zip", dependencies=[Depends(verify_api_key)])
async def ingest_request_examples_zip(
    file: UploadFile = File(..., description="ZIP 파일 (폴더 압축, 내부 PDF 전부 처리). 100개면 1번 업로드."),
):
    """PR request examples PDF를 ZIP 하나로 업로드. (스케일링용)"""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="ZIP file required.")
    content = await file.read()
    pdfs = _extract_pdfs_from_zip(content)
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files found inside the ZIP.")
    results = []
    for pdf_bytes, name in pdfs:
        path = _save_upload_to_temp(pdf_bytes)
        try:
            text = extract_text_from_pdf(path)
            ingest_request_examples(text)
            results.append({"filename": name, "ok": True})
        except Exception as e:
            results.append({"filename": name, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "store": "request_examples", "processed": len(results), "results": results}


@router.post("/email-examples", dependencies=[Depends(verify_api_key)])
async def ingest_email_examples_pdf(
    files: list[UploadFile] = File(..., description="PDF 파일 여러 개 (폴더에서 다 선택 후 업로드 가능)"),
):
    """Email draft examples PDF 업로드. 여러 파일 한 번에 가능."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file required.")
    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename or "(unknown)", "ok": False, "error": "PDF file required."})
            continue
        content = await file.read()
        path = _save_upload_to_temp(content)
        try:
            text = extract_text_from_pdf(path)
            ingest_email_examples(text)
            results.append({"filename": file.filename, "ok": True})
        except Exception as e:
            results.append({"filename": file.filename, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "store": "email_examples", "processed": len(results), "results": results}


@router.post("/email-examples/zip", dependencies=[Depends(verify_api_key)])
async def ingest_email_examples_zip(
    file: UploadFile = File(..., description="ZIP 파일 (폴더 압축, 내부 PDF 전부 처리). 100개면 1번 업로드."),
):
    """Email draft examples PDF를 ZIP 하나로 업로드. (스케일링용)"""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="ZIP file required.")
    content = await file.read()
    pdfs = _extract_pdfs_from_zip(content)
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files found inside the ZIP.")
    results = []
    for pdf_bytes, name in pdfs:
        path = _save_upload_to_temp(pdf_bytes)
        try:
            text = extract_text_from_pdf(path)
            ingest_email_examples(text)
            results.append({"filename": name, "ok": True})
        except Exception as e:
            results.append({"filename": name, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "store": "email_examples", "processed": len(results), "results": results}
