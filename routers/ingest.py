"""
문서 수집 API: Supplier/Item History PDF, Analysis/Request/Email 예시 PDF 업로드 → 벡터스토어 저장.
- 여러 파일(files) 또는 ZIP 하나(zip_file)로 대량 업로드 가능. (ZIP = 폴더 압축 후 한 번에 전송)
"""
import io
import re
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from services.security import verify_api_key

from utils.pdf_utils import extract_text_from_pdf
from services.vector_store import (
    ingest_supplier_history,
    ingest_item_history,
    ingest_analysis_examples,
    ingest_request_examples,
    ingest_email_examples,
)

router = APIRouter()


def _extract_supplier_name(text: str) -> str:
    """Supplier: ... 헤더에서 이름 추출."""
    m = re.search(r"Supplier\s*:\s*(.+?)(?:\r?\n|$)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("Missing 'Supplier: ...' header in supplier history document.")
    return m.group(1).split("\n")[0].strip()


def _extract_item_code(text: str) -> str | None:
    """ItemCode: 100004 패턴 추출."""
    m = re.search(r"ItemCode\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    return m.group(1) if m else None


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


@router.post("/supplier-history", dependencies=[Depends(verify_api_key)])
async def ingest_supplier_history_pdf(
    files: list[UploadFile] = File(..., description="PDF 파일 여러 개 (폴더에서 다 선택 후 업로드 가능)"),
):
    """Supplier history PDF 업로드. 여러 파일 한 번에 가능. 각 파일 내 'Supplier: <Name>' 한 줄 필수."""
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
            supplier_name = _extract_supplier_name(text)
            ingest_supplier_history(text, supplier_name)
            results.append({"filename": file.filename, "ok": True, "supplier_name": supplier_name})
        except Exception as e:
            results.append({"filename": file.filename, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "processed": len(results), "results": results}


@router.post("/supplier-history/zip", dependencies=[Depends(verify_api_key)])
async def ingest_supplier_history_zip(
    file: UploadFile = File(..., description="ZIP 파일 (폴더 압축, 내부 PDF 전부 처리). 100개면 1번 업로드."),
):
    """Supplier history PDF를 ZIP 하나로 업로드. ZIP 안의 모든 .pdf를 읽어서 벡터스토어에 넣음. (스케일링용)"""
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
            supplier_name = _extract_supplier_name(text)
            ingest_supplier_history(text, supplier_name)
            results.append({"filename": name, "ok": True, "supplier_name": supplier_name})
        except Exception as e:
            results.append({"filename": name, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "processed": len(results), "results": results}


@router.post("/item-history", dependencies=[Depends(verify_api_key)])
async def ingest_item_history_pdf(
    files: list[UploadFile] = File(..., description="PDF 파일 여러 개 (폴더에서 다 선택 후 업로드 가능)"),
):
    """Item history PDF 업로드. 여러 파일 한 번에 가능. 각 파일 내 'ItemCode: XXXXX' 있으면 메타데이터로 사용."""
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
            item_code = _extract_item_code(text)
            ingest_item_history(text, item_code)
            results.append({"filename": file.filename, "ok": True, "item_code": item_code})
        except Exception as e:
            results.append({"filename": file.filename, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "processed": len(results), "results": results}


@router.post("/item-history/zip", dependencies=[Depends(verify_api_key)])
async def ingest_item_history_zip(
    file: UploadFile = File(..., description="ZIP 파일 (폴더 압축, 내부 PDF 전부 처리). 100개면 1번 업로드."),
):
    """Item history PDF를 ZIP 하나로 업로드. ZIP 안의 모든 .pdf 처리. (스케일링용)"""
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
            item_code = _extract_item_code(text)
            ingest_item_history(text, item_code)
            results.append({"filename": name, "ok": True, "item_code": item_code})
        except Exception as e:
            results.append({"filename": name, "ok": False, "error": str(e)})
        finally:
            path.unlink(missing_ok=True)
    return {"ok": True, "processed": len(results), "results": results}


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
