"""
파이프라인 API: CSV 업로드 → Item Grouping → Analysis Agent → Report / PR / Email 생성.
스트리밍 시 complete 이벤트에 생성 파일 base64 포함 (HF Space 등 비영속 환경에서 다운로드 보장).
"""
import asyncio
import base64
import json
import queue
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from fastapi.responses import StreamingResponse
from langsmith import traceable

from services.security import verify_api_access
from config import settings
from schemas import RunPipelineRequest, RunPipelineResponse
from utils.csv_utils import parse_csv_rows
from utils.docx_utils import (
    save_analysis_docx,
    save_pr_docx,
    save_email_draft_docx,
    save_evaluation_docx,
    markdown_to_docx_bytes,
    _sanitize_filename,
)
from services.item_grouping import group_by_supplier_and_recommend
from services.agents import (
    run_purchasing_pipeline_graph,
    purchasing_graph,
)

router = APIRouter()


@traceable(run_type="chain", name="Purchasing_Pipeline")
def _run_pipeline(
    csv_content: str,
    csv_filename: str,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    embed_files: bool = False,
) -> RunPipelineResponse:
    """CSV 내용 + 파일명으로 파이프라인 실행 (공통 로직). progress_callback(step, detail) 호출.
    스트리밍 시(progress_callback 있음): 디스크 저장 없이 메모리에서 .docx → base64만 전달.
    embed_files=True: 디스크 저장 없이 메모리에서 .docx → base64만 JSON에 담아 반환 (HF Space 등)."""
    stream_mode = progress_callback is not None
    in_memory = stream_mode or embed_files

    def progress(step: str, detail: dict[str, Any] | None = None) -> None:
        if progress_callback:
            progress_callback(step, detail or {})

    progress("csv_parsing", {})
    rows = parse_csv_rows(csv_content, csv_filename)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No valid CSV rows. Ensure columns: SupplierName, ItemCode, ItemName, CurrentStock, WksToOOS, RiskLevel (or similar).",
        )
    progress("item_grouping", {})
    groups = group_by_supplier_and_recommend(rows)
    if not groups:
        raise HTTPException(
            status_code=400,
            detail="No groups by supplier. Check SupplierName/Supplier column.",
        )
    progress("item_grouping_done", {"count": len(groups)})

    reports = []
    requests_list = []
    emails = []
    evaluations = []

    for group in groups:
        snapshot_date = group["snapshot_date"]
        supplier = group["supplier"]
        items = group["items"]
        risk_level = items[0].get("risk_level", "N/A") if items else "N/A"
        safe_supplier = _sanitize_filename(supplier)

        # --- LangGraph Execution with Progress Streaming ---
        input_data = {
            "snapshot_date": snapshot_date,
            "supplier": supplier,
            "items": items,
        }
        
        final_state = {}
        # 그래프 스트리밍을 통해 각 노드 완료 시 진행 상황 알림
        for event in purchasing_graph.stream(
            {
                "snapshot_date": snapshot_date,
                "supplier": supplier,
                "items": items,
                "risk_level": risk_level,
                "iteration_count": 0,
                "is_valid_email": False
            },
            config={
                "run_name": f"Purchasing-Workflow-{supplier}",
                "metadata": {"supplier": supplier, "env": "production"}
            }
        ):
            # event는 { "node_name": { "state_key": value } } 형태
            node_name = list(event.keys())[0]
            progress(node_name, {"supplier": supplier})
            
            # 마지막 상태를 계속 업데이트
            for k, v in event[node_name].items():
                final_state[k] = v

        # --- 결과 추출 및 파일 생성 (기존 로직 유지) ---
        analysis_output = final_state.get("analysis_output", {})
        report_md = final_state.get("report_md", "")
        pr_md = final_state.get("pr_md", "")
        email_text = final_state.get("email_text", "")
        evaluation_md = final_state.get("evaluation_md", "")

        # 0. Evaluation Report 처리
        evaluation_filename = f"evaluation_{snapshot_date}_{safe_supplier}.docx"
        if in_memory:
            docx_bytes = markdown_to_docx_bytes(evaluation_md)
            b64 = base64.b64encode(docx_bytes).decode("ascii")
            evaluations.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "markdown": evaluation_md,
                "filename": evaluation_filename, "saved_path": "", "content_base64": b64,
            })
            if stream_mode:
                progress("file_ready", {"filename": evaluation_filename, "content_base64": b64})
        else:
            evaluation_saved_path = save_evaluation_docx(snapshot_date, supplier, evaluation_md)
            evaluations.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "markdown": evaluation_md,
                "filename": evaluation_filename, "saved_path": evaluation_saved_path,
                "content_base64": _read_file_base64(evaluation_saved_path),
            })

        # 1. Report 처리
        analysis_filename = f"analysis_{snapshot_date}_{safe_supplier}.docx"
        if in_memory:
            docx_bytes = markdown_to_docx_bytes(report_md)
            b64 = base64.b64encode(docx_bytes).decode("ascii")
            reports.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "markdown": report_md,
                "filename": analysis_filename, "saved_path": "", "content_base64": b64,
            })
            if stream_mode:
                progress("file_ready", {"filename": analysis_filename, "content_base64": b64})
        else:
            analysis_saved_path = save_analysis_docx(snapshot_date, supplier, report_md)
            reports.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "markdown": report_md,
                "filename": analysis_filename, "saved_path": analysis_saved_path,
                "content_base64": _read_file_base64(analysis_saved_path),
            })

        # 2. PR 처리
        pr_filename = f"pr_{snapshot_date}_{safe_supplier}.docx"
        if in_memory:
            docx_bytes = markdown_to_docx_bytes(pr_md)
            b64 = base64.b64encode(docx_bytes).decode("ascii")
            requests_list.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "markdown": pr_md,
                "filename": pr_filename, "saved_path": "", "content_base64": b64,
            })
            if stream_mode:
                progress("file_ready", {"filename": pr_filename, "content_base64": b64})
        else:
            pr_saved_path = save_pr_docx(snapshot_date, supplier, pr_md)
            requests_list.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "markdown": pr_md,
                "filename": pr_filename, "saved_path": pr_saved_path,
                "content_base64": _read_file_base64(pr_saved_path),
            })

        # 3. Email 처리
        email_filename = f"email_draft_{snapshot_date}_{safe_supplier}.docx"
        if in_memory:
            docx_bytes = markdown_to_docx_bytes(email_text)
            b64 = base64.b64encode(docx_bytes).decode("ascii")
            emails.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "text": email_text,
                "filename": email_filename, "saved_path": "", "content_base64": b64,
            })
            if stream_mode:
                progress("file_ready", {"filename": email_filename, "content_base64": b64})
        else:
            email_saved_path = save_email_draft_docx(snapshot_date, supplier, email_text)
            emails.append({
                "snapshot_date": snapshot_date, "supplier": supplier, "text": email_text,
                "filename": email_filename, "saved_path": email_saved_path,
                "content_base64": _read_file_base64(email_saved_path),
            })

    response = RunPipelineResponse(
        groups=groups,
        reports=reports,
        requests=requests_list,
        emails=emails,
        evaluations=evaluations,
    )
    result_dump = response.model_dump()
    if stream_mode:
        progress("complete", {"result": result_dump})
    return response


def _read_file_base64(path_str: str) -> str | None:
    """Read file and return base64 string so Framer can download without /api/output/download (HF Space)."""
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


@router.post("/run", response_model=RunPipelineResponse, dependencies=[Depends(verify_api_access)])
async def run_pipeline_upload(file: UploadFile = File(..., description="CSV 파일 (내용은 서버에서 읽음)")):
    """
    CSV **파일**만 업로드하면 서버가 내용을 읽어 파이프라인 실행.
    파일명에 DDMMYY(예: 050425)가 있으면 스냅샷 날짜(2025-04-25)로 사용.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required. Upload a .csv file.")

    content = await file.read()
    try:
        csv_content = content.decode("utf-8")
    except UnicodeDecodeError:
        csv_content = content.decode("cp1252", errors="replace")  # fallback

    csv_filename = file.filename or "stock.csv"
    return _run_pipeline(csv_content, csv_filename, progress_callback=None, embed_files=True)


@router.post("/run/embed", response_model=RunPipelineResponse, dependencies=[Depends(verify_api_access)])
async def run_pipeline_embed(file: UploadFile = File(..., description="CSV 파일 (디스크 저장 없이 JSON에 base64 포함)")):
    """
    CSV 업로드 후 파이프라인 실행. **디스크에 저장하지 않고** 생성된 .docx를 base64로 JSON에 담아 반환.
    HF Space 등에서 브라우저가 응답 수신 후 바로 다운로드할 수 있도록 함 (스트리밍/임시 저장 불필요).
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required.")
    content = await file.read()
    try:
        csv_content = content.decode("utf-8")
    except UnicodeDecodeError:
        csv_content = content.decode("cp1252", errors="replace")
    csv_filename = file.filename or "stock.csv"
    return _run_pipeline(csv_content, csv_filename, progress_callback=None, embed_files=True)


async def _stream_pipeline_events(csv_content: str, csv_filename: str):
    """파이프라인 진행 이벤트를 SSE로 yield."""
    q: queue.Queue = queue.Queue()

    def progress(step: str, detail: dict[str, Any]) -> None:
        q.put({"step": step, **detail})

    def run_sync() -> None:
        try:
            _run_pipeline(csv_content, csv_filename, progress_callback=progress)
        except Exception as e:
            q.put({"step": "error", "error": str(e)})

    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, run_sync)

    while True:
        try:
            msg = await loop.run_in_executor(None, lambda: q.get(block=True, timeout=0.3))
        except queue.Empty:
            if task.done():
                break
            await asyncio.sleep(0.05)
            continue
        if msg.get("step") == "complete":
            yield f"data: {json.dumps(msg)}\n\n"
            break
        if msg.get("step") == "error":
            yield f"data: {json.dumps(msg)}\n\n"
            break
        yield f"data: {json.dumps(msg)}\n\n"

    await task


@router.post("/run/stream", dependencies=[Depends(verify_api_access)])
async def run_pipeline_stream(file: UploadFile = File(..., description="CSV 파일 (진행 상황 스트리밍)")):
    """
    CSV 업로드 후 파이프라인 실행. 진행 단계를 Server-Sent Events로 스트리밍.
    step: csv_parsing, item_grouping, item_grouping_done, analysis, report, pr, email, complete
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required.")
    content = await file.read()
    try:
        csv_content = content.decode("utf-8")
    except UnicodeDecodeError:
        csv_content = content.decode("cp1252", errors="replace")
    csv_filename = file.filename or "stock.csv"
    return StreamingResponse(
        _stream_pipeline_events(csv_content, csv_filename),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/run/json", response_model=RunPipelineResponse, dependencies=[Depends(verify_api_access)])
def run_pipeline_json(req: RunPipelineRequest):
    """
    JSON으로 CSV 내용 + 파일명 전달 (API/스크립트용).
    브라우저에서 파일만 올리고 싶으면 POST /api/run (파일 업로드) 사용.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    return _run_pipeline(req.csv_content, req.csv_filename or "")


@router.post("/group-only")
def run_group_only(req: RunPipelineRequest):
    """
    CSV만 파싱하여 Item Grouping만 수행 (LLM 호출 없음).
    권장 PO일/납기일/수량 계산 결과만 반환.
    """
    rows = parse_csv_rows(req.csv_content, req.csv_filename or "")
    if not rows:
        raise HTTPException(status_code=400, detail="No valid CSV rows.")
    groups = group_by_supplier_and_recommend(rows)
    return {"groups": groups}


