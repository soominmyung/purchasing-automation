"""
Pipeline API: stock CSV -> item grouping -> analysis -> analysis report, purchase request and supplier
email. Generated files can be returned base64-encoded in the response or the stream, for clients
that do not fetch them from the server.
"""
import asyncio
from datetime import date
import base64
import json
import queue
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, Depends
from fastapi.responses import StreamingResponse
from langsmith import traceable

from services.security import verify_api_access
from config import run_date, settings
from schemas import RunPipelineRequest, RunPipelineResponse
from utils.csv_utils import (
    SNAPSHOT_DATE_KEY, SNAPSHOT_SOURCE_KEY, SNAPSHOT_WARNINGS_KEY, SnapshotDateError, parse_csv_rows,
)
from utils.docx_utils import (
    markdown_for_web,
    save_analysis_docx,
    save_pr_docx,
    save_email_draft_docx,
    save_evaluation_docx,
    markdown_to_docx_bytes,
    _sanitize_filename,
)
from services.item_grouping import group_by_supplier_and_recommend, group_risk_level, screen_rows
from services.agents import (
    AGENT_MODELS,
    run_purchasing_pipeline_graph,
    purchasing_graph,
)
from services.run_record import RunRecord, client_ip

router = APIRouter()


@traceable(run_type="chain", name="Purchasing_Pipeline")
def _run_pipeline(
    csv_content: str,
    csv_filename: str,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    embed_files: bool = False,
    ip: str = "unknown",
) -> RunPipelineResponse:
    """Run the pipeline and keep a run record (services/run_record.py), also when it fails."""
    record = RunRecord(csv_content, csv_filename or "", ip, AGENT_MODELS)
    try:
        response = _run_pipeline_inner(csv_content, csv_filename, progress_callback, embed_files, record)
    except Exception as e:
        record.finish(error=e)
        raise
    record.finish()
    return response


def _run_pipeline_inner(
    csv_content: str,
    csv_filename: str,
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
    embed_files: bool,
    record: RunRecord,
) -> RunPipelineResponse:
    """Run the pipeline on the content and file name of a CSV, calling progress_callback(step, detail).
    When streaming (progress_callback given) or with embed_files=True, the .docx files are built in
    memory and returned base64-encoded instead of being saved to disk."""
    stream_mode = progress_callback is not None
    in_memory = stream_mode or embed_files

    def progress(step: str, detail: dict[str, Any] | None = None) -> None:
        if progress_callback:
            progress_callback(step, detail or {})

    progress("csv_parsing", {})
    try:
        rows = parse_csv_rows(csv_content, csv_filename)
    except SnapshotDateError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No valid CSV rows. Ensure columns: SupplierName, ItemCode, ItemName, AvailableStock, WksToOOS (or similar).",
        )
    record.set_snapshot(rows, (SNAPSHOT_DATE_KEY, SNAPSHOT_SOURCE_KEY, SNAPSHOT_WARNINGS_KEY))
    _require_fresh_snapshot(rows[0][SNAPSHOT_DATE_KEY])
    progress("item_grouping", {})
    groups = group_by_supplier_and_recommend(rows)
    if not groups:
        _, excluded = screen_rows(rows)
        reasons = " ".join(e["line"] for e in excluded[:10])
        raise HTTPException(status_code=400, detail=f"No row could be analysed. {reasons}")
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
        entry = record.start_group(group)
        final_state: dict[str, Any] = {}
        files: dict[str, str] = {}
        try:
            snapshot_date = group["snapshot_date"]
            supplier = group["supplier"]
            items = group["items"]
            risk_level = group.get("risk_level") or group_risk_level(items)
            safe_supplier = _sanitize_filename(supplier)

            # --- LangGraph Execution with Progress Streaming ---
            input_data = {
                "snapshot_date": snapshot_date,
                "supplier": supplier,
                "items": items,
            }
        
            # report progress as each node of the graph finishes
            for event in purchasing_graph.stream(
                {
                    "snapshot_date": snapshot_date,
                    "supplier": supplier,
                    "items": items,
                    "risk_level": risk_level,
                    "data_issue_lines": group.get("data_issue_lines", []),
                    "run_id": record.run_id,
                    "iteration_count": 0,
                    "is_valid_email": False
                },
                config=(
                    {"run_name": "Purchasing-Workflow", "metadata": {"env": "production", "run_id": record.run_id}}
                    if settings.langsmith_hide_content else
                    {"run_name": f"Purchasing-Workflow-{supplier}",
                     "metadata": {"supplier": supplier, "env": "production", "run_id": record.run_id}}
                )
            ):
                # event: {node_name: {state_key: value}}
                node_name = list(event.keys())[0]
                progress(node_name, {"supplier": supplier})
            
                # keep the latest state
                for k, v in event[node_name].items():
                    if k in ("validation_log", "document_issues", "analysis_attempts"):  # accumulated
                        final_state.setdefault(k, []).extend(v)
                    else:
                        final_state[k] = v

            # --- outputs and files ---
            analysis_output = final_state.get("analysis_output", {})
            report_md = final_state.get("report_md", "")
            pr_md = final_state.get("pr_md", "")
            email_text = final_state.get("email_md", "")
            evaluation_md = final_state.get("evaluation_md", "")

            # 0. Evaluation Report: the operator's (2026-10-28). Always saved on the server
            # beside the run record; sent with the drafts only when deliver_evaluation_report is on.
            evaluation_filename = f"evaluation_{snapshot_date}_{safe_supplier}_{record.run_id}.docx"
            files["evaluation"] = evaluation_filename
            evaluation_saved_path = save_evaluation_docx(snapshot_date, supplier, evaluation_md, run_id=record.run_id)
            if settings.deliver_evaluation_report:
                b64 = _read_file_base64(evaluation_saved_path)
                evaluations.append({
                    "snapshot_date": snapshot_date, "supplier": supplier, "markdown": evaluation_md,
                    "checks": final_state.get("evaluation_result"),
                    "filename": evaluation_filename, "saved_path": "" if in_memory else evaluation_saved_path,
                    "content_base64": b64,
                })
                if stream_mode:
                    progress("file_ready", {"filename": evaluation_filename, "content_base64": b64})

            # 1. analysis report
            analysis_filename = f"analysis_{snapshot_date}_{safe_supplier}_{record.run_id}.docx"
            files["analysis"] = analysis_filename
            if in_memory:
                docx_bytes = markdown_to_docx_bytes(report_md)
                b64 = base64.b64encode(docx_bytes).decode("ascii")
                reports.append({
                    "snapshot_date": snapshot_date, "supplier": supplier, "markdown": markdown_for_web(report_md),
                    "filename": analysis_filename, "saved_path": "", "content_base64": b64,
                })
                if stream_mode:
                    progress("file_ready", {"filename": analysis_filename, "content_base64": b64})
            else:
                analysis_saved_path = save_analysis_docx(snapshot_date, supplier, report_md, run_id=record.run_id)
                reports.append({
                    "snapshot_date": snapshot_date, "supplier": supplier, "markdown": markdown_for_web(report_md),
                    "filename": analysis_filename, "saved_path": analysis_saved_path,
                    "content_base64": _read_file_base64(analysis_saved_path),
                })

            # Every item still Low after the analysis: the run ended with the analysis report, whose
            # Summary says so; no purchase request or email file
            if not pr_md and not email_text:
                progress("nothing_to_request", {"supplier": supplier})
                record.finish_group(entry, final_state, files)
                continue

            # 2. purchase request
            pr_filename = f"pr_{snapshot_date}_{safe_supplier}_{record.run_id}.docx"
            files["pr"] = pr_filename
            if in_memory:
                docx_bytes = markdown_to_docx_bytes(pr_md)
                b64 = base64.b64encode(docx_bytes).decode("ascii")
                requests_list.append({
                    "snapshot_date": snapshot_date, "supplier": supplier, "markdown": markdown_for_web(pr_md),
                    "filename": pr_filename, "saved_path": "", "content_base64": b64,
                })
                if stream_mode:
                    progress("file_ready", {"filename": pr_filename, "content_base64": b64})
            else:
                pr_saved_path = save_pr_docx(snapshot_date, supplier, pr_md, run_id=record.run_id)
                requests_list.append({
                    "snapshot_date": snapshot_date, "supplier": supplier, "markdown": markdown_for_web(pr_md),
                    "filename": pr_filename, "saved_path": pr_saved_path,
                    "content_base64": _read_file_base64(pr_saved_path),
                })

            # 3. supplier email
            email_filename = f"email_draft_{snapshot_date}_{safe_supplier}_{record.run_id}.docx"
            files["email_draft"] = email_filename
            if in_memory:
                docx_bytes = markdown_to_docx_bytes(email_text)
                b64 = base64.b64encode(docx_bytes).decode("ascii")
                emails.append({
                    "snapshot_date": snapshot_date, "supplier": supplier, "text": markdown_for_web(email_text),
                    "filename": email_filename, "saved_path": "", "content_base64": b64,
                })
                if stream_mode:
                    progress("file_ready", {"filename": email_filename, "content_base64": b64})
            else:
                email_saved_path = save_email_draft_docx(snapshot_date, supplier, email_text, run_id=record.run_id)
                emails.append({
                    "snapshot_date": snapshot_date, "supplier": supplier, "text": markdown_for_web(email_text),
                    "filename": email_filename, "saved_path": email_saved_path,
                    "content_base64": _read_file_base64(email_saved_path),
                })

        except Exception as e:
            record.finish_group(entry, final_state, files, error=e)
            raise
        record.finish_group(entry, final_state, files)

    response = RunPipelineResponse(
        groups=groups,
        reports=reports,
        requests=requests_list,
        emails=emails,
        evaluations=evaluations,
        run_id=record.run_id,
    )
    result_dump = response.model_dump()
    if stream_mode:
        progress("complete", {"result": result_dump})
    return response


def _require_fresh_snapshot(snapshot_date: str) -> None:
    """Refuse stock data older than max_snapshot_age_days: the analysis is as of the snapshot
    date, and an older snapshot no longer describes the stock."""
    if not settings.require_fresh_snapshot:
        return
    # run_date() is fixed for runs on past sample data; in production it is the actual date
    # of the run.
    age = (run_date() - date.fromisoformat(snapshot_date)).days
    if age > settings.max_snapshot_age_days:
        raise HTTPException(
            status_code=400,
            detail=(f"The stock data is from {snapshot_date}, {age} days ago; at most "
                    f"{settings.max_snapshot_age_days} days is accepted. Export a current CSV from SAP and run again."),
        )


def _read_file_base64(path_str: str) -> str | None:
    """A saved file as base64, for clients that do not fetch it from /api/output/download."""
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


@router.post("/run", response_model=RunPipelineResponse, dependencies=[Depends(verify_api_access)])
async def run_pipeline_upload(request: Request, file: UploadFile = File(..., description="Stock CSV")):
    """
    Run the pipeline on an uploaded stock CSV. The snapshot date comes from the SnapshotDate line,
    or else from a DDMMYY date in the file name (050425 -> 2025-04-05).
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
    return _run_pipeline(csv_content, csv_filename, progress_callback=None, embed_files=True,
                         ip=client_ip(request))


@router.post("/run/embed", response_model=RunPipelineResponse, dependencies=[Depends(verify_api_access)])
async def run_pipeline_embed(request: Request, file: UploadFile = File(..., description="Stock CSV")):
    """
    Run the pipeline and return the generated .docx files base64-encoded in the JSON response,
    without saving them on the server.
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
    return _run_pipeline(csv_content, csv_filename, progress_callback=None, embed_files=True,
                         ip=client_ip(request))


async def _stream_pipeline_events(csv_content: str, csv_filename: str, ip: str = "unknown"):
    """Yield the progress events of the pipeline as Server-Sent Events."""
    q: queue.Queue = queue.Queue()

    def progress(step: str, detail: dict[str, Any]) -> None:
        q.put({"step": step, **detail})

    def run_sync() -> None:
        try:
            _run_pipeline(csv_content, csv_filename, progress_callback=progress, ip=ip)
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
async def run_pipeline_stream(request: Request, file: UploadFile = File(..., description="Stock CSV")):
    """
    Run the pipeline and stream its progress as Server-Sent Events.
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
        _stream_pipeline_events(csv_content, csv_filename, client_ip(request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/run/json", response_model=RunPipelineResponse, dependencies=[Depends(verify_api_access)])
def run_pipeline_json(req: RunPipelineRequest, request: Request):
    """
    Run the pipeline on CSV content and a file name sent as JSON (for scripts). To upload a file,
    use POST /api/run.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    return _run_pipeline(req.csv_content, req.csv_filename or "", ip=client_ip(request))


@router.post("/group-only")
def run_group_only(req: RunPipelineRequest):
    """
    Parse the CSV and group the items only (no LLM call): the recommended order and delivery
    dates and quantities.
    """
    try:
        rows = parse_csv_rows(req.csv_content, req.csv_filename or "")
    except SnapshotDateError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not rows:
        raise HTTPException(status_code=400, detail="No valid CSV rows.")
    groups = group_by_supplier_and_recommend(rows)
    return {"groups": groups}


