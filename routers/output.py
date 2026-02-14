"""Serve generated .docx files for download (Framer / HF Space). Temp files deleted after 5 min when use_temp_for_output."""
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from services.security import verify_api_access
from fastapi.responses import FileResponse

from config import settings
from utils.docx_utils import ANALYSIS_DIR, PR_DIR, EMAIL_DRAFT_DIR, TEMP_DIR, cleanup_temp_output

router = APIRouter()

_SAFE_FILENAME = re.compile(r"^(analysis_|pr_|email_draft_)[a-zA-Z0-9_\-\.]+\.docx$")
_USE_TEMP = getattr(settings, "use_temp_for_output", False)


def _resolve_path(filename: str) -> Path | None:
    if not _SAFE_FILENAME.match(filename):
        return None
    if _USE_TEMP:
        cleanup_temp_output()
        path = TEMP_DIR / filename
        if path.is_file():
            return path
        return None
    for directory in (ANALYSIS_DIR, PR_DIR, EMAIL_DRAFT_DIR):
        path = directory / filename
        if path.is_file():
            return path
    return None


@router.get("/output/list", dependencies=[Depends(verify_api_access)])
def list_output():
    """
    List generated .docx in output/temp (when use_temp_for_output). For Framer download list.
    Files older than temp_output_max_age_minutes are removed before listing.
    """
    if not _USE_TEMP:
        return {"files": [], "temp_expiry_minutes": 0}
    cleanup_temp_output()
    files = []
    if TEMP_DIR.is_dir():
        for f in sorted(TEMP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.is_file() and f.suffix.lower() == ".docx":
                files.append({"filename": f.name, "created_at": f.stat().st_mtime})
    return {"files": files, "temp_expiry_minutes": getattr(settings, "temp_output_max_age_minutes", 5)}


@router.get("/output/download", dependencies=[Depends(verify_api_access)])
def download_output(filename: str):
    """
    Download a generated .docx from output/temp (when use_temp) or output/analysis|pr|email_draft.
    """
    path = _resolve_path(filename)
    if not path:
        raise HTTPException(status_code=404, detail="File not found or invalid filename.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
