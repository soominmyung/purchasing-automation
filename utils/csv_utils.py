"""CSV 파싱 및 파일명에서 스냅샷 날짜 추출 (n8n Code 노드 로직)."""
import csv
import io
import re
from typing import Any


def snapshot_date_from_filename(file_name: str) -> str | None:
    """
    파일명에서 DDMMYY 패턴 추출 → YYYY-MM-DD.
    e.g. Urgent_Stock_050425.csv → 2025-04-25
    """
    if not file_name or not isinstance(file_name, str):
        return None
    m = re.search(r"(\d{2})(\d{2})(\d{2})", file_name)
    if not m:
        return None
    dd, mm, yy = m.group(1), m.group(2), m.group(3)
    year = 2000 + int(yy)
    month = str(int(mm)).zfill(2)
    day = str(int(dd)).zfill(2)
    return f"{year}-{month}-{day}"


def _normalize_key(key: str) -> str:
    """BOM, 공백, 언더스코어 제거 후 소문자."""
    return key.replace("\ufeff", "").replace(" ", "").replace("_", "").lower()


def find_field(row: dict[str, Any], target_name: str) -> Any:
    """유연 필드 매칭 (n8n findField)."""
    norm_target = _normalize_key(target_name)
    for k, v in row.items():
        norm_k = _normalize_key(k)
        if norm_k == norm_target:
            return v
    return None


def parse_csv_rows(
    csv_content: str,
    csv_filename: str | None = None,
) -> list[dict[str, Any]]:
    """
    CSV 텍스트를 파싱하고, 파일명에서 스냅샷 날짜를 붙여 각 행 반환.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(reader)
    snapshot_date = snapshot_date_from_filename(csv_filename or "") if csv_filename else None

    out = []
    for row in rows:
        # 스냅샷 날짜가 없으면 첫 행에서 snapshot_date 컬럼 찾기
        snap = snapshot_date or find_field(row, "snapshotdate") or find_field(row, "snapshot_date")
        if snap:
            snapshot_date = str(snap).strip() if snapshot_date is None else snapshot_date
        r = dict(row)
        if snapshot_date:
            r["snapshot_date"] = snapshot_date
        out.append(r)
    return out
