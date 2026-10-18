"""
CSV parsing and snapshot-date resolution.

Input format: an optional reference line holding the snapshot date, a blank line, then
the inventory table.

    SnapshotDate,2025-04-05

    ItemCode,ItemName,SupplierName,CurrentStock,WksToOOS
    100000,ItemA,SupplierA,100,25

The snapshot date is taken from that reference cell first. If the cell is absent or not
a valid date, the DDMMYY date in the file name is used (Urgent_Stock_050425.csv →
2025-04-05). If neither gives a valid date the file is rejected: every order date is
calculated from it, so there is no safe default.
"""
import csv
import io
import re
from datetime import date, datetime
from typing import Any

# Keys the parser adds to each row. They are chosen so they cannot collide with a CSV
# column under find_field's loose matching (which ignores case, spaces and underscores).
SNAPSHOT_DATE_KEY = "pipeline_snapshot_date"
SNAPSHOT_SOURCE_KEY = "pipeline_snapshot_source"
SNAPSHOT_WARNINGS_KEY = "pipeline_snapshot_warnings"
ROW_NUMBER_KEY = "pipeline_row_number"   # line number in the original file, as a spreadsheet shows it

_REFERENCE_LABEL = "snapshotdate"
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DDMMYY = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")


class SnapshotDateError(ValueError):
    """Neither the reference cell nor the file name gives a usable snapshot date."""


def _check_real_past_date(d: date) -> str | None:
    """None when the date is usable, otherwise the reason it is not."""
    if d > date.today():
        return f"{d.isoformat()} is in the future"
    return None


def snapshot_date_from_filename(file_name: str) -> str | None:
    """
    DDMMYY in the file name → YYYY-MM-DD, e.g. Urgent_Stock_050425.csv → 2025-04-05.
    None when there is no six-digit group or it is not a real, non-future date.
    """
    return _date_from_filename(file_name)[0]


def _date_from_filename(file_name: str | None) -> tuple[str | None, str]:
    if not file_name or not isinstance(file_name, str):
        return None, "no file name"
    m = _DDMMYY.search(file_name)
    if not m:
        return None, "no DDMMYY date in the file name"
    dd, mm, yy = (int(g) for g in m.groups())
    try:
        d = date(2000 + yy, mm, dd)
    except ValueError:
        return None, f"'{m.group(0)}' in the file name is not a real date"
    reason = _check_real_past_date(d)
    return (None, reason) if reason else (d.isoformat(), "")


def _date_from_reference(value: str | None) -> tuple[str | None, str]:
    if value is None:
        return None, "no SnapshotDate reference line"
    text = value.strip()
    if not text:
        return None, "SnapshotDate reference cell is empty"
    if not _ISO_DATE.match(text):
        return None, f"SnapshotDate '{text}' is not in YYYY-MM-DD form"
    try:
        d = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None, f"SnapshotDate '{text}' is not a real date"
    reason = _check_real_past_date(d)
    return (None, reason) if reason else (d.isoformat(), "")


def _split_reference_line(csv_content: str) -> tuple[str | None, str, int]:
    """
    Separate an optional leading "SnapshotDate,<value>" line from the table.
    Returns (reference value or None if there is no such line, table text,
    number of file lines that precede the table).
    """
    text = csv_content.lstrip("﻿")
    lines = text.splitlines(keepends=True)
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None:
        return None, text, 0
    cells = next(csv.reader([lines[first]]))
    if not cells or _normalize_key(cells[0]) != _REFERENCE_LABEL:
        return None, "".join(lines[first:]), first
    value = cells[1] if len(cells) > 1 else ""
    start = first + 1
    while start < len(lines) and not lines[start].strip():  # blank separator line(s) before the header
        start += 1
    return value, "".join(lines[start:]), start


def resolve_snapshot_date(reference_value: str | None, file_name: str | None) -> tuple[str, str, list[str]]:
    """
    Returns (snapshot_date, source, warnings). The reference cell wins when it is valid;
    otherwise the file name. Raises SnapshotDateError when neither is usable.
    """
    ref_date, ref_reason = _date_from_reference(reference_value)
    if ref_date:
        return ref_date, "reference cell", []
    name_date, name_reason = _date_from_filename(file_name)
    if name_date:
        warnings = [] if reference_value is None else [
            f"the SnapshotDate cell was ignored ({ref_reason}); the date in the file name ({name_date}) was used."]
        return name_date, "file name", warnings
    raise SnapshotDateError(
        "Snapshot date could not be determined: "
        f"reference cell — {ref_reason}; file name — {name_reason}. "
        "Add a 'SnapshotDate,YYYY-MM-DD' line above the table or name the file with DDMMYY."
    )


def _normalize_key(key: str) -> str:
    """Strip BOM, spaces, and underscores then lowercase."""
    return key.replace("﻿", "").replace(" ", "").replace("_", "").lower()


def find_field(row: dict[str, Any], target_name: str) -> Any:
    """Flexible field name matching (n8n findField)."""
    norm_target = _normalize_key(target_name)
    for k, v in row.items():
        if k is None:
            continue
        norm_k = _normalize_key(k)
        if norm_k == norm_target:
            return v
    return None


def parse_csv_rows(
    csv_content: str,
    csv_filename: str | None = None,
) -> list[dict[str, Any]]:
    """
    Parse the inventory CSV and attach the resolved snapshot date to every row.
    Raises SnapshotDateError when no usable snapshot date can be found.
    """
    reference_value, table, lines_before_table = _split_reference_line(csv_content)
    snapshot_date, source, warnings = resolve_snapshot_date(reference_value, csv_filename)
    out = []
    reader = csv.DictReader(io.StringIO(table))
    for row in reader:
        r = dict(row)
        r[SNAPSHOT_DATE_KEY] = snapshot_date
        r[SNAPSHOT_SOURCE_KEY] = source
        r[SNAPSHOT_WARNINGS_KEY] = warnings
        r[ROW_NUMBER_KEY] = lines_before_table + reader.line_num
        out.append(r)
    return out
