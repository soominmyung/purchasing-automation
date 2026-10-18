"""
Item Grouping, Suggested dates & quantities logic.
- Groups CSV rows by supplier
- Validates CurrentStock / WksToOOS: a figure is valid, missing, or invalid, and an
  invalid one is reported as such rather than coerced to 0 or to a default
- Calculates per-item recommended_latest_po_date, recommended_latest_delivery_date, timing labels
- Derives risk_level from WksToOOS (any RiskLevel column in the CSV is ignored)
- suggested_quantity: order quantity to cover 26 weeks after latest delivery
- system_notes: fixed sentences the pipeline guarantees in the item's notes
"""
import math
import re
from datetime import datetime, timedelta
from typing import Any

from utils.csv_utils import (
    ROW_NUMBER_KEY,
    SNAPSHOT_DATE_KEY,
    SNAPSHOT_SOURCE_KEY,
    SNAPSHOT_WARNINGS_KEY,
    find_field,
)

# Constants matching n8n workflow
IMPORT_LEAD_WEEKS = 16
INTERNAL_LEAD_WEEKS = 2
TOTAL_LEAD_WEEKS = IMPORT_LEAD_WEEKS + INTERNAL_LEAD_WEEKS
COVERAGE_WEEKS_AFTER_DELIVERY = 26

# Risk level is derived from weeks-to-stockout rather than taken from the source data,
# so it can never disagree with WksToOOS. With a lead time of about four months:
#   Critical  under the lead time: a stockout is certain even if ordered today
#   High      under 6 months of cover (26 weeks)
#   Medium    6 to 8 months (8 months is ~34.7 weeks, taken as 35)
#   Low       8 months or more
RISK_HIGH_BELOW_WEEKS = 26
RISK_MEDIUM_BELOW_WEEKS = 35
RISK_ORDER = ["Critical", "High", "Medium", "Low"]

# Figures beyond these are treated as input errors, never clipped to a bound.
MAX_PLAUSIBLE_WKS_TO_OOS = 520        # ten years of cover
MAX_PLAUSIBLE_STOCK = 10_000_000

_MISSING_TOKENS = {"", "n/a", "na", "none", "null", "nan", "-"}
_PLAIN_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def parse_figure(raw: Any, field: str, max_value: float) -> tuple[float | None, dict | None]:
    """
    Read one numeric field. Returns (value, None) when valid, or (None, issue) where
    issue is {"field", "status": "missing" | "invalid", "raw", "reason"}.
    Zero and fractions are valid (WksToOOS is stock divided by weekly demand, so it is
    rarely whole); negatives, non-numbers and implausibly large figures are not.
    """
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in _MISSING_TOKENS):
        return None, {"field": field, "status": "missing", "raw": raw, "reason": "no value"}
    if isinstance(raw, bool):
        return None, {"field": field, "status": "invalid", "raw": raw, "reason": "not a number"}
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip()
        if _THOUSANDS.match(text):
            text = text.replace(",", "")
        if not _PLAIN_NUMBER.match(text):
            return None, {"field": field, "status": "invalid", "raw": raw, "reason": "not a number"}
        value = float(text)
    if not math.isfinite(value):
        return None, {"field": field, "status": "invalid", "raw": raw, "reason": "not a finite number"}
    if value < 0:
        return None, {"field": field, "status": "invalid", "raw": raw, "reason": "negative"}
    if value > max_value:
        return None, {"field": field, "status": "invalid", "raw": raw,
                      "reason": f"implausibly large, over {max_value:,.0f}"}
    # Whole numbers stay whole, so reports show "450 units" rather than "450.0".
    return (int(value) if value.is_integer() else value), None


def _parse_wks(raw: Any) -> tuple[float | None, dict | None]:
    return parse_figure(raw, "WksToOOS", MAX_PLAUSIBLE_WKS_TO_OOS)


def _wks(raw: Any) -> float | None:
    return _parse_wks(raw)[0]


def _stock(raw: Any) -> float | None:
    return parse_figure(raw, "CurrentStock", MAX_PLAUSIBLE_STOCK)[0]


def _parse_date(date_str: str) -> datetime:
    """YYYY-MM-DD → datetime (UTC midnight)."""
    if not date_str or not isinstance(date_str, str):
        raise ValueError(f"snapshot_date is missing or invalid: {date_str}")
    return datetime.strptime(date_str.strip() + " 00:00:00", "%Y-%m-%d %H:%M:%S")


def _format_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _add_days(d: datetime, days: int) -> datetime:
    return d + timedelta(days=days)


def _build_timing_label_from_weeks(weeks_diff: float) -> str:
    if weeks_diff <= 0.5:
        return "immediately"
    if weeks_diff <= 1:
        return "within 1 week"
    if weeks_diff <= 2:
        return "within 2 weeks"
    if weeks_diff <= 4:
        return "within 2–4 weeks"
    if weeks_diff <= 8:
        return "within 4–8 weeks"
    return f"within {round(weeks_diff)} weeks"


def risk_level_from_weeks(wks_to_oos_raw: Any) -> str:
    """Critical / High / Medium / Low from weeks-to-stockout; N/A when missing or invalid."""
    wks = _wks(wks_to_oos_raw)
    if wks is None:
        return "N/A"
    if wks < TOTAL_LEAD_WEEKS:
        return "Critical"
    if wks < RISK_HIGH_BELOW_WEEKS:
        return "High"
    if wks < RISK_MEDIUM_BELOW_WEEKS:
        return "Medium"
    return "Low"


def group_risk_level(items: list[dict[str, Any]]) -> str:
    """The most severe risk among a supplier's items; N/A only if none is known."""
    known = [it.get("risk_level") for it in items if it.get("risk_level") in RISK_ORDER]
    return min(known, key=RISK_ORDER.index) if known else "N/A"


def expected_stockout_weeks(wks_to_oos_raw: Any) -> float | None:
    """Weeks the item will be out of stock even if ordered today; 0 when the lead time fits."""
    wks = _wks(wks_to_oos_raw)
    if wks is None:
        return None
    return round(max(0.0, TOTAL_LEAD_WEEKS - wks), 1)


def build_recommendations_for_item(snapshot_date_str: str, wks_to_oos_raw: Any) -> dict[str, Any]:
    """
    Per-item recommended PO date, delivery date, and timing labels.
    Without a valid WksToOOS nothing is calculated: every field is "N/A" rather than a
    date built on an assumed figure.
    """
    wks = _wks(wks_to_oos_raw)
    if wks is None:
        return {
            "recommended_latest_po_date": "N/A",
            "recommended_latest_delivery_date": "N/A",
            "recommended_latest_po_timing": "N/A",
            "recommended_latest_delivery_timing": "N/A",
        }
    snapshot_date = _parse_date(snapshot_date_str)

    # Last day current stock can sustain = latest required delivery date
    latest_delivery_date = _add_days(snapshot_date, int(wks * 7))
    # Subtract lead time from that date = PO deadline
    latest_po_date = _add_days(latest_delivery_date, -TOTAL_LEAD_WEEKS * 7)
    overdue = latest_po_date < snapshot_date
    if overdue:
        latest_po_date = snapshot_date

    weeks_until_po = (latest_po_date - snapshot_date).total_seconds() / (7 * 24 * 3600)
    weeks_until_delivery = (latest_delivery_date - snapshot_date).total_seconds() / (7 * 24 * 3600)

    return {
        "recommended_latest_po_date": _format_date(latest_po_date),
        "recommended_latest_delivery_date": _format_date(latest_delivery_date),
        "recommended_latest_po_timing": "overdue" if overdue else _build_timing_label_from_weeks(weeks_until_po),
        "recommended_latest_delivery_timing": _build_timing_label_from_weeks(weeks_until_delivery),
    }


def compute_suggested_quantity_at_latest_delivery(
    current_stock_raw: Any,
    wks_to_oos_raw: Any,
) -> int | None:
    """
    Order quantity to cover 26 weeks after delivery date. Weekly demand is inferred as
    stock / weeks, so it needs both figures valid and above zero; otherwise None.
    """
    stock, wks = _stock(current_stock_raw), _wks(wks_to_oos_raw)
    if stock is None or wks is None or not (stock > 0 and wks > 0):
        return None
    weekly_demand = stock / wks
    target_stock_at_delivery = weekly_demand * COVERAGE_WEEKS_AFTER_DELIVERY
    order_qty = target_stock_at_delivery
    if order_qty <= 0:
        return 0
    return int(order_qty) + (1 if order_qty % 1 else 0)


def _system_notes(stock: float | None, wks: float | None, issues: list[dict], gap: float | None) -> list[str]:
    notes = []
    for issue in issues:
        if issue["status"] == "missing":
            notes.append(f"Data gap — {issue['field']} is missing; manual stock review needed.")
        else:
            notes.append(f"Invalid {issue['field']} value '{issue['raw']}' ({issue['reason']}); "
                         "manual review needed.")
    if wks is not None and gap:
        if wks == 0:
            notes.append(f"Critical: already out of stock; a replacement order takes about "
                         f"{TOTAL_LEAD_WEEKS} weeks to arrive.")
        else:
            notes.append(f"Critical: even if ordered today, a stockout of about {gap:g} weeks is "
                         f"expected (lead time {TOTAL_LEAD_WEEKS} weeks).")
    return notes


def build_item(snapshot_date: str, item_code: Any, item_name: Any,
               current_stock_raw: Any, wks_to_oos_raw: Any) -> dict[str, Any]:
    """One item in the shape the agents receive."""
    stock, stock_issue = parse_figure(current_stock_raw, "CurrentStock", MAX_PLAUSIBLE_STOCK)
    wks, wks_issue = _parse_wks(wks_to_oos_raw)
    issues = [i for i in (stock_issue, wks_issue) if i]
    rec = build_recommendations_for_item(snapshot_date, wks_to_oos_raw)
    gap = expected_stockout_weeks(wks_to_oos_raw)
    return {
        "item_code": str(item_code),
        "item_name": str(item_name),
        "risk_level": risk_level_from_weeks(wks_to_oos_raw),
        "current_stock": stock,
        "wks_to_oos": wks,
        "expected_stockout_weeks": gap,
        "suggested_quantity": compute_suggested_quantity_at_latest_delivery(current_stock_raw, wks_to_oos_raw),
        **rec,
        "data_issues": issues,
        "system_notes": _system_notes(stock, wks, issues, gap),
    }


DATA_ISSUES_HEADING = "## Data Issues Requiring Attention"

# Identity fields decide which supplier an item belongs to and which item it is. If one
# of them is unusable the row is not analysed at all; numeric fields are kept and flagged.
_SPREADSHEET_ERROR = re.compile(r"^#(REF!|N/A|VALUE!|NAME\?|DIV/0!|NULL!|NUM!|SPILL!|CALC!)$", re.I)


def _identity_issue(raw: Any, field: str) -> str | None:
    """Why an identity field cannot be used, or None when it can."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in _MISSING_TOKENS):
        return f"{field} is missing"
    if isinstance(raw, str) and _SPREADSHEET_ERROR.match(raw.strip()):
        return f"{field} holds the spreadsheet error '{raw.strip()}'"
    return None


def _read_row(row: dict[str, Any]) -> dict[str, Any]:
    stock = find_field(row, "currentstock")
    if stock is None:
        stock = find_field(row, "current_stock")
    wks = find_field(row, "wkstooos")
    if wks is None:
        wks = find_field(row, "wks_to_oos")
    # parse_csv_rows has already resolved one snapshot date for the whole file; rows
    # built by hand (tests, scripts) may still carry a snapshot_date field instead.
    return {
        "row": row.get(ROW_NUMBER_KEY),
        "snapshot_date": str(row.get(SNAPSHOT_DATE_KEY) or find_field(row, "snapshotdate") or "").strip(),
        "snapshot_source": row.get(SNAPSHOT_SOURCE_KEY),
        "snapshot_warnings": row.get(SNAPSHOT_WARNINGS_KEY) or [],
        "supplier_raw": find_field(row, "suppliername") or find_field(row, "supplier"),
        "item_code_raw": find_field(row, "itemcode"),
        "item_name_raw": find_field(row, "itemname") or find_field(row, "item_name"),
        "stock_raw": stock,
        "wks_raw": wks,
    }


def screen_rows(csv_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split rows into those that can be analysed and those that cannot. Checked in order:
      1. SupplierName, ItemCode or ItemName unusable
      2. the same ItemCode more than once for a supplier (all copies go, since it is
         unclear which is right; checked before step 3 so a bad copy cannot hide its twin)
      3. CurrentStock or WksToOOS missing or invalid (0 is a valid value for both)
    Each exclusion carries the sentence shown in the report.
    """
    kept, excluded = [], []
    for row in csv_rows:
        r = _read_row(row)
        where = f"Row {r['row']}" if r["row"] else "A row"
        reasons = [msg for msg in (
            _identity_issue(r["supplier_raw"], "SupplierName"),
            _identity_issue(r["item_code_raw"], "ItemCode"),
            _identity_issue(r["item_name_raw"], "ItemName"),
            None if r["snapshot_date"] else "the snapshot date is missing",
        ) if msg]
        supplier = None if _identity_issue(r["supplier_raw"], "SupplierName") else str(r["supplier_raw"]).strip()
        if reasons:
            excluded.append({"rows": [r["row"]], "supplier": supplier,
                             "line": f"{where} was excluded from the analysis: {'; '.join(reasons)}."})
            continue
        r["supplier"] = supplier
        r["item_code"] = str(r["item_code_raw"]).strip()
        r["item_name"] = str(r["item_name_raw"]).strip()
        kept.append(r)

    seen: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in kept:
        seen.setdefault((r["supplier"], r["item_code"]), []).append(r)
    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    for (supplier, code), copies in duplicates.items():
        rows = [c["row"] for c in copies]
        where = f"rows {', '.join(str(n) for n in rows)}" if all(rows) else f"{len(copies)} rows"
        excluded.append({"rows": rows, "supplier": supplier,
                         "line": f"ItemCode {code} appears more than once ({where}); all copies were "
                                 "excluded because it is unclear which is correct."})
    kept = [r for r in kept if (r["supplier"], r["item_code"]) not in duplicates]

    analysable = []
    for r in kept:
        issues = [i for i in (parse_figure(r["stock_raw"], "CurrentStock", MAX_PLAUSIBLE_STOCK)[1],
                              _parse_wks(r["wks_raw"])[1]) if i]
        if not issues:
            analysable.append(r)
            continue
        where = f"Row {r['row']}" if r["row"] else "A row"
        reasons = [f"{i['field']} is missing" if i["status"] == "missing"
                   else f"{i['field']} value '{i['raw']}' is invalid ({i['reason']})" for i in issues]
        excluded.append({"rows": [r["row"]], "supplier": r["supplier"],
                         "line": f"{where} ({r['item_name']}, {r['item_code']}) was excluded from the "
                                 f"analysis: {'; '.join(reasons)}."})
    excluded.sort(key=lambda e: min((n for n in e["rows"] if n), default=0))
    return analysable, excluded


def data_issue_lines(items: list[dict[str, Any]], snapshot_warnings: list[str] | None = None,
                     exclusion_lines: list[str] | None = None) -> list[str]:
    """
    One plain sentence per problem in the input data, for the report's opening section:
    snapshot date first, then excluded rows, then items with unusable figures.
    Empty when the data is clean, in which case the section is left out entirely.
    """
    lines = [f"Snapshot date: {w}" for w in (snapshot_warnings or [])]
    lines += exclusion_lines or []
    for it in items:
        label = f"{it.get('item_name')} ({it.get('item_code')})"
        issues = it.get("data_issues") or []
        if issues:
            parts = []
            for i in issues:
                if i["status"] == "missing":
                    parts.append(f"{i['field']} is missing")
                else:
                    parts.append(f"{i['field']} value '{i['raw']}' is invalid ({i['reason']})")
            fields = {i["field"] for i in issues}
            not_done = ("risk level, order dates and suggested quantity were not calculated"
                        if "WksToOOS" in fields else "suggested quantity was not calculated")
            lines.append(f"{label}: {'; '.join(parts)}. As a result, {not_done}.")
    return lines


def insert_data_issues_section(markdown: str, lines: list[str]) -> str:
    """Put the data-issues section before the document's first '## ' heading (after its header)."""
    if not lines:
        return markdown
    section = DATA_ISSUES_HEADING + "\n" + "\n".join(f"- {ln}" for ln in lines) + "\n\n"
    text = markdown or ""
    m = re.search(r"(?m)^## ", text)
    if not m:
        return section + text
    return text[:m.start()] + section + text[m.start():]


def group_by_supplier_and_recommend(
    csv_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Groups CSV rows by supplier and attaches
    per-item recommended dates and quantities.
    """
    groups: dict[str, dict[str, Any]] = {}
    kept, excluded = screen_rows(csv_rows)

    for r in kept:
        item_obj = build_item(r["snapshot_date"], r["item_code"], r["item_name"], r["stock_raw"], r["wks_raw"])
        if r["supplier"] not in groups:
            groups[r["supplier"]] = {
                "snapshot_date": r["snapshot_date"],
                "snapshot_date_source": r["snapshot_source"],
                "snapshot_date_warnings": r["snapshot_warnings"],
                "supplier": r["supplier"],
                "items": [],
            }
        groups[r["supplier"]]["items"].append(item_obj)

    # An exclusion belongs in its supplier's report. Rows with no usable supplier, or
    # whose supplier has nothing left to analyse, appear in every report so they are
    # never silently lost.
    unassigned = [(f"{e['supplier']}: " if e["supplier"] else "") + e["line"]
                  for e in excluded if e["supplier"] not in groups]
    for g in groups.values():
        own = [e for e in excluded if e["supplier"] == g["supplier"]]
        g["excluded_rows"] = own
        g["risk_level"] = group_risk_level(g["items"])
        g["data_issue_lines"] = data_issue_lines(
            g["items"], g["snapshot_date_warnings"], [e["line"] for e in own] + unassigned)
    return list(groups.values())
