"""
Item Grouping, Suggested dates & quantities (n8n Code 노드 로직 Python 포팅).
- CSV 행을 supplier별로 그룹화
- 품목별 recommended_latest_po_date, recommended_latest_delivery_date, timing 라벨 계산
- suggested_quantity: 최근 납기일 이후 26주 커버용 수량
"""
from datetime import datetime, timedelta
from typing import Any

from utils.csv_utils import find_field

# n8n 상수와 동일
IMPORT_LEAD_WEEKS = 16
INTERNAL_LEAD_WEEKS = 2
TOTAL_LEAD_WEEKS = IMPORT_LEAD_WEEKS + INTERNAL_LEAD_WEEKS
COVERAGE_WEEKS_AFTER_DELIVERY = 26


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


def build_recommendations_for_item(snapshot_date_str: str, wks_to_oos_raw: Any) -> dict[str, Any]:
    """품목별 권장 PO일/납기일 및 timing 라벨."""
    snapshot_date = _parse_date(snapshot_date_str)
    try:
        wks = float(wks_to_oos_raw)
    except (TypeError, ValueError):
        wks = TOTAL_LEAD_WEEKS
    effective_wks_to_oos = wks if wks > 0 else TOTAL_LEAD_WEEKS

    # 현재 재고로 버틸 수 있는 마지막 날 = 최종 납기 필요일
    latest_delivery_date = _add_days(snapshot_date, int(effective_wks_to_oos * 7))
    # 그 날에서 리드타임만큼 역산 = PO 발주 마감일
    latest_po_date = _add_days(latest_delivery_date, -TOTAL_LEAD_WEEKS * 7)
    if latest_po_date < snapshot_date:
        latest_po_date = snapshot_date

    weeks_until_po = (latest_po_date - snapshot_date).total_seconds() / (7 * 24 * 3600)
    weeks_until_delivery = (latest_delivery_date - snapshot_date).total_seconds() / (7 * 24 * 3600)

    return {
        "recommended_latest_po_date": _format_date(latest_po_date),
        "recommended_latest_delivery_date": _format_date(latest_delivery_date),
        "recommended_latest_po_timing": _build_timing_label_from_weeks(weeks_until_po),
        "recommended_latest_delivery_timing": _build_timing_label_from_weeks(weeks_until_delivery),
    }


def compute_suggested_quantity_at_latest_delivery(
    current_stock_raw: Any,
    wks_to_oos_raw: Any,
) -> int | None:
    """납기일 이후 26주 커버용 주문 수량."""
    try:
        stock = float(current_stock_raw)
        wks = float(wks_to_oos_raw)
    except (TypeError, ValueError):
        return None
    if not (stock > 0 and wks > 0):
        return None
    weekly_demand = stock / wks
    target_stock_at_delivery = weekly_demand * COVERAGE_WEEKS_AFTER_DELIVERY
    order_qty = target_stock_at_delivery
    if order_qty <= 0:
        return 0
    return int(order_qty) + (1 if order_qty % 1 else 0)


def group_by_supplier_and_recommend(
    csv_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    CSV 파싱 결과(각 행에 snapshot_date 등)를 supplier별로 그룹화하고,
    품목별 권장일/수량을 붙여서 반환.
    """
    groups: dict[str, dict[str, Any]] = {}

    for row in csv_rows:
        snapshot_date = str(find_field(row, "snapshotdate") or find_field(row, "snapshot_date") or "").strip()
        supplier = str(find_field(row, "suppliername") or find_field(row, "supplier") or "").strip()
        risk_level = str(find_field(row, "risklevel") or find_field(row, "risk_level") or "").strip()
        item_code_raw = find_field(row, "itemcode")
        item_name = find_field(row, "itemname") or find_field(row, "item_name")
        current_stock_raw = find_field(row, "currentstock") or find_field(row, "current_stock")
        wks_to_oos_raw = find_field(row, "wkstooos") or find_field(row, "wks_to_oos")

        if not snapshot_date or not supplier or item_code_raw is None or item_name is None:
            continue

        rec = build_recommendations_for_item(snapshot_date, wks_to_oos_raw)
        suggested_qty = compute_suggested_quantity_at_latest_delivery(current_stock_raw, wks_to_oos_raw)

        item_obj = {
            "item_code": str(item_code_raw),
            "item_name": str(item_name),
            "risk_level": risk_level or "N/A",
            "current_stock": float(current_stock_raw) if current_stock_raw is not None else None,
            "wks_to_oos": float(wks_to_oos_raw) if wks_to_oos_raw is not None else None,
            "suggested_quantity": suggested_qty,
            "recommended_latest_po_date": rec["recommended_latest_po_date"],
            "recommended_latest_delivery_date": rec["recommended_latest_delivery_date"],
            "recommended_latest_po_timing": rec["recommended_latest_po_timing"],
            "recommended_latest_delivery_timing": rec["recommended_latest_delivery_timing"],
        }

        if supplier not in groups:
            groups[supplier] = {
                "snapshot_date": snapshot_date,
                "supplier": supplier,
                "items": [],
            }
        groups[supplier]["items"].append(item_obj)

    return list(groups.values())
