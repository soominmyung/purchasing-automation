"""
MCP server exposing the pipeline's stock figures, worked out by the pipeline's own code.

In the pipeline the figures are calculated by code and handed to the analysis agent; the agent
never works a date or a quantity out itself. When the analysis finds a lead time other than
18 weeks (a supplier notice, a port delay), code recalculates the risk level and the latest
order date at that lead time (services/documents.with_lead_times). This server gives the same
figures to any MCP client: for a stock CSV as the pipeline reads it, or for one item at any
lead time. Nothing here calls a model, and no prompt is published: the analysis prompt is
written for the pipeline's input, not for a client's conversation.

Run: python mcp_servers/replenishment_server.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config reads .env from the working directory; an MCP client starts us anywhere

from mcp.server.mcpserver import MCPServer

from services.documents import items_after_changes, with_lead_times
from services.item_grouping import (
    COVERAGE_WEEKS_AFTER_DELIVERY,
    HIGH_MARGIN_WEEKS,
    IMPORT_LEAD_WEEKS,
    INTERNAL_LEAD_WEEKS,
    MEDIUM_MARGIN_WEEKS,
    TOTAL_LEAD_WEEKS,
    build_item,
    group_by_supplier_and_recommend,
)
from utils.csv_utils import parse_csv_rows

mcp = MCPServer(
    name="purchasing-replenishment",
    instructions=(
        f"Stock figures as the purchasing pipeline calculates them: risk level, latest delivery and "
        f"order dates, the expected stockout gap and the suggested quantity. The standard lead time is "
        f"{TOTAL_LEAD_WEEKS} weeks; give another lead time when a supplier or the history says it differs."
    ),
)

FIELDS = ("item_code", "item_name", "available_stock", "wks_to_oos", "lead_time_weeks", "risk_level",
          "recommended_latest_delivery_date", "recommended_latest_po_date", "recommended_latest_po_timing",
          "po_should_have", "expected_stockout_weeks", "suggested_quantity", "system_notes")


def _figures(item: dict, snapshot_date: str, lead_time_weeks: float | None) -> dict:
    changes = {"lead_time_changes": [{"items": [item["item_code"]], "lead_time_weeks": lead_time_weeks}]
               if lead_time_weeks else []}
    out = items_after_changes([item], changes, snapshot_date)[0]
    out["po_should_have"] = with_lead_times([item], changes, snapshot_date)[0].get("po_should_have")
    if lead_time_weeks and lead_time_weeks != TOTAL_LEAD_WEEKS:
        out["system_notes"] = []  # the pipeline's notes are written for the 18-week figures
    return {k: out[k] for k in FIELDS if out.get(k) not in (None, [], "")}


@mcp.tool(
    description=(
        "Figures for one item: risk level, latest delivery date (when stock runs out), latest order "
        "date and how soon it is, weeks out of stock even if ordered now, and the quantity that covers "
        f"{COVERAGE_WEEKS_AFTER_DELIVERY} weeks after delivery. snapshot_date is YYYY-MM-DD; "
        f"lead_time_weeks defaults to {TOTAL_LEAD_WEEKS}. An order date that has already passed is "
        "given as the snapshot date, with po_should_have saying when it should have been placed."
    )
)
def item_figures(snapshot_date: str, available_stock: float, weeks_to_out_of_stock: float,
                 item_code: str = "", item_name: str = "", lead_time_weeks: float | None = None) -> dict:
    item = build_item(snapshot_date, item_code or "item", item_name or item_code or "item",
                      available_stock, weeks_to_out_of_stock)
    return _figures(item, snapshot_date, lead_time_weeks)


@mcp.tool(
    description=(
        "Figures for every item in a stock CSV, as the pipeline reads it (a SnapshotDate line, then "
        "ItemCode, ItemName, SupplierName, AvailableStock, WksToOOS), grouped by supplier, most urgent "
        "supplier first. Rows the pipeline cannot use are listed under data_issues."
    )
)
def stock_csv_figures(csv_text: str) -> list[dict]:
    groups = group_by_supplier_and_recommend(parse_csv_rows(csv_text, "stock.csv"))
    return [{"supplier": g["supplier"], "snapshot_date": g["snapshot_date"], "risk_level": g["risk_level"],
             "items": [_figures(it, g["snapshot_date"], None) for it in g["items"]],
             **({"data_issues": g["data_issue_lines"]} if g.get("data_issue_lines") else {})}
            for g in groups]


@mcp.tool(description="The lead time, risk thresholds and coverage the figures are built on.")
def replenishment_policy() -> dict:
    return {
        "lead_time_weeks": {"total": TOTAL_LEAD_WEEKS, "import": IMPORT_LEAD_WEEKS, "internal": INTERNAL_LEAD_WEEKS},
        "risk_levels": {
            "Critical": "weeks to stockout below the lead time: a stockout even if ordered today",
            "High": f"below the lead time + {HIGH_MARGIN_WEEKS} weeks",
            "Medium": f"below the lead time + {MEDIUM_MARGIN_WEEKS} weeks",
            "Low": "otherwise; no purchase request is prepared for a Low item",
        },
        "coverage_weeks_after_delivery": COVERAGE_WEEKS_AFTER_DELIVERY,
        "latest_order_date": "the latest delivery date (when stock runs out) minus the lead time",
    }


if __name__ == "__main__":
    mcp.run()
