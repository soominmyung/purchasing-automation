"""Pydantic schemas: n8n workflow input/output structures.

Runtime contract — only RunPipelineRequest / RunPipelineResponse are enforced; they
validate the API boundary in routers/pipeline.py.

The remaining models document the shapes handed between agents inside the pipeline.
Those hand-offs are plain dicts at runtime: the analysis agent's output format is
enforced by its prompt, with a fallback in agents.py if parsing fails. They are kept
as a structural reference, not as validation.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ----- Item Grouping input (CSV row) -----
class ItemRow(BaseModel):
    """Single CSV row (flexible field name matching)."""
    item_code: str
    item_name: str
    risk_level: str = "N/A"
    current_stock: Optional[float] = None
    wks_to_oos: Optional[float] = None
    suggested_quantity: Optional[int] = None
    recommended_latest_po_date: Optional[str] = None
    recommended_latest_delivery_date: Optional[str] = None
    recommended_latest_po_timing: Optional[str] = None
    recommended_latest_delivery_timing: Optional[str] = None


# ----- Analysis Agent input -----
class AnalysisInputItem(BaseModel):
    item_code: str
    item_name: str
    risk_level: str = "N/A"
    current_stock: Optional[float] = None
    wks_to_oos: Optional[float] = None
    suggested_quantity: Optional[int] = None
    recommended_latest_po_date: Optional[str] = None
    recommended_latest_delivery_date: Optional[str] = None
    recommended_latest_po_timing: Optional[str] = None
    recommended_latest_delivery_timing: Optional[str] = None


class AnalysisInput(BaseModel):
    """JSON input for Analysis Agent (same shape as Item Grouping output)."""
    snapshot_date: str
    supplier: str
    items: list[AnalysisInputItem]


# ----- Analysis Agent output -----
class CriticalQuestion(BaseModel):
    target: Literal["general"] | str
    question: str
    reason: Literal["supplier_history", "item_history", "generic"]


class ReplenishmentTimelineItem(BaseModel):
    item_code: str
    item_name: str
    supplier: str
    risk_level: str
    current_stock: Optional[float] = None
    wks_to_oos: Optional[float] = None
    suggested_quantity: Optional[int] = None
    snapshot_date: str
    recommended_latest_po_timing: Optional[str] = None
    recommended_latest_delivery_timing: Optional[str] = None
    recommended_latest_po_date: Optional[str] = None
    recommended_latest_delivery_date: Optional[str] = None
    notes: Optional[str] = None


class AnalysisOutput(BaseModel):
    purchasing_report_markdown: str
    critical_questions: list[CriticalQuestion]
    replenishment_timeline: list[ReplenishmentTimelineItem]


# ----- API request/response -----
class RunPipelineRequest(BaseModel):
    """Pipeline input: CSV content + filename (for snapshot date extraction)."""
    csv_content: str
    csv_filename: Optional[str] = None  # e.g. Urgent_Stock_050425.csv → 2025-04-25


class RunPipelineResponse(BaseModel):
    """Pipeline execution result."""
    groups: list[dict[str, Any]] = Field(default_factory=list)  # Item Grouping result
    reports: list[dict[str, Any]] = Field(default_factory=list)  # snapshot_date_supplier → markdown
    requests: list[dict[str, Any]] = Field(default_factory=list)  # purchase request markdown
    emails: list[dict[str, Any]] = Field(default_factory=list)  # email draft text
    evaluations: list[dict[str, Any]] = Field(default_factory=list)  # AI quality report
