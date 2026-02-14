"""Pydantic 스키마: n8n 워크플로 입출력 구조."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ----- Item Grouping 입력 (CSV row) -----
class ItemRow(BaseModel):
    """CSV 한 행 (필드명 유연 매칭용)."""
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


# ----- Analysis Agent 입력 -----
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
    """Analysis Agent에 넣는 JSON (Item Grouping 출력과 동일)."""
    snapshot_date: str
    supplier: str
    items: list[AnalysisInputItem]


# ----- Analysis Agent 출력 -----
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


# ----- API 요청/응답 -----
class RunPipelineRequest(BaseModel):
    """파이프라인 실행: CSV 내용 + 파일명(스냅샷 날짜 추출용)."""
    csv_content: str
    csv_filename: Optional[str] = None  # e.g. Urgent_Stock_050425.csv → 2025-04-25


class RunPipelineResponse(BaseModel):
    """파이프라인 실행 결과."""
    groups: list[dict[str, Any]] = Field(default_factory=list)  # Item Grouping 결과
    reports: list[dict[str, Any]] = Field(default_factory=list)  # snapshot_date_supplier → markdown
    requests: list[dict[str, Any]] = Field(default_factory=list)  # purchase request markdown
    emails: list[dict[str, Any]] = Field(default_factory=list)  # email draft text
