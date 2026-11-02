"""Pydantic schemas: the API's request and response structures, validated at the API boundary in
routers/pipeline.py. What passes between the agents inside the pipeline is plain dicts: the
analysis agent's output is described by its prompt (prompts/v5/analysis.md) and checked by the
code checks (services/quality_checks.py).
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunPipelineRequest(BaseModel):
    """Pipeline input: CSV content + filename (for snapshot date extraction)."""
    csv_content: str
    csv_filename: Optional[str] = None  # e.g. Urgent_Stock_050425.csv → 2025-04-05


class RunPipelineResponse(BaseModel):
    """Pipeline execution result."""
    groups: list[dict[str, Any]] = Field(default_factory=list)  # Item Grouping result
    reports: list[dict[str, Any]] = Field(default_factory=list)  # snapshot_date_supplier → markdown
    requests: list[dict[str, Any]] = Field(default_factory=list)  # purchase request markdown
    emails: list[dict[str, Any]] = Field(default_factory=list)  # email draft text
    evaluations: list[dict[str, Any]] = Field(default_factory=list)  # AI quality report
    run_id: Optional[str] = None  # the run record's id (runs/<run_id>.json)
