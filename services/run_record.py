"""
Run records: one JSON file per pipeline run, kept on the company server.

A record holds everything needed to explain a run afterwards and to track quality, cost
and failures across runs: the CSV exactly as received, who sent it (client IP), the
snapshot date and where it came from, per supplier the items as preprocessed, the history
documents the model read (their text as it was then), every output, the evaluation, each
email check, and per node the time taken, tokens per model and any error.

These files contain the company data that is kept out of LangSmith (docs/observability.md),
so the folder needs restricted access and a retention period. Aggregation across runs:
scripts/summarize_runs.py.
"""
import contextvars
import json
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from config import settings

RUNS_DIR = Path(__file__).resolve().parent.parent / settings.run_records_dir

# The node list of the supplier group being run, and the entry of the node running now.
# Set per group by RunRecord; read by the node wrapper and by the judge (note_usage).
_group_nodes: contextvars.ContextVar[list | None] = contextvars.ContextVar("group_nodes", default=None)
_current_node: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_node", default=None)


def new_run_id() -> str:
    return f"{datetime.now():%Y-%m-%d_%H%M%S}_{uuid.uuid4().hex[:4]}"


def client_ip(request: Any) -> str:
    """The requesting machine's address. Behind a proxy (the Cloud Run demo) the connection
    comes from the proxy, so the first X-Forwarded-For address is used when configured."""
    if request is None:
        return "unknown"
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def note_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Add token use to the node running now (for calls LangChain does not see, e.g. the judge)."""
    entry = _current_node.get()
    if entry is None:
        return
    u = entry["usage"].setdefault(model, {"input_tokens": 0, "output_tokens": 0})
    u["input_tokens"] += int(input_tokens or 0)
    u["output_tokens"] += int(output_tokens or 0)


def instrument(name: str, node: Callable) -> Callable:
    """Wrap a graph node so each call records its time, tokens per model and error."""
    from langchain_core.callbacks import get_usage_metadata_callback

    @wraps(node)
    def wrapper(state):
        nodes = _group_nodes.get()
        if nodes is None:  # not inside a recorded run (scripts, training data)
            return node(state)
        entry = {"node": name, "seconds": None, "usage": {}, "error": None}
        nodes.append(entry)
        token = _current_node.set(entry)
        started = time.perf_counter()
        try:
            with get_usage_metadata_callback() as cb:
                result = node(state)
            for model, u in (cb.usage_metadata or {}).items():
                note_usage(model, u.get("input_tokens", 0), u.get("output_tokens", 0))
            return result
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - started, 2)
            _current_node.reset(token)

    return wrapper


def _totals(nodes: list[dict]) -> dict[str, Any]:
    tokens: dict[str, dict[str, int]] = {}
    for n in nodes:
        for model, u in n["usage"].items():
            t = tokens.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
            t["input_tokens"] += u["input_tokens"]
            t["output_tokens"] += u["output_tokens"]
    return {"seconds": round(sum(n["seconds"] or 0 for n in nodes), 2), "tokens": tokens}


class RunRecord:
    """Collects one run and writes it to runs/<run_id>.json, also when the run fails."""

    def __init__(self, csv_content: str, csv_filename: str, ip: str, models: dict[str, str]):
        self.run_id = new_run_id()
        self.data: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "status": "running",
            "client_ip": ip,
            "prompt_version": settings.prompt_version,
            "models": models,
            "input": {"filename": csv_filename, "raw_csv": csv_content},
            "groups": [],
            "error": None,
        }

    def set_snapshot(self, rows: list[dict], snapshot_keys: tuple[str, str, str]) -> None:
        date_key, source_key, warnings_key = snapshot_keys
        first = rows[0] if rows else {}
        self.data["input"].update({"snapshot_date": first.get(date_key),
                                   "snapshot_source": first.get(source_key),
                                   "snapshot_warnings": first.get(warnings_key) or []})

    def start_group(self, group: dict) -> dict:
        entry = {
            "supplier": group["supplier"],
            "risk_level": group.get("risk_level"),
            "items": group["items"],
            "data_issue_lines": group.get("data_issue_lines", []),
            "status": "running",
            "nodes": [],
            "error": None,
        }
        self.data["groups"].append(entry)
        self._token = _group_nodes.set(entry["nodes"])
        return entry

    def finish_group(self, entry: dict, final_state: dict, files: dict[str, str], error: Exception | None = None) -> None:
        _group_nodes.reset(self._token)
        log = final_state.get("validation_log") or []
        entry.update({
            "status": "failed" if error else "completed",
            "error": f"{type(error).__name__}: {error}" if error else None,
            "history_docs": final_state.get("history_docs") or [],
            "outputs": {
                "analysis": final_state.get("analysis_output"),
                "report_md": final_state.get("report_md"),
                "approval": final_state.get("approval"),
                "pr_md": final_state.get("pr_md"),
                "email_text": final_state.get("email_text"),
                "email_md": final_state.get("email_md"),
                "files": files,
            },
            # the summary's limit is "about 100 words"; kept to see how often it is passed
            "summary_words": len(str((final_state.get("analysis_output") or {}).get("summary") or "").split()),
            "document_issues": final_state.get("document_issues") or [],
            "evaluation": final_state.get("evaluation_result"),
            "approval_check": final_state.get("approval_check"),
            "email_validation": {
                "attempts": final_state.get("iteration_count", 0),
                "passed": bool(final_state.get("is_valid_email")),
                "log": log,
            },
            "totals": _totals(entry["nodes"]),
        })

    def finish(self, error: Exception | None = None) -> Path | None:
        self.data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        if error is not None:
            self.data["status"] = "failed"
            self.data["error"] = f"{type(error).__name__}: {getattr(error, 'detail', None) or error}"
        else:
            failed = [g for g in self.data["groups"] if g["status"] != "completed"]
            self.data["status"] = "failed" if failed else "completed"
        if not settings.run_records_enabled:
            return None
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNS_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path
