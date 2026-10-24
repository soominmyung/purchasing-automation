"""
Run a development case (dev/cases/) and keep the record in dev/runs/.

Development cases are for working on prompts and the pipeline; they are never used to
compare models (that is the evaluation set, eval_data/). Every run is kept, with the exact
prompt text it used, because prompts are edited in place while a version is being written.

Run:
  python scripts/run_dev_case.py dev/cases/nordic_fasteners.json --label "what changed"
      the whole pipeline once, printing each node as it finishes
  python scripts/run_dev_case.py dev/cases/nordic_fasteners.json --label "..." \\
         --email-only 5 --from dev/runs/<folder>
      the email agent alone, five times, on the analysis of an earlier run

Out: dev/runs/<date_time>_<label>/record.json, plus each document as .md and .docx
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from services import documents, run_record  # noqa: E402
from services.agents import AGENT_EFFORT, AGENT_MODELS, purchasing_graph, run_email_draft_agent  # noqa: E402
from services.item_grouping import group_by_supplier_and_recommend, group_risk_level  # noqa: E402
from services.prompts import PROMPTS_DIR  # noqa: E402
from utils.csv_utils import parse_csv_rows  # noqa: E402
from utils.docx_utils import save_markdown_to_docx  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent.parent / "dev" / "runs"
DOCUMENTS = (("analysis_report", "report_md"), ("purchase_request", "pr_md"),
             ("email_draft", "email_md"), ("evaluation", "evaluation_md"))


def prompts_used() -> dict[str, str]:
    """The exact text of every prompt of the version in use."""
    folder = PROMPTS_DIR / settings.prompt_version
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))}


def settings_used() -> dict:
    keys = ("prompt_version", "evaluator_model", "analysis_rewrites", "llm_timeout_seconds", "llm_retries",
            "judge_timeout_seconds", "judge_retries", "run_date")
    return {k: getattr(settings, k, None) for k in keys}


def new_folder(label: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:50] or "run"
    folder = RUNS_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{slug}"
    folder.mkdir(parents=True)
    return folder


def load_case(path: Path) -> tuple[dict, dict]:
    case = json.loads(path.read_text(encoding="utf-8"))
    group = group_by_supplier_and_recommend(parse_csv_rows(case["csv"], f"{case['name']}.csv"))[0]
    return case, group


def full_run(case: dict, group: dict, folder: Path) -> dict:
    nodes: list[dict] = []
    run_record._group_nodes.set(nodes)
    state0 = {"snapshot_date": group["snapshot_date"], "supplier": group["supplier"], "items": group["items"],
              "risk_level": group_risk_level(group["items"]), "iteration_count": 0, "correction_feedback": "",
              "is_valid_email": False, "history_override": case["history"],
              "data_issue_lines": group.get("data_issue_lines", []), "run_id": folder.name}
    state, started = dict(state0), time.perf_counter()
    for event in purchasing_graph.stream(state0):
        for node, update in event.items():
            print(f"{time.perf_counter() - started:6.1f}s  {node} done", flush=True)
            for k, v in (update or {}).items():
                if k in ("validation_log", "document_issues", "analysis_attempts"):
                    state.setdefault(k, []).extend(v)
                else:
                    state[k] = v
    for name, key in DOCUMENTS:
        (folder / f"{name}.md").write_text(state.get(key) or "", encoding="utf-8")
        save_markdown_to_docx(folder / f"{name}.docx", state.get(key) or "")
    return {
        "seconds": round(time.perf_counter() - started, 1),
        "nodes": nodes,
        "outputs": {"analysis": state.get("analysis_output"), "approval": state.get("approval"),
                    "email_text": state.get("email_text"),
                    **{key: state.get(key) for _, key in DOCUMENTS}},
        "evaluation": state.get("evaluation_result"),
        "approval_check": state.get("approval_check"),
        "analysis_attempts": state.get("analysis_attempts"),
        "analysis_used_attempt": state.get("analysis_used_attempt"),
        "email_validation": state.get("validation_log"),
        "document_issues": state.get("document_issues"),
    }


def email_only(group: dict, earlier: Path, times: int, folder: Path) -> dict:
    analysis = json.loads((earlier / "record.json").read_text(encoding="utf-8"))["outputs"]["analysis"]
    payload = documents.email_input(group["snapshot_date"], group["supplier"], group["items"], analysis)
    runs = []
    for n in range(1, times + 1):
        started = time.perf_counter()
        text = run_email_draft_agent(payload)
        md, problems = documents.render_email(text, group["items"], analysis, group["supplier"])
        runs.append({"seconds": round(time.perf_counter() - started, 1), "email_text": text,
                     "email_md": md, "document_issues": problems})
        (folder / f"email_draft_{n}.md").write_text(md, encoding="utf-8")
        print(f"--- {n}\n{md}", flush=True)
    return {"from_run": earlier.name, "email_input": payload, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--label", required=True, help="what was changed before this run")
    ap.add_argument("--email-only", type=int, default=0, metavar="N")
    ap.add_argument("--from", dest="earlier", type=Path, help="run folder whose analysis the email uses")
    ap.add_argument("--analysis-model", help="analysis agent's model for this run (its rewrites too), "
                                             "e.g. gpt-6.1-sol or claude-sonnet-5-5")
    ap.add_argument("--analysis-effort", help="analysis agent's reasoning effort (low, medium, high, ...; "
                                              "'default' leaves it to the API)")
    args = ap.parse_args()
    if args.analysis_model:
        AGENT_MODELS["analysis"] = args.analysis_model  # recorded in the run's "models"
    if args.analysis_effort:
        AGENT_EFFORT["analysis"] = None if args.analysis_effort == "default" else args.analysis_effort
    case, group = load_case(args.case)
    folder = new_folder(args.label)
    record = {"label": args.label, "case": case["name"], "started": datetime.now().isoformat(timespec="seconds"),
              "kind": "email only" if args.email_only else "full run",
              "settings": settings_used(), "models": AGENT_MODELS, "efforts": AGENT_EFFORT, "prompts": prompts_used()}
    if args.email_only:
        if not args.earlier:
            ap.error("--email-only needs --from <run folder>")
        record.update(email_only(group, args.earlier, args.email_only, folder))
    else:
        record.update(full_run(case, group, folder))
    (folder / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str),
                                        encoding="utf-8")
    print("record:", folder)


if __name__ == "__main__":
    main()
