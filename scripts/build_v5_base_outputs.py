"""
Base outputs for judge validation, prompt v5: the production analysis agent and approval agent
on the 25 evaluation scenarios (eval_data/eval_scenarios.jsonl and holdout_scenarios.jsonl).

Each scenario goes through the production preprocessing (scripts/gen_eval_outputs_gpt4o.py,
preprocess), then one analysis call with the scenario's history documents in place of retrieval,
then the purchase request's narrative from the same input the pipeline gives it. No checks and no
rewrites: these are the outputs the calibration set is built from (seeded faults are edits of
them; hand scoring judges them as they are), so they are kept exactly as the models wrote them.
Every prompt's text is saved with the run.

Run: python scripts/build_v5_base_outputs.py
Out: eval_results/v5_base_outputs.jsonl (one line per scenario; resumes where it stopped)
  --timing: after the analysis prompt was told to say when each check action comes (before the
  order, alongside it or after it; 2026-10-24): eval_results/v5_base_outputs_timing.jsonl
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gen_eval_outputs_gpt4o import SNAPSHOT_DATE, load_scenarios, preprocess  # noqa: E402
from services import documents  # noqa: E402
from services.agents import AGENT_MODELS, run_analysis_agent, run_approval_agent  # noqa: E402
from services.prompts import ANALYSIS_AGENT_SYSTEM, APPROVAL_AGENT_SYSTEM, PROMPT_VERSION  # noqa: E402

OUT_FILE = ROOT / "eval_results" / "v5_base_outputs.jsonl"
if "--timing" in sys.argv:  # 2026-10-24: analysis prompt asks each check action to say when it comes
    OUT_FILE = ROOT / "eval_results" / "v5_base_outputs_timing.jsonl"


def main() -> None:
    done = set()
    if OUT_FILE.exists():
        done = {json.loads(l)["scenario_id"] for l in open(OUT_FILE, encoding="utf-8") if l.strip()}
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for s in load_scenarios():
            if s["scenario_id"] in done:
                continue
            group = preprocess(s)
            input_json = {"snapshot_date": SNAPSHOT_DATE, "supplier": group["supplier"], "items": group["items"]}
            started = time.time()
            analysis, docs = run_analysis_agent(input_json, history_override=s["history"])
            analysis_seconds = round(time.time() - started, 1)
            approval, payload = None, {}
            if isinstance(analysis, dict) and "unparsed_output" not in analysis and documents.requested_items(group["items"]):
                payload = documents.approval_input(group["supplier"], group["items"], analysis, docs, SNAPSHOT_DATE)
                approval = run_approval_agent(payload)
            f.write(json.dumps({
                "scenario_id": s["scenario_id"], "category": s.get("category"), "labels": s.get("labels"),
                "models": {"analysis": AGENT_MODELS["analysis"], "approval": AGENT_MODELS["approval"]},
                "prompt_version": PROMPT_VERSION,
                "prompts": {"analysis": ANALYSIS_AGENT_SYSTEM, "approval_request": APPROVAL_AGENT_SYSTEM},
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "analysis_seconds": analysis_seconds,
                "input": input_json, "data_issue_lines": group["data_issue_lines"], "history_docs": docs,
                "analysis": analysis, "approval_input": payload, "approval": approval,
            }, ensure_ascii=False) + "\n")
            f.flush()
            ok = isinstance(analysis, dict) and "unparsed_output" not in analysis
            print(f"  {s['scenario_id']:<8} {group['supplier']:<26} items={len(group['items'])} "
                  f"{'ok' if ok else 'UNPARSED'} {analysis_seconds}s approval={'yes' if approval else 'no'}", flush=True)
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    main()
