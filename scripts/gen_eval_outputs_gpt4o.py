"""
Run the production analysis agent (GPT-4o) on the evaluation scenarios.

GPT-4o is not the answer key here; it is one of the systems being scored, so it is run
exactly as the pipeline runs it:

  scenario rows -> services/item_grouping (risk level, Critical gap, code-written notes,
  exclusion of rows with unusable figures) -> run_analysis_agent with the scenario's
  histories injected as the tool results -> apply_system_notes, as analysis_node does

Full inputs and outputs are saved so the run can be re-scored under a different rubric.

Run: python scripts/gen_eval_outputs_gpt4o.py
Out: eval_results/eval_v2_gpt4o_outputs.jsonl
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agents import apply_system_notes, run_analysis_agent
from services.item_grouping import group_by_supplier_and_recommend
from services.prompts import PROMPT_VERSION
from utils.csv_utils import ROW_NUMBER_KEY, SNAPSHOT_DATE_KEY, SNAPSHOT_SOURCE_KEY

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_FILES = [ROOT / "eval_data" / "eval_scenarios.jsonl",
                  ROOT / "eval_data" / "holdout_scenarios.jsonl"]
OUT_FILE = ROOT / "eval_results" / "eval_v2_gpt4o_outputs.jsonl"

# Fixed so every run sees the same date; the training data used the same one.
SNAPSHOT_DATE = "2026-03-02"


def load_scenarios():
    rows = []
    for path in SCENARIO_FILES:
        rows += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows


def preprocess(scenario: dict) -> dict:
    """The scenario's inventory as the pipeline would receive it from a CSV, then grouped."""
    rows = []
    for n, it in enumerate(scenario["input_data"], start=2):  # line 1 is the CSV header
        rows.append({
            "ItemCode": it.get("ItemCode"), "ItemName": it.get("ItemName"),
            "SupplierName": it.get("SupplierName") or scenario["supplier"],
            "AvailableStock": it.get("AvailableStock"), "WksToOOS": it.get("WksToOOS"),
            SNAPSHOT_DATE_KEY: SNAPSHOT_DATE, SNAPSHOT_SOURCE_KEY: "fixed for evaluation",
            ROW_NUMBER_KEY: n,
        })
    groups = group_by_supplier_and_recommend(rows)
    if len(groups) != 1:
        raise ValueError(f"{scenario['scenario_id']}: expected one supplier group, got {len(groups)}")
    return groups[0]


def main():
    scenarios = load_scenarios()
    done = set()
    if OUT_FILE.exists():
        done = {json.loads(l)["scenario_id"] for l in open(OUT_FILE, encoding="utf-8") if l.strip()}

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for s in scenarios:
            if s["scenario_id"] in done:
                continue
            group = preprocess(s)
            input_json = {"snapshot_date": SNAPSHOT_DATE, "supplier": group["supplier"], "items": group["items"]}
            started = time.time()
            analysis, used_sh, used_ih = run_analysis_agent(
                input_json,
                supplier_history_override=s["supplier_history"],
                item_history_override=s["item_history"],
            )
            seconds = round(time.time() - started, 1)
            analysis = apply_system_notes(analysis, group["items"])
            f.write(json.dumps({
                "scenario_id": s["scenario_id"],
                "model": "gpt-4o",
                "temperature": 0,
                "prompt_version": PROMPT_VERSION,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "seconds": seconds,
                "input": input_json,
                "data_issue_lines": group["data_issue_lines"],
                "supplier_history": used_sh,
                "item_history": used_ih,
                "output": analysis,
            }, ensure_ascii=False) + "\n")
            f.flush()
            qs = analysis.get("critical_questions") if isinstance(analysis, dict) else None
            crit = sum(1 for it in group["items"] if it["risk_level"] == "Critical")
            print(f"  {s['scenario_id']:<8} {s['supplier']:<26} items={len(group['items'])} "
                  f"critical={crit} excluded={len(group['excluded_rows'])} "
                  f"questions={len(qs) if isinstance(qs, list) else 'n/a'} {seconds}s")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    main()
