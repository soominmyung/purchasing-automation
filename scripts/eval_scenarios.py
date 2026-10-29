"""
The 25 evaluation scenarios (eval_data/eval_scenarios.jsonl and holdout_scenarios.jsonl), read and
turned into the input the pipeline would build from a stock CSV: each scenario's rows go through
the production preprocessing (services/item_grouping: risk level, gap, order and delivery dates,
suggested quantity, rows with unusable figures excluded).

Shared by the scripts that run models on the scenarios (build_v5_base_outputs, screen_base_models,
build_sft_teacher_data, bench_vllm_v5).
"""
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.item_grouping import group_by_supplier_and_recommend
from utils.csv_utils import ROW_NUMBER_KEY, SNAPSHOT_DATE_KEY, SNAPSHOT_SOURCE_KEY

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_FILES = [ROOT / "eval_data" / "eval_scenarios.jsonl",
                  ROOT / "eval_data" / "holdout_scenarios.jsonl"]

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
