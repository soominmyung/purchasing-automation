"""
Put the original 5-example SFT holdout into the same labelled format as
eval_data/eval_scenarios.jsonl, so both sets can be scored by one rubric.

These are the last 5 scenarios of the teacher dataset — the ones train_sft.py leaves
out (HOLDOUT_SIZE = 5). Unlike the generated evaluation set, each of them mixes
several issues at once, so several checks apply to the same scenario.

Labels were assigned by reading each scenario's histories and inventory:
  H1 SupplierTech      supplier: recent delays from transport strikes      item: GadgetY demand volatility
  H2 EuroSupply        supplier: recent price spike                        item: PartW lead time up
                       (ComponentZ's source RiskLevel "Low" against 2 weeks to stockout no longer
                        applies: RiskLevel is derived from WksToOOS, as production now does)
  H3 MegaManufacturers supplier: production halt, vague recovery timeline  item: MachineAddon spikes, DeviceConnector erratic supply
  H4 CraftySupplies    supplier: quality issues, no company named          item: ToolsetB swings, HandleGrip rise expected
                       data: ToolsetB stock and weeks-to-stockout are N/A
  H5 EnviroGoods       supplier: operations halted, no company named       item: PanelFiber boom demand, EcoBag frequent substitution

Run: python scripts/build_holdout_eval.py   (then scripts/set_eval_risk_mix.py)
Out: eval_data/holdout_scenarios.jsonl
"""
import ast
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.item_grouping import risk_level_from_weeks

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "training_data" / "teacher_dataset_20260302.jsonl"
OUT_FILE = ROOT / "eval_data" / "holdout_scenarios.jsonl"
HOLDOUT_SIZE = 5

LABELS = {
    "H1": dict(supplier_issue="material", item_issue="material", data_issue="none", history_present=True),
    "H2": dict(supplier_issue="material", item_issue="material", data_issue="none", history_present=True),
    "H3": dict(supplier_issue="material", item_issue="material", data_issue="none", history_present=True),
    "H4": dict(supplier_issue="material", item_issue="material", data_issue="missing_values", history_present=True),
    "H5": dict(supplier_issue="material", item_issue="material", data_issue="none", history_present=True),
}


def lit(x):
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return ast.literal_eval(x)


def main():
    examples = [json.loads(l) for l in open(TRAIN_FILE, encoding="utf-8") if l.strip()]
    holdout = examples[-HOLDOUT_SIZE:]
    rows = []
    for i, ex in enumerate(holdout, 1):
        sid = f"H{i}"
        inp = lit(ex["input"])
        inv = inp["inventory"]
        for it in inv:  # same derivation as the production pipeline
            it["RiskLevel"] = risk_level_from_weeks(it.get("WksToOOS"))
        rows.append({
            "scenario_id": sid,
            "category": "legacy_holdout",
            "labels": LABELS[sid],
            "supplier": inv[0]["SupplierName"] if inv else "",
            "input_data": inv,
            "supplier_history": inp.get("supplier_history", ""),
            "item_history": inp.get("item_history", ""),
            "source_index": len(examples) - HOLDOUT_SIZE + i - 1,
        })
    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        print(f"  {r['scenario_id']} {r['supplier']:<18} {r['labels']}")
    print(f"\n{len(rows)} holdout scenarios -> {OUT_FILE}")


if __name__ == "__main__":
    main()
