"""
Spread the evaluation items across all four risk levels.

The scenarios were generated before the 18-week lead time was set, with WksToOOS mostly
between 1 and 15 weeks, so almost every item came out Critical. That left nothing to test
whether a model keeps non-Critical items calm, or separates Critical from non-Critical
items in the same supplier.

Each analysable item is given a target risk level below (in input order; rows with N/A
figures are skipped and stay excluded). An item already in its target band keeps its
weeks. Otherwise its weeks are drawn from the band with a fixed seed, and AvailableStock is
scaled by the same factor so the implied weekly demand, and so the consistency of the two
figures, is unchanged. Histories and labels are not touched.

Mix: 23 Critical / 13 High / 10 Medium / 7 Low out of 53 items. 7 scenarios have no
Critical item, 4 have only Critical items, 14 mix both.

The two high_stock_short_cover items (large stock, one week of cover) keep their figures.

Run after generate_eval_scenarios.py and build_holdout_eval.py:
     python scripts/set_eval_risk_mix.py
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.item_grouping import parse_figure, risk_level_from_weeks

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "eval_data" / "eval_scenarios.jsonl", ROOT / "eval_data" / "holdout_scenarios.jsonl"]

C, H, M, L = "Critical", "High", "Medium", "Low"
PLAN = {
    "eval_01": [M, L, H], "eval_02": [C, L], "eval_03": [H, M], "eval_04": [C, C],
    "eval_05": [C, M], "eval_06": [C, H], "eval_07": [L, M], "eval_08": [C, H, L],
    "eval_09": [H, H], "eval_10": [C, C], "eval_11": [C, M], "eval_12": [C, C, H],
    "eval_13": [M, L], "eval_14": [C, H], "eval_15": [C, C], "eval_16": [H, L],
    "eval_17": [C, M], "eval_18": [C, H, L], "eval_19": [C, M], "eval_20": [H, C],
    "H1": [C, M], "H2": [C, H], "H3": [C, C], "H4": [M], "H5": [H, C],
}
# Inclusive week ranges inside each band (Critical < 18 <= High < 26 <= Medium < 35 <= Low).
BANDS = {C: (2, 16), H: (18, 25), M: (26, 34), L: (36, 52)}
SEED = 20261018


def round_stock(x: float) -> int:
    return max(5, int(round(x / 5.0)) * 5)


def main():
    rng = random.Random(SEED)
    changed = 0
    for path in FILES:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            analysable = [it for it in r["input_data"]
                          if parse_figure(it.get("WksToOOS"), "WksToOOS", 520)[0] is not None
                          and parse_figure(it.get("AvailableStock"), "AvailableStock", 1e7)[0] is not None]
            targets = PLAN[r["scenario_id"]]
            if len(targets) != len(analysable):
                raise SystemExit(f"{r['scenario_id']}: plan has {len(targets)} levels for {len(analysable)} items")
            for it, target in zip(analysable, targets):
                if risk_level_from_weeks(it["WksToOOS"]) == target:
                    continue
                old_wks, old_stock = float(it["WksToOOS"]), float(it["AvailableStock"])
                new_wks = rng.randint(*BANDS[target])
                it["WksToOOS"] = new_wks
                it["AvailableStock"] = round_stock(old_stock * new_wks / old_wks)
                changed += 1
                print(f"  {r['scenario_id']:<8} {it['ItemName']:<26} {old_stock:>6.0f}/{old_wks:>4.0f}w"
                      f" -> {it['AvailableStock']:>6}/{new_wks:>3}w  {target}")
            for it in r["input_data"]:
                it["RiskLevel"] = risk_level_from_weeks(it.get("WksToOOS"))
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{changed} items changed")


if __name__ == "__main__":
    main()
