"""
Tables for the Gemma 4 12B SFT results: every model on the 25
evaluation cases (prompt with the figures_after_changes rule, tag fac-rule) and on the tie-in
generalisation set (tag tie-in).

Measures, per model:
  ready              first answers passing every blocking check
  changes applied    of the cases where Sol changed a lead time, those where the model set a lead
                     time other than 18 weeks too (an item listed at 18 weeks is not a change)
  items wrong        of Sol's changed items the model also changed, those with a figure C10 found wrong
  failures           blocking failures by check

Run: python scripts/report_sft_results.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "eval_results"


def load(name: str, tag: str) -> dict:
    p = R / f"sft_screen_{name}_{tag}.jsonl"
    return {json.loads(l)["scenario_id"]: json.loads(l) for l in open(p, encoding="utf-8")} if p.exists() else {}


def changed(analysis: dict) -> dict:
    out = {}
    for x in (analysis or {}).get("lead_time_changes") or []:
        w = x.get("lead_time_weeks") if isinstance(x, dict) else None
        if isinstance(w, (int, float)) and w != 18:
            for c in x.get("items") or []:
                out[str(c)] = w
    return out


def row(name: str, rows: dict, ref: dict) -> dict:
    ev = [r["evaluation"] for r in rows.values()]
    ref_cases = {k: changed(r["analysis"]) for k, r in ref.items() if changed(r["analysis"])}
    cases = items = bad = 0
    for k, sc in ref_cases.items():
        r = rows.get(k)
        if not r:
            continue
        mc = changed(r["analysis"])
        cases += bool(set(sc) & set(mc))
        d = str(r["evaluation"]["checks"].get("C10", {}).get("detail", ""))
        wrong = set(re.findall(r"(\d+) new_\w+ [^;]*?by the rules", d))
        for it in sc:
            if it in mc:
                items += 1
                bad += it in wrong
    secs = sorted(r["analysis_seconds"] for r in rows.values()) or [0]
    return {"model": name, "n": len(rows), "ready": sum(e["verdict"] == "ready" for e in ev),
            "ref_cases": len(ref_cases), "cases": cases, "items": items, "bad": bad,
            "fails": Counter(c for e in ev for c in e["failed"]), "median_s": secs[len(secs) // 2]}


def table(tag: str, names: list[str], ref_name: str, ref_rows: dict) -> None:
    print(f"\n### {tag}\n")
    print("| Model | Ready | Lead-time changes applied (of Sol's) | Changed items with a wrong figure | Blocking failures by check | Time (median) |")
    print("|---|---|---|---|---|---|")
    for n in [ref_name] + names:
        rows = ref_rows if n == ref_name else load(n, tag)
        if not rows:
            continue
        x = row(n, rows, ref_rows)
        f = ", ".join(f"{k} {v}" for k, v in x["fails"].most_common()) or "none"
        print(f"| {n} | {x['ready']}/{x['n']} | {x['cases']}/{x['ref_cases']} | {x['bad']}/{x['items']} | {f} | {x['median_s']} s |")


def main() -> None:
    sizes = ["sft-n25_s1", "sft-n25_s2", "sft-n50_s1", "sft-n50_s2", "sft-n85_s1", "sft-n85_s2"]
    table("fac-rule", ["gemma-4-12b-it"] + sizes, "gpt-6.1-sol", load("gpt-6.1-sol", "fac-rule"))
    # tie-in: Sol's runs are the teacher file for the generalisation set
    tie = {}
    p = ROOT / "training_data" / "sft_v5" / "teacher_generalisation_tie_in.jsonl"
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        tie[r["scenario_id"]] = {**r, "analysis_seconds": r["seconds"]}
    table("tie-in", ["gemma-4-12b-it"] + sizes, "gpt-6.1-sol", tie)


if __name__ == "__main__":
    main()
