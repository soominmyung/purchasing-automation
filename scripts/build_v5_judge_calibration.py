"""
Calibration set for judge validation, prompt v5 (docs/judge_validation.md).

Seeded faults: each is one base output with one deliberate fault aimed at one judge check, so the
right verdict for that check is known to be "fail". Every fault is written out below as an exact
edit (the text before and after, or the field changed), so each can be read and reviewed; the build
stops if the text to change is not found. After the edit the code checks (C1-C10) are run again: a
fault code would catch first is dropped, since the judge would never see it. A fault that changes a
lead time also gets its figures after the change worked out again by the pipeline's rules, so C10
passes and the fault is the one intended.

Real cases: outputs from development runs on which the judge was seen to be wrong or right, with
the verdict a person settled (2026-10-24).

Bases: eval_results/v5_base_outputs.jsonl (gpt-6.1-sol on the 25 evaluation scenarios), and two
Sol development runs whose lead-time change rests on a stated size (Kestrel: four weeks longer).

Run: python scripts/build_v5_judge_calibration.py
Out: eval_data/judge_calibration_v5.jsonl
"""
import copy
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import quality_checks as q  # noqa: E402
from services.item_grouping import group_by_supplier_and_recommend  # noqa: E402
from utils.csv_utils import parse_csv_rows  # noqa: E402

OUT = ROOT / "eval_data" / "judge_calibration_v5.jsonl"
BLOCKING_CODE = [k for k in q.CHECK_NAMES if k.startswith("C") and q.SEVERITY[k] == q.BLOCKING]


# ----------------------------------------------------------------------------- bases

def dev_base(pattern: str, case: str) -> dict:
    d = sorted(glob.glob(str(ROOT / "dev" / "runs" / pattern)))[0]
    r = json.loads(Path(d, "record.json").read_text(encoding="utf-8"))
    c = json.loads((ROOT / "dev" / "cases" / f"{case}.json").read_text(encoding="utf-8"))
    g = group_by_supplier_and_recommend(parse_csv_rows(c["csv"], f"{c['name']}.csv"))[0]
    return {"base": f"dev:{Path(d).name}", "snapshot_date": g["snapshot_date"], "supplier": g["supplier"],
            "items": g["items"], "history_docs": c["history"], "analysis": r["outputs"]["analysis"],
            "approval_input": None, "approval": None, "record": r}


def load_bases() -> dict[str, dict]:
    bases = {}
    for line in open(ROOT / "eval_results" / "v5_base_outputs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        bases[r["scenario_id"]] = {"base": r["scenario_id"], "snapshot_date": r["input"]["snapshot_date"],
                                   "supplier": r["input"]["supplier"], "items": r["input"]["items"],
                                   "history_docs": r["history_docs"], "analysis": r["analysis"],
                                   "approval_input": r["approval_input"], "approval": r["approval"]}
    bases["kestrel"] = dev_base("*sol-repeat-1-*", "kestrel_components")
    bases["nordic"] = dev_base("*sol-repeat-1-*", "nordic_fasteners")
    if bases["kestrel"]["record"]["case"] != "kestrel_components":  # the first sol-repeat-1 run is Kestrel
        raise SystemExit("unexpected dev run order")
    nordic = sorted(glob.glob(str(ROOT / "dev" / "runs" / "*sol-repeat-1-*")))[1]
    bases["nordic"] = dev_base(Path(nordic).name, "nordic_fasteners")
    return bases


# ----------------------------------------------------------------------------- edit helpers

def finding(a: dict, fid: int) -> dict:
    return next(f for f in a["findings"] if f["id"] == fid)


def replace_in(obj: dict, key: str, old: str, new: str) -> dict:
    text = obj[key]
    if text.count(old) != 1:
        raise SystemExit(f"text to change not found once: {old!r} in {text[:120]!r}")
    obj[key] = text.replace(old, new)
    return {"field": key, "before": old, "after": new}


def relead(a: dict, items: list[dict], snapshot: str, weeks_by_code: dict[str, int]) -> None:
    """Set each item's lead time and work its figures after the change out again (C10 passes)."""
    entries = {}
    for code, w in weeks_by_code.items():
        entries.setdefault(w, []).append(code)
    old = {c: e for e in a["lead_time_changes"] for c in e["items"]}
    a["lead_time_changes"] = [{"items": codes, "lead_time_weeks": w,
                               "reason": old.get(codes[0], {}).get("reason", ""),
                               "findings": old.get(codes[0], {}).get("findings", [])} for w, codes in entries.items()]
    final = {str(i["item_code"]): i for i in q.figures_after_changes(items, a, snapshot)}
    a["figures_after_changes"] = [{"item_code": c, "new_lead_time_weeks": final[c]["lead_time_weeks"],
                                   "new_risk_level": final[c]["risk_level"],
                                   "new_gap_weeks": final[c]["expected_stockout_weeks"],
                                   "new_latest_po_date": final[c]["recommended_latest_po_date"]}
                                  for e in a["lead_time_changes"] for c in e["items"]]


# ----------------------------------------------------------------------------- the faults

def seeds(B: dict) -> list[dict]:
    out = []

    def add(base, check, description, fn):
        b = copy.deepcopy(B[base])
        edits = fn(b)
        out.append({"base": b["base"], "target_check": check, "expected": "fail",
                    "description": description, "edits": edits, **{k: b[k] for k in
                    ("snapshot_date", "supplier", "items", "history_docs", "analysis", "approval_input", "approval")}})

    # J1a: an incident in no document
    for base, fid in (("eval_01", 1), ("eval_14", 1), ("H3", 1)):
        add(base, "J1a", "an invented incident added to a document finding",
            lambda b, fid=fid: [replace_in(finding(b["analysis"], fid), "text", finding(b["analysis"], fid)["text"],
                                           finding(b["analysis"], fid)["text"].rstrip(".") +
                                           ". It also halted production for two weeks in February 2026 after a fire at its main plant.")])
    # J1b: a document made stronger or weaker
    add("eval_14", "J1b", "a demand surge reported as a slight rise",
        lambda b: [replace_in(finding(b["analysis"], 2), "text", "experienced a surge in demand", "experienced a slight rise in demand")])
    add("nordic", "J1b", "a 60% demand rise reported as 15%",
        lambda b: [replace_in(finding(b["analysis"], 2), "text", "about 60%", "about 15%")])
    add("eval_11", "J1b", "significant delays reported as minor",
        lambda b: [replace_in(finding(b["analysis"], 1), "text", "significant delivery delays", "minor delivery delays")])
    # J1c: a supplier-wide document presented as one item's
    def misattribute(b, fid, code):
        f = finding(b["analysis"], fid)
        name = next(i["item_name"] for i in b["items"] if str(i["item_code"]) == code)
        before = dict(f)
        f["items"] = [code]
        f["text"] = f"{name}: " + f["text"][0].lower() + f["text"][1:]
        return [{"field": f"finding {fid}", "before": f"items [] · {before['text']}", "after": f"items [{code}] · {f['text']}"}]
    add("eval_08", "J1c", "a supplier-wide quality note presented as TitaniumValve's", lambda b: misattribute(b, 2, "708001"))
    add("nordic", "J1c", "the supplier-wide customs hold presented as AnchorPlug's", lambda b: misattribute(b, 1, "410003"))
    add("eval_12", "J1c", "a supplier-wide logistics problem presented as one item's",
        lambda b: misattribute(b, next(f["id"] for f in b["analysis"]["findings"] if not f["items"] and not f["basis"].get("stock")),
                               str(b["items"][-1]["item_code"])))
    # J2a / J2b: the questions on a problem removed
    def drop_questions(b, supplier_wide):
        a = b["analysis"]
        target = {f["id"] for f in a["findings"] if not f["basis"].get("stock") and (not f["items"]) == supplier_wide}
        gone = [x["question"] for x in a["critical_questions"] if set(x.get("findings") or []) & target]
        if not gone:
            raise SystemExit(f"{b['base']}: no question to remove")
        a["critical_questions"] = [x for x in a["critical_questions"] if not set(x.get("findings") or []) & target]
        return [{"field": "critical_questions", "before": " / ".join(gone), "after": "(removed)"}]
    for base in ("eval_11", "eval_12", "eval_13"):
        add(base, "J2a", "every question on the supplier's problem removed", lambda b: drop_questions(b, True))
    for base in ("eval_14", "eval_15", "eval_16"):
        add(base, "J2b", "every question on the item's problem removed", lambda b: drop_questions(b, False))
    # J2c: a question presuming a problem the data does not show
    def presume(b):
        a = b["analysis"]
        f = next(f for f in a["findings"] if not f["items"] and not f["basis"].get("stock"))
        item = b["items"][0]["item_name"]
        qn = {"question": f"Have the repeated late deliveries of {item} over the past year been resolved, and will they affect this order?",
              "ask": "supplier", "items": [str(b["items"][0]["item_code"])], "findings": [f["id"]]}
        a["critical_questions"].append(qn)
        return [{"field": "critical_questions", "before": "(none)", "after": qn["question"]}]
    for base in ("eval_01", "eval_03", "eval_05"):
        add(base, "J2c", "a question presuming repeated late deliveries", presume)
    # J6a: the lead time changed the wrong way (shorter, for a reason to make it longer)
    def shorter(b, weeks):
        a = b["analysis"]
        before = [(e["items"], e["lead_time_weeks"]) for e in a["lead_time_changes"]]
        if a.get("supplier_lead_time"):
            a["supplier_lead_time"]["lead_time_weeks"] = weeks
        relead(a, b["items"], b["snapshot_date"], {c: weeks for e in a["lead_time_changes"] for c in e["items"]})
        return [{"field": "lead_time_changes", "before": str(before), "after": f"every item {weeks} weeks (reasons unchanged)"}]
    add("kestrel", "J6a", "lead times cut to 14 weeks for a notice of four weeks longer", lambda b: shorter(b, 14))
    add("H2", "J6a", "lead time cut to 15 weeks for a lead time that increased slightly", lambda b: shorter(b, 15))
    # J6b: the size out of keeping with the documents
    def resize(b, weeks):
        a = b["analysis"]
        before = [(e["items"], e["lead_time_weeks"]) for e in a["lead_time_changes"]]
        if a.get("supplier_lead_time"):
            a["supplier_lead_time"]["lead_time_weeks"] = weeks
        relead(a, b["items"], b["snapshot_date"], {c: weeks for e in a["lead_time_changes"] for c in e["items"]})
        return [{"field": "lead_time_changes", "before": str(before), "after": f"every item {weeks} weeks (reasons unchanged)"}]
    add("kestrel", "J6b", "30 weeks for a notice of four weeks longer (22)", lambda b: resize(b, 30))
    add("eval_13", "J6b", "34 weeks for an undocumented strike delay, reason still a 4-week allowance", lambda b: resize(b, 34))
    add("H2", "J6b", "26 weeks for a lead time that increased slightly", lambda b: resize(b, 26))
    # J6c: one item left at 18 weeks with no reason the findings support
    def leave_out(b, code):
        a = b["analysis"]
        weeks = {c: e["lead_time_weeks"] for e in a["lead_time_changes"] for c in e["items"]}
        weeks[code] = 18
        relead(a, b["items"], b["snapshot_date"], weeks)
        e = next(e for e in a["lead_time_changes"] if code in e["items"])
        e["reason"] = "Kept at the standard lead time."
        return [{"field": "lead_time_changes", "before": f"{code} at the supplier-wide lead time",
                 "after": f"{code} kept at 18 weeks: 'Kept at the standard lead time.'"}]
    add("kestrel", "J6c", "OilSeal left at 18 weeks under a notice for all products", lambda b: leave_out(b, "520005"))
    add("eval_12", "J6c", "one item left at 18 weeks under a supplier-wide delay",
        lambda b: leave_out(b, str(b["items"][-1]["item_code"])))
    # J7a: a stock figure in the text that is not the item's figure after the changes
    add("kestrel", "J7a", "the summary keeps the 18-week gap (6) for the 22-week one (10)",
        lambda b: [replace_in(b["analysis"], "summary", "a longest gap of 10 weeks", "a longest gap of 6 weeks")])
    add("eval_14", "J7a", "the summary's 15-week gap written as twelve weeks, in words",
        lambda b: [replace_in(b["analysis"], "summary", "a 15-week gap", "a twelve-week gap")])
    add("eval_08", "J7a", "a stock finding keeps the 18-week gap (6) for the 20-week one (8)",
        lambda b: [replace_in(finding(b["analysis"], 4), "text", "8-week gap", "6-week gap")])
    add("nordic", "J7a", "an action's 9-week gap written as a shortfall of 13 weeks",
        lambda b: [replace_in(b["analysis"]["recommended_actions"][0], "action", "the 9-week gap", "a shortfall of 13 weeks")])
    # J8: a supplier question revealing the stock position
    def reveal(b):
        a = b["analysis"]
        x = next(x for x in a["critical_questions"] if x["ask"] == "supplier")
        it = next((i for i in b["items"] if i["risk_level"] == "Critical"), b["items"][0])
        old = x["question"]
        x["question"] = old + f" Our available stock of {it['item_name']} covers only {it['wks_to_oos']} weeks."
        return [{"field": "supplier question", "before": old, "after": x["question"]}]
    for base in ("eval_02", "eval_04", "eval_10"):
        add(base, "J8", "a supplier question giving the weeks of stock left", reveal)
    # J9: an action its cited finding gives no reason for
    def unfounded(b):
        a = b["analysis"]
        f = next(f for f in a["findings"] if not f["items"] and not f["basis"].get("stock"))
        x = {"action": f"Move this order from {b['supplier']} to an alternative supplier.", "items": [], "findings": [f["id"]]}
        a["recommended_actions"].append(x)
        return [{"field": "recommended_actions", "before": "(none)", "after": f"{x['action']} citing F{f['id']}: {f['text'][:80]}"}]
    for base in ("eval_01", "eval_03"):
        add(base, "J9", "an action to change supplier, citing the supplier's good record", unfounded)
    # J10: a document with a problem left unused
    def unuse(b):
        a = b["analysis"]
        target = {f["id"] for f in a["findings"] if f["items"] and not f["basis"].get("stock") and f["basis"].get("documents")}
        gone = [x.get("question") or x.get("action") for k in ("critical_questions", "recommended_actions")
                for x in a[k] if set(x.get("findings") or []) & target]
        for k in ("critical_questions", "recommended_actions"):
            a[k] = [x for x in a[k] if not set(x.get("findings") or []) & target]
        return [{"field": "questions and actions", "before": " / ".join(gone), "after": "(removed)"}]
    for base in ("eval_15", "eval_16"):
        add(base, "J10", "every question and action on the item's problem removed", unuse)
    # J11: a judgement written into a finding
    for base in ("eval_09", "eval_17", "H4"):
        add(base, "J11", "a judgement added to a document finding",
            lambda b: [replace_in(b["analysis"]["findings"][0], "text", b["analysis"]["findings"][0]["text"],
                                  b["analysis"]["findings"][0]["text"].rstrip(".") + ", which makes this supplier a risky choice for this order.")])
    # P1a / P1b / P1c on the purchase request's Justification
    for base in ("eval_01", "eval_05", "eval_17"):
        add(base, "P1a", "an unsupported risk added to the Justification",
            lambda b: [replace_in(b["approval"], "justification", b["approval"]["justification"],
                                  b["approval"]["justification"].rstrip() + " The supplier's chronic capacity problems make further delays likely.")])
    add("eval_08", "P1b", "a figure judged: an alarming 8-week gap",
        lambda b: [replace_in(b["approval"], "justification", "an 8-week gap", "an alarming 8-week gap")])
    add("eval_14", "P1b", "a figure judged: a dramatic 15-week gap",
        lambda b: [replace_in(b["approval"], "justification", "a 15-week gap", "a dramatic 15-week gap")])
    add("eval_01", "P1c", "steady demand written as rising demand",
        lambda b: [replace_in(b["approval"], "justification", "The steady demand", "The rising demand")])
    add("eval_14", "P1c", "a recent demand surge written as a long-standing rise",
        lambda b: [replace_in(b["approval"], "justification", "a recent surge in demand", "a long-standing rise in demand")])
    add("eval_11", "P1c", "a provisional strike allowance written as a supplier-confirmed delay",
        lambda b: [replace_in(b["approval"], "justification", "supplier-wide strike contingency", "supplier-confirmed strike delay")])
    return out


# ----------------------------------------------------------------------------- real cases

def real_cases() -> list[dict]:
    def from_run(pattern, case, attempt, check, expected, why):
        d = sorted(glob.glob(str(ROOT / "dev" / "runs" / pattern)))
        r = None
        for p in d:
            rr = json.loads(Path(p, "record.json").read_text(encoding="utf-8"))
            if rr["case"] == case:
                r, d = rr, p
                break
        c = json.loads((ROOT / "dev" / "cases" / f"{case}.json").read_text(encoding="utf-8"))
        g = group_by_supplier_and_recommend(parse_csv_rows(c["csv"], f"{c['name']}.csv"))[0]
        a = next(x for x in r["analysis_attempts"] if x["attempt"] == attempt)["output"]
        return {"base": f"real:{Path(d).name}#attempt{attempt}", "target_check": check, "expected": expected,
                "description": why, "edits": [], "snapshot_date": g["snapshot_date"], "supplier": g["supplier"],
                "items": g["items"], "history_docs": c["history"], "analysis": a, "approval_input": None, "approval": None}
    return [
        from_run("*round-2-run-2-gpt-4o", "nordic_fasteners", 1, "J6a", "fail",
                 "GPT-4o raised HexBoltM12's lead time to 22 weeks for a customs hold the supplier says it has fixed; the judge passed J6a"),
        from_run("*round-2-run-2-gpt-6-luna", "kestrel_components", 2, "J6b", "fail",
                 "Luna's rewrite cut GasketKit's 50% rise to 5,400 ('the uplift above the 4,800 units of available stock'); the judge passed it"),
        from_run("*round-2-run-2-gpt-6-luna", "kestrel_components", 1, "J6b", "pass",
                 "Luna's 50% rise to 7,800 for a 50% demand rise; the judge failed it, saying the uplift included stock held"),
        from_run("*round-2-run-1-claude-sonnet-5-5", "kestrel_components", 1, "J7a", "pass",
                 "Sonnet wrote that ValveSeat's latest order date is the snapshot date, which it is (2025-04-05); the judge failed it"),
        from_run("2026-10-24_011306_*", "nordic_fasteners", 3, "J7a", "fail",
                 "GPT-4o kept 'WasherSteel runs out 2 weeks before a new order can arrive' after withdrawing the 22-week lead time (gap 0)"),
    ]


def main() -> None:
    B = load_bases()
    rows, dropped = [], []
    for n, s in enumerate(seeds(B) + real_cases(), 1):
        code = q.code_checks(s["items"], s["analysis"], s["history_docs"], s["snapshot_date"])
        failing = [k for k in BLOCKING_CODE if code[k]["pass"] is False]
        s["id"] = f"v5-{n:03d}"
        s["code_checks_failed"] = failing
        if s["target_check"].startswith("J"):
            s["applicable"] = q.applicable_judge_checks(s["items"], s["history_docs"], analysis=s["analysis"])
            if s["target_check"] not in s["applicable"]:
                failing = failing + [f"{s['target_check']} not applicable"]
        if failing and not s["base"].startswith("real:"):
            dropped.append((s["id"], s["base"], s["target_check"], failing))
            continue
        rows.append(s)
    with open(OUT, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    by = {}
    for s in rows:
        by.setdefault(s["target_check"], []).append(s["expected"])
    print(f"{len(rows)} kept -> {OUT}")
    for k in sorted(by):
        print(f"  {k}: {len(by[k])} ({', '.join(by[k])})")
    for d in dropped:
        print("  dropped:", d)


if __name__ == "__main__":
    main()
