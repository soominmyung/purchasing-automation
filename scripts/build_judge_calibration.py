"""
Build the seeded-error half of the judge calibration set (docs/judge_validation.md).

Each entry is one of GPT-4o's analyses (eval_results/eval_v2_gpt4o_outputs.jsonl) with
exactly one deliberate fault aimed at one check, so the correct verdict for that check is
known to be fail. Every change is written out below as an exact edit, so it can be read and
reviewed; an edit whose original text is not found exactly once stops the build.

Seeds are placed only on outputs that pass the targeted check before the edit.

Run: python scripts/build_judge_calibration.py
Out: eval_data/judge_calibration.jsonl
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "eval_results" / "eval_v2_gpt4o_outputs.jsonl"
SCENARIOS = [ROOT / "eval_data" / "eval_scenarios.jsonl", ROOT / "eval_data" / "holdout_scenarios.jsonl"]
OUT_FILE = ROOT / "eval_data" / "judge_calibration.jsonl"

NO_HISTORY = "No historical supplier or item data available; analysis based on current stock/deadlines only."

# Edit operations:
#   ("md", old, new)              replace text in purchasing_report_markdown
#   ("note", item_code, old, new) replace text in that item's timeline notes
#   ("set_note", item_code, text) replace that item's notes entirely
#   ("drop_q", [indices])         remove critical questions by position
#   ("add_q", question_dict)      append a critical question
SEEDS = [
    # --- J1: invented history -------------------------------------------------------
    ("J1", "invented", "eval_01", "Late shipments that are in neither history added to the summary",
     [("md", "Supplier history indicates consistent timely deliveries and stable pricing.",
       "Supplier history indicates consistent timely deliveries and stable pricing. Two shipments in January 2026 arrived about three weeks late.")]),
    ("J1", "invented", "eval_02", "A product recall that is in neither history added to SwitchButtonX's notes",
     [("note", "702002", "History reviewed: no notable supplier or item issues.",
       "History reviewed: no notable supplier or item issues. A batch of SwitchButtonX was recalled in November 2025 for faulty contacts.")]),
    ("J1", "invented", "eval_10", "CopperWiringBundle's regular consumption replaced with a doubling of demand the history does not mention",
     [("md", "- CopperWiringBundle maintains regular consumption with no notable item-specific issues.",
       "- CopperWiringBundle demand has roughly doubled since January.")]),
    ("J1", "invented", "eval_14", "A price increase that is in neither history added to the supplier summary",
     [("md", "TechMarvelSupplies has been reliable with consistent deliveries and competitive pricing.",
       "TechMarvelSupplies has been reliable with consistent deliveries and competitive pricing. It has announced a 12% price increase effective April 2026.")]),
    ("J1", "invented", "H4", "A past stockout that is in neither history added to HandleGrip's notes",
     [("note", "500041", "due to an upcoming product accessory launch.",
       "due to an upcoming product accessory launch. HandleGrip ran out of stock for two weeks in December 2025.")]),

    # --- J1: distorted history ------------------------------------------------------
    ("J1", "distorted", "eval_11", "An unresolved strike reported as resolved",
     [("md", "  - Significant delivery delays due to a strike at the main warehouse, unresolved.",
       "  - Significant delivery delays due to a strike at the main warehouse, since resolved.")]),
    ("J1", "distorted", "eval_14", "A demand surge reported as a slight increase",
     [("md", "  - **NanoResistorTypeA**: Recent demand surge due to new product application.",
       "  - **NanoResistorTypeA**: Slight recent increase in demand due to new product application.")]),
    ("J1", "distorted", "H2", "A slight lead-time increase reported as a sharp one",
     [("note", "300021", "PartW's lead time has increased slightly", "PartW's lead time has increased sharply")]),

    # --- J1b: false "no history" sentence -------------------------------------------
    ("J1b", "false_no_history", "eval_05", "AluminumSheetY's notes replaced with the fixed no-history sentence",
     [("set_note", "705002", NO_HISTORY)]),
    ("J1b", "false_no_history", "eval_09", "MicroProcessorX's notes replaced with the fixed no-history sentence",
     [("set_note", "709002", NO_HISTORY)]),
    ("J1b", "false_no_history", "eval_13", "UltraLEDPanel's notes replaced with the fixed no-history sentence",
     [("set_note", "713002", NO_HISTORY)]),
    ("J1b", "false_no_history", "H1", "GadgetY's notes replaced with the fixed no-history sentence",
     [("set_note", "200011", NO_HISTORY)]),

    # --- J2a: supplier problem not asked about --------------------------------------
    ("J2a", "supplier_question_removed", "eval_08", "Both questions about the delays and quality problems removed (no questions left)",
     [("drop_q", [0, 1])]),
    ("J2a", "supplier_question_removed", "eval_11", "The question about the warehouse strike removed (no questions left)",
     [("drop_q", [0])]),
    ("J2a", "supplier_question_removed", "H2", "The question about the price spike removed; the ComponentZ question remains",
     [("drop_q", [0])]),
    ("J2a", "supplier_question_removed", "H5", "The question about the operational halt removed; the EcoBag question remains",
     [("drop_q", [0])]),

    # --- J2b: item problem not asked about ------------------------------------------
    ("J2b", "item_question_removed", "eval_14", "The question about NanoResistorTypeA's demand surge removed (no questions left)",
     [("drop_q", [0])]),
    ("J2b", "item_question_removed", "eval_16", "The question about TitaniumAlloyWheel's demand surge removed (no questions left)",
     [("drop_q", [0])]),
    ("J2b", "item_question_removed", "H1", "The question about GadgetY's demand volatility removed; the supplier question remains",
     [("drop_q", [1])]),
    ("J2b", "item_question_removed", "H3", "Both item questions removed; the supplier question remains",
     [("drop_q", [1, 2])]),

    # --- J2c: question presumes a problem the data does not show --------------------
    ("J2c", "presumptive_question", "eval_01", "A question presuming repeated delivery delays added",
     [("add_q", {"target": "general", "question": "Have the repeated delivery delays from GreenLineSupplies been resolved?",
                 "reason": "supplier_history"})]),
    ("J2c", "presumptive_question", "eval_03", "A question presuming volatile demand for FastenerX added",
     [("add_q", {"target": "703001", "question": "What is driving the recent volatility in FastenerX demand?",
                 "reason": "item_history"})]),
    ("J2c", "presumptive_question", "eval_05", "A question presuming a quality defect in SteelRodX added",
     [("add_q", {"target": "705001", "question": "Has the quality defect found in the last SteelRodX shipment been corrected?",
                 "reason": "item_history"})]),
    ("J2c", "presumptive_question", "eval_06", "A question presuming a production backlog added (no history given)",
     [("add_q", {"target": "general", "question": "When will IndusFabricators clear its current production backlog?",
                 "reason": "supplier_history"})]),

    # --- J3: absence of history not stated ------------------------------------------
    ("J3", "no_history_statement_removed", "eval_06", "Every statement that no history is available removed from the summary and notes",
     [("md", "- **History:**\n  - " + NO_HISTORY + "\n", ""),
      ("note", "706001", " " + NO_HISTORY, ""),
      ("set_note", "706002", "")]),
    ("J3", "no_history_statement_removed", "eval_07",
     "Every statement that no history is available removed from the summary and notes, and the question that mentioned it",
     [("md", "\n\n" + NO_HISTORY + "\n", "\n"),
      ("set_note", "707001", ""), ("set_note", "707002", ""), ("drop_q", [0])]),

    # --- J5: Critical item missing from the summary ---------------------------------
    ("J5", "critical_omitted", "eval_08", "The Critical item TitaniumValve's stockout line removed from the summary",
     [("md", "- **Critical Item:** TitaniumValve is expected to have a stockout of about 6 weeks. Consider expedited shipping or alternative sourcing.\n", "")]),
    ("J5", "critical_omitted", "eval_10", "CopperWiringBundle, one of two Critical items, removed from the summary",
     [("md", "- **CopperWiringBundle** is Critical with an expected stockout of 12 weeks. Consider expedited shipping or alternative sourcing.\n", "")]),
    ("J5", "critical_omitted", "eval_12", "HighVoltageCapacitor, one of two Critical items, removed from the summary",
     [("md", "  - HighVoltageCapacitor: Expected stockout of 16 weeks. Consider expedited shipping or alternative sourcing.  \n", "")]),
    ("J5", "critical_omitted", "H1", "The Critical item WidgetX's stockout line removed from the summary",
     [("md", "  - **WidgetX**: Expected stockout of 14 weeks. Recommend expedited shipping or alternative sourcing.\n", "")]),

    # --- J5: wrong stockout figure --------------------------------------------------
    ("J5", "critical_wrong_figure", "eval_04", "ConnectorABC's expected stockout changed from 10 to 8 weeks (its weeks to stockout)",
     [("md", "ConnectorABC: Expected stockout of 10 weeks.", "ConnectorABC: Expected stockout of 8 weeks.")]),
    ("J5", "critical_wrong_figure", "eval_15", "GrapheneCoating's expected stockout changed from 16 to 2 weeks (its weeks to stockout)",
     [("md", "**GrapheneCoating** is Critical with an expected stockout of 16 weeks.", "**GrapheneCoating** is Critical with an expected stockout of 2 weeks.")]),
    ("J5", "critical_wrong_figure", "H5", "EcoBag's expected stockout changed from 15 to 3 weeks (its weeks to stockout)",
     [("md", "**EcoBag**: Expected stockout of 15 weeks.", "**EcoBag**: Expected stockout of 3 weeks.")]),
]



def _replace_once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{where}: expected the original text once, found it {n} times:\n  {old!r}")
    return text.replace(old, new)


def _entry(output: dict, item_code: str, where: str) -> dict:
    hits = [e for e in output["replenishment_timeline"] if str(e.get("item_code")) == item_code]
    if len(hits) != 1:
        raise SystemExit(f"{where}: no single timeline entry for {item_code}")
    return hits[0]


def apply(output: dict, edits: list, where: str) -> dict:
    out = copy.deepcopy(output)
    for op in edits:
        kind = op[0]
        if kind == "md":
            out["purchasing_report_markdown"] = _replace_once(out["purchasing_report_markdown"], op[1], op[2], where)
        elif kind == "note":
            e = _entry(out, op[1], where)
            e["notes"] = _replace_once(str(e.get("notes") or ""), op[2], op[3], where)
        elif kind == "set_note":
            _entry(out, op[1], where)["notes"] = op[2]
        elif kind == "drop_q":
            qs = out["critical_questions"]
            if max(op[1]) >= len(qs):
                raise SystemExit(f"{where}: question index out of range")
            out["critical_questions"] = [q for i, q in enumerate(qs) if i not in op[1]]
        elif kind == "add_q":
            out["critical_questions"] = list(out["critical_questions"]) + [op[1]]
        else:
            raise SystemExit(f"{where}: unknown edit {kind}")
    if out == output:
        raise SystemExit(f"{where}: edits changed nothing")
    return out


def main():
    source_text = SOURCE.read_bytes()
    source_sha = hashlib.sha256(source_text).hexdigest()[:12]
    originals = {r["scenario_id"]: r for r in map(json.loads, source_text.decode("utf-8").splitlines()) if r}
    scenarios = {}
    for path in SCENARIOS:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                s = json.loads(line)
                scenarios[s["scenario_id"]] = s

    rows = []
    for n, (check, fault, sid, description, edits) in enumerate(SEEDS, 1):
        cid = f"seed_{n:02d}"
        base = originals[sid]
        rows.append({
            "id": cid,
            "kind": "seeded",
            "scenario_id": sid,
            "category": scenarios[sid]["category"],
            "labels": scenarios[sid]["labels"],
            "target_check": check,
            "fault": fault,
            "expected": {check: "fail"},
            "description": description,
            "edits": [list(e) for e in edits],
            "source": {"file": SOURCE.name, "sha256_12": source_sha, "model": base["model"]},
            "input": base["input"],
            "supplier_history": base["supplier_history"],
            "item_history": base["item_history"],
            "output": apply(base["output"], edits, f"{cid} ({sid} {check})"),
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        key = f"{r['target_check']} {r['fault']}"
        counts[key] = counts.get(key, 0) + 1
    for k, v in counts.items():
        print(f"  {k:<40} {v}")
    print(f"\n{len(rows)} seeded outputs -> {OUT_FILE}")


if __name__ == "__main__":
    main()
