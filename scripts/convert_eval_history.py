"""
Turn each evaluation scenario's two free-text histories into history documents, the form
the pipeline now gives the model (one list of documents, each naming the supplier and items
it is about; see services/vector_store.py).

Default conversion, text unchanged:
  supplier_history -> one supplier-wide document   {supplier: <supplier>, items: []}
  item_history     -> one document about the items it names, whoever supplies them
                      {supplier: "", items: [named items]}; when it names none, all items
  "No ... history found." -> no document

Exceptions, set out below:
  SPLIT  item histories whose sentences are about different items, and where a sentence
         describes the supplier's delivery of an item (a supply-side event), split so that
         sentence becomes a supplier x item document
  EXTRA  documents added to test a rule that the two-text form could not express

eval_08 is deliberately left as converted by default: its supplier-wide document reports
quality discrepancies without naming an item, which must not be attributed to any item.

Run after generate_eval_scenarios.py, build_holdout_eval.py and set_eval_risk_mix.py:
     python scripts/convert_eval_history.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.quality_checks import has_history

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "eval_data" / "eval_scenarios.jsonl", ROOT / "eval_data" / "holdout_scenarios.jsonl"]

# scenario -> documents replacing its item history: (supplier or "", items, exact sentence(s))
SPLIT = {
    "H2": [
        ("", ["ComponentZ"], "ComponentZ historically substitutes rarely used items that experience sudden maintenance demands."),
        ("EuroSupply", ["PartW"], "PartW's lead time has increased slightly because of new manufacturing processes."),
    ],
    "H3": [
        ("", ["MachineAddon"], "MachineAddon often sees sudden spikes due to large-scale industrial uptakes."),
        ("MegaManufacturers", ["DeviceConnector"], "DeviceConnector is utilized primarily as a substitute in high-demand scenarios, despite having erratic supply timelines."),
    ],
}

# scenario -> documents added
EXTRA = {
    # About an item of this supplier that is not in the analysis: may be mentioned as context
    # about the supplier, never in another item's notes.
    "eval_12": [
        ("TechWareSolutions", ["PowerModuleZ"],
         "A batch of PowerModuleZ delivered by TechWareSolutions in January 2026 failed incoming "
         "inspection because of loose terminals and was returned."),
    ],
}


def doc(supplier: str, items: list[str], text: str) -> dict:
    return {"supplier": supplier, "items": items, "item_codes": [], "date": "", "text": text}


def convert(r: dict) -> list[dict]:
    names = [it["ItemName"] for it in r["input_data"] if it.get("ItemName")]
    docs = []
    if has_history(r.get("supplier_history")):
        docs.append(doc(r["supplier"], [], r["supplier_history"]))
    ih = r.get("item_history")
    if r["scenario_id"] in SPLIT:
        parts = SPLIT[r["scenario_id"]]
        if " ".join(t for _, _, t in parts) != ih:
            raise SystemExit(f"{r['scenario_id']}: the split sentences do not reproduce the item history exactly")
        docs += [doc(s, items, t) for s, items, t in parts]
    elif has_history(ih):
        named = [n for n in names if n.lower() in ih.lower()]
        docs.append(doc("", named or names, ih))
    docs += [doc(s, items, t) for s, items, t in EXTRA.get(r["scenario_id"], [])]
    return docs


def main():
    for path in FILES:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            if "history" in r and "supplier_history" not in r:
                continue  # already converted
            r["history"] = convert(r)
            r.pop("supplier_history", None)
            r.pop("item_history", None)
            scope = lambda d: (f"{d['supplier']} x {', '.join(d['items'])}" if d["supplier"] and d["items"] else
                               f"{d['supplier']} (all items)" if d["supplier"] else f"{', '.join(d['items'])} (any supplier)")
            print(f"  {r['scenario_id']:<8} " + (" | ".join(scope(d) for d in r["history"]) or "(no history)"))
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
