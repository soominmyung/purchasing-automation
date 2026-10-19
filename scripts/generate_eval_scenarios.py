"""
Generate a held-out evaluation set for the analysis agent.

The training scenarios (training_data/) were generated with the adversarial mix left
to GPT-4o, and in practice every one of them carried a supplier or item incident. That
makes it impossible to test the opposite behaviour: not raising a question when there
is nothing to ask about. This script fixes the mix instead of leaving it to chance.

Each scenario is labelled with facts about the scenario, not with the expected answer:
  supplier_issue / item_issue : "material" | "none"  (a supplier history with no company
                                named counts as material: the tool only returns that supplier's documents)
  data_issue                  : "none" | "missing_values"
  history_present             : bool
The rubric decides what a good analysis does with those facts.

RiskLevel is overwritten with the value the production pipeline derives from WksToOOS
(services/item_grouping.risk_level_from_weeks), so inputs match what the agent sees in use.

Supplier and item names are kept disjoint from the training set so no scenario
overlaps with anything the fine-tuned model has seen.

Run: python scripts/generate_eval_scenarios.py   (then scripts/set_eval_risk_mix.py)
     python scripts/generate_eval_scenarios.py --replace eval_21,eval_25   (redo some, keep the rest)
Out: eval_data/eval_scenarios.jsonl
"""
import ast
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import settings
from services.item_grouping import risk_level_from_weeks

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "training_data" / "teacher_dataset_20260302.jsonl"
OUT_FILE = ROOT / "eval_data" / "eval_scenarios.jsonl"

# The exact strings the history tools return when nothing is on file.
NO_SUPPLIER_HISTORY = "No supplier history found."
NO_ITEM_HISTORY = "No item history found."

CONSISTENT = "Stock and weeks-to-out-of-stock must be mutually consistent."

# (category, count, labels, instruction)
PLAN = [
    ("clean", 5,
     dict(supplier_issue="none", item_issue="none", data_issue="none", history_present=True),
     "Both histories are genuinely unremarkable: steady deliveries, stable prices, normal "
     "demand. Write them as ordinary record-keeping notes. Do NOT mention any delay, price "
     "rise, quality problem, shortage, spike, strike or risk, not even a minor or resolved one. "
     + CONSISTENT),
    ("no_history", 2,
     dict(supplier_issue="none", item_issue="none", data_issue="none", history_present=False),
     "Only the inventory rows matter; the histories will be replaced. " + CONSISTENT),
    ("ambiguous_attribution", 3,
     dict(supplier_issue="material", item_issue="none", data_issue="none", history_present=True),
     "The supplier history describes a recent problem (for example delays or a quality issue) "
     "but NEVER names the supplier or any company; it is written without a subject so it is "
     "unclear who it refers to. The item history is unremarkable. " + CONSISTENT),
    ("clear_supplier_incident", 3,
     dict(supplier_issue="material", item_issue="none", data_issue="none", history_present=True),
     "The supplier history names this supplier explicitly and describes a recent, unresolved "
     "problem that affects ordering now (for example ongoing delivery delays, a recent price "
     "rise, a recent quality failure). The item history is unremarkable. " + CONSISTENT),
    ("clear_item_incident", 3,
     dict(supplier_issue="none", item_issue="material", data_issue="none", history_present=True),
     "The item history names one of the items explicitly and describes a recent issue that "
     "affects ordering now (for example a recent demand surge, a lengthened lead time, a "
     "recent stockout). The supplier history is unremarkable. " + CONSISTENT),
    ("high_stock_short_cover", 2,
     dict(supplier_issue="none", item_issue="none", data_issue="none", history_present=True),
     "Exactly one item has numbers that contradict each other: a high AvailableStock (300 or "
     "more) together with WksToOOS of 1 or 2, and RiskLevel Low. Other items are consistent. "
     "Both histories are unremarkable and give no explanation for the contradiction."),
    ("missing_values", 2,
     dict(supplier_issue="none", item_issue="none", data_issue="missing_values", history_present=True),
     "Exactly one item has AvailableStock and WksToOOS given as the string N/A. Other items are "
     "complete and consistent. Both histories are unremarkable."),
]

SYSTEM = """You generate one realistic purchasing scenario for testing a purchasing-analysis AI.
Inventory schema per item: ItemCode (6-digit string), ItemName, SupplierName, RiskLevel
(High|Medium|Low), AvailableStock (integer), WksToOOS (integer weeks until out of stock).
All items in a scenario come from the same supplier.

Return JSON: {"supplier": str, "input_data": [items], "supplier_history": str, "item_history": str}
- 2 or 3 items.
- supplier_history: 1-3 sentences about the supplier. item_history: 1-3 sentences about the items.
- Do not reuse any of these supplier names: SUPPLIERS
- Do not reuse any of these item names: ITEMS
- Item codes must start with 7.
"""

SUPPLIER_INCIDENT = re.compile(
    r"\b(delay|delayed|late|recall|defect|strike|shortage|price (rise|increase|spike)|failure|halt)", re.I)
ITEM_INCIDENT = re.compile(
    r"\b(spike|surge|stockout|stock-out|shortage|volatil|fluctuat|lead time (increase|extend|length))", re.I)
# In contradiction / missing-value scenarios the history must not hint at the data
# problem, or at a demand change that would turn it into an item incident.
DATA_HINT = re.compile(
    r"\b(error|discrepan|mismatch|numbers|figures|inventory level|stock level|report|"
    r"popular|growing|increas|rising|demand (rise|grow))", re.I)


def lit(x):
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return ast.literal_eval(x)


def existing_names():
    sup, items = set(), set()
    for line in open(TRAIN_FILE, encoding="utf-8"):
        inv = lit(json.loads(line)["input"])["inventory"]
        sup |= {it["SupplierName"] for it in inv if it.get("SupplierName")}
        items |= {it["ItemName"] for it in inv if it.get("ItemName")}
    return sup, items


def check(cat, s, banned_sup, banned_items):
    """Return a list of problems; empty means the scenario matches its category."""
    p = []
    inv = s.get("input_data") or []
    sup = s.get("supplier", "")
    if not 2 <= len(inv) <= 3:
        p.append("item count")
    if sup in banned_sup:
        p.append("supplier name reused")
    if any(it.get("ItemName") in banned_items for it in inv):
        p.append("item name reused")
    if any(it.get("SupplierName") != sup for it in inv):
        p.append("mixed suppliers")
    sh, ih = s.get("supplier_history", ""), s.get("item_history", "")

    if cat == "ambiguous_attribution" and sup.lower() in sh.lower():
        p.append("ambiguous history names the supplier")
    if cat == "clear_supplier_incident" and sup.lower() not in sh.lower():
        p.append("supplier incident does not name the supplier")
    if cat == "clear_item_incident" and not any(it["ItemName"].lower() in ih.lower() for it in inv):
        p.append("item incident does not name an item")
    if cat == "high_stock_short_cover":
        hit = [it for it in inv if isinstance(it.get("AvailableStock"), (int, float))
               and it["AvailableStock"] >= 300 and it.get("WksToOOS") in (1, 2)]
        if len(hit) != 1:
            p.append("contradiction pattern")
    if cat == "missing_values":
        if sum(1 for it in inv if it.get("AvailableStock") == "N/A" and it.get("WksToOOS") == "N/A") != 1:
            p.append("missing-value pattern")

    # categories whose supplier / item history must stay clean
    if cat in ("clean", "high_stock_short_cover", "missing_values", "clear_item_incident") \
            and SUPPLIER_INCIDENT.search(sh):
        p.append("supplier history carries an incident")
    if cat in ("clean", "high_stock_short_cover", "missing_values", "clear_supplier_incident",
               "ambiguous_attribution") and ITEM_INCIDENT.search(ih):
        p.append("item history carries an incident")
    if cat in ("high_stock_short_cover", "missing_values") and (DATA_HINT.search(ih) or DATA_HINT.search(sh)):
        p.append("history hints at the data problem")
    return p


def renumber_codes(rows):
    """Give every item a code unique across the whole set: 7 + scenario number + position."""
    for r in rows:
        n = int(r["scenario_id"].split("_")[1])
        for i, it in enumerate(r["input_data"]):
            it["ItemCode"] = f"7{n:02d}{i + 1:03d}"


def generate_one(client, cat, instruction, banned_sup, banned_items):
    sys_prompt = SYSTEM.replace("SUPPLIERS", ", ".join(sorted(banned_sup))) \
                       .replace("ITEMS", ", ".join(sorted(banned_items)))
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": f"Scenario type: {cat}.\n{instruction}"}],
        response_format={"type": "json_object"},
        temperature=0.9,
    )
    return json.loads(resp.choices[0].message.content)


def make_valid(client, cat, instruction, banned_sup, banned_items, tries=5):
    for _ in range(tries):
        s = generate_one(client, cat, instruction, banned_sup, banned_items)
        if cat == "no_history":
            s["supplier_history"], s["item_history"] = NO_SUPPLIER_HISTORY, NO_ITEM_HISTORY
        problems = check(cat, s, banned_sup, banned_items)
        if not problems:
            return s
        print(f"  rejected {cat}: {', '.join(problems)}")
    raise SystemExit(f"could not produce a valid '{cat}' scenario")


def apply_risk_rule(rows):
    """RiskLevel comes from WksToOOS, exactly as the production pipeline derives it."""
    for r in rows:
        for it in r["input_data"]:
            it["RiskLevel"] = risk_level_from_weeks(it.get("WksToOOS"))


def write(rows):
    renumber_codes(rows)
    apply_risk_rule(rows)
    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} scenarios -> {OUT_FILE}")


def main():
    client = OpenAI(api_key=settings.openai_api_key)
    banned_sup, banned_items = existing_names()
    spec = {cat: (labels, instruction) for cat, _, labels, instruction in PLAN}

    # --replace eval_21,eval_25 : regenerate only those scenarios, keep the rest as reviewed
    if len(sys.argv) > 2 and sys.argv[1] == "--replace":
        targets = set(sys.argv[2].split(","))
        rows = [json.loads(l) for l in open(OUT_FILE, encoding="utf-8")]
        for r in rows:
            if r["scenario_id"] not in targets:
                banned_sup.add(r["supplier"])
                banned_items |= {it["ItemName"] for it in r["input_data"] if it.get("ItemName")}
        for i, r in enumerate(rows):
            if r["scenario_id"] in targets:
                labels, instruction = spec[r["category"]]
                s = make_valid(client, r["category"], instruction, banned_sup, banned_items)
                banned_sup.add(s["supplier"])
                banned_items |= {it["ItemName"] for it in s["input_data"] if it.get("ItemName")}
                rows[i] = {"scenario_id": r["scenario_id"], "category": r["category"], "labels": labels, **s}
                print(f"  {r['scenario_id']} {r['category']:<24} {s['supplier']} (replaced)")
        write(rows)
        return

    rows, n = [], 0
    for cat, count, labels, instruction in PLAN:
        for _ in range(count):
            s = make_valid(client, cat, instruction, banned_sup, banned_items)
            n += 1
            banned_sup.add(s["supplier"])
            banned_items |= {it["ItemName"] for it in s["input_data"] if it.get("ItemName")}
            rows.append({"scenario_id": f"eval_{n:02d}", "category": cat, "labels": labels, **s})
            print(f"  eval_{n:02d} {cat:<24} {s['supplier']}")
    write(rows)


if __name__ == "__main__":
    main()
