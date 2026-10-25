"""
Training scenarios for the SFT retraining (docs/sft_plan.md, section 2), in the evaluation set's
format: one supplier, its stock rows, and history documents with headers (supplier, items, date).

The mix is fixed by type, not left to the generator. Names are kept apart from the 25
evaluation cases (and from each other); item codes are assigned here. RiskLevel is overwritten
with the pipeline's own (services/item_grouping.risk_level_from_weeks). The generalisation set
(--tie-in) is a type no training scenario has: a supplier pushing a newly launched product with
the order.

Run:
  python scripts/generate_sft_scenarios.py                 the training scenarios (resumes)
  python scripts/generate_sft_scenarios.py --tie-in        the generalisation scenarios
Out: training_data/sft_v5/scenarios.jsonl, training_data/sft_v5/generalisation_tie_in.jsonl
"""
import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402
from services.item_grouping import risk_level_from_weeks  # noqa: E402

OUT_DIR = ROOT / "training_data" / "sft_v5"
MODEL, EFFORT = "gpt-6.1-sol", "low"
PER_TYPE = 9
WORKERS = 12
SNAPSHOT = "2026-03-02"  # the evaluation set's fixed snapshot date

COMMON = (
    "The company imports food and household products from overseas suppliers, mostly in Korea, and "
    "sells them in the UK. Write history documents as a purchasing team keeps them: short notes, "
    "reports or emails of 1-4 sentences each, factual, in British English, dated within the 12 months "
    f"before {SNAPSHOT} (YYYY-MM-DD). Each document is about the supplier as a whole (items empty), "
    "or about particular items of this supplier (supplier and items set), or about items whoever "
    "supplies them (supplier empty, items set). "
    "History documents never restate "
    "the current stock, weeks of cover or a risk level: those come from the stock data, not the history."
)
LETTERS = "ABCDEFGHIJKLMNOPRSTUVWYZ"
LINES = ["frozen dumplings and ready meals", "sauces and pastes", "noodles and rice", "snacks and confectionery",
         "beverages and teas", "kitchenware and storage", "cleaning and household goods", "personal care",
         "seaweed and dried seafood", "kimchi and pickles", "baking and flour products", "health foods"]

# (type, labels, instruction)
TYPES = [
    ("clean", dict(supplier_issue="none", item_issue="none"),
     "Unremarkable history: steady deliveries, stable prices, normal demand. No delay, price rise, "
     "quality problem, shortage, spike, strike or risk, not even a resolved one."),
    ("no_history", dict(supplier_issue="none", item_issue="none"),
     "No history documents at all (history is an empty list)."),
    ("delay_unsized", dict(supplier_issue="material", item_issue="none"),
     "The supplier has delivery delays, a strike, a production halt or a backlog, described in words "
     "(e.g. 'significant delays', 'ongoing', 'unresolved', 'slight') but with NO number of weeks."),
    ("lead_time_notice", dict(supplier_issue="material", item_issue="none"),
     "The supplier has announced lead times longer by a stated number of weeks (2-6), for all its "
     "products or for some items only."),
    ("item_quality_incident", dict(supplier_issue="none", item_issue="material"),
     "A quality or safety incident with one item from this supplier (a defective batch, a "
     "contamination, a labelling error), with what was done and what is still open."),
    ("supplier_quality_incident", dict(supplier_issue="material", item_issue="none"),
     "A quality problem across the supplier's products (an audit finding, repeated complaints)."),
    ("demand_uk", dict(supplier_issue="none", item_issue="material"),
     "A demand change for one or two items in the UK market (a retailer listing, a promotion, a "
     "seasonal rise or a fall), with or without a size."),
    ("demand_korea", dict(supplier_issue="none", item_issue="material"),
     "A demand surge for an item in Korea, the supplier's home market (it may affect allocation or "
     "lead time, not UK sales)."),
    ("demand_unstated_market", dict(supplier_issue="none", item_issue="material"),
     "A demand surge for an item where the document does not say in which market."),
    ("critical_mixed", dict(supplier_issue="material", item_issue="none"),
     "The history mixes a minor or resolved issue (a late shipment that was recovered, a corrected "
     "invoice) with normal notes."),
    ("all_low", dict(supplier_issue="none", item_issue="none"),
     "History unremarkable or one mild note; in about half of these, a supplier notice that lead "
     "times are 3-5 weeks longer."),
    ("missing_values", dict(supplier_issue="none", item_issue="none"),
     "Ordinary history (the stock data will have gaps, which you do not write about)."),
    ("ambiguous_attribution", dict(supplier_issue="material", item_issue="none"),
     "A document mentions problems without making clear whether they are the supplier's or the "
     "items', or who reported them ('there have been reports of …')."),
]

TIE_IN = ("tie_in_sale", dict(supplier_issue="material", item_issue="none"),
          "The supplier, when asked for its usual products, pushes a newly launched product (give it "
          "a product name) to be bought with the order, and has hinted it gives supply priority to "
          "buyers who take it. Vary: stated in a supplier email or reported by a colleague; on a "
          "Critical or a Medium item; with or without the supply threat.")

SCHEMA = (
    'Return JSON: {"supplier": "<new company name>", "items": ["<CamelCase product name>", ...], '
    '"history": [{"supplier": "<name or empty>", "items": ["<product name>", ...], "date": "YYYY-MM-DD", '
    '"text": "<document>"}]}'
)

# Weeks to stockout by band, set in code so every type gets a realistic risk mix (the generator
# made almost every item Critical). Risk at 18 weeks: Critical < 18, High < 26, Medium < 35, Low.
BANDS = {"Critical": (2, 17), "High": (18, 25), "Medium": (26, 34), "Low": (35, 70)}
MIX = ["Critical"] * 2 + ["High"] * 3 + ["Medium"] * 3 + ["Low"] * 3


def figures(typ: str, n: int, r: random.Random) -> list[tuple]:
    if typ == "all_low":
        bands = ["Low"] * n
    elif typ == "critical_mixed":
        bands = ["Critical"] + [r.choice(MIX) for _ in range(n - 1)]
    else:
        bands = [r.choice(MIX) for _ in range(n)]
    out = []
    for b in bands:
        lo, hi = BANDS[b]
        wks = r.randint(lo, hi) if r.random() < 0.6 else round(r.uniform(lo, hi), 1)
        weekly = r.choice([12, 20, 35, 50, 80, 120, 200, 350])
        out.append([round(weekly * wks), wks])
    if typ == "all_low" and r.random() < 0.5:  # one item just inside Low, so a longer lead time moves it
        weekly = out[0][0] / out[0][1]
        out[0][1] = r.choice([36, 37, 38])
        out[0][0] = round(weekly * out[0][1])
    if typ == "missing_values":
        for k in r.sample(range(n), k=min(n, r.choice([1, 2]))):
            field = r.choice([0, 1])
            out[k][field] = r.choice([None, "", "-40", "n/a", "TBC"])
    return [tuple(x) for x in out]


def banned_names() -> tuple[set[str], set[str]]:
    sup, items = set(), set()
    for p in (ROOT / "eval_data" / "eval_scenarios.jsonl", ROOT / "eval_data" / "holdout_scenarios.jsonl"):
        for line in open(p, encoding="utf-8"):
            s = json.loads(line)
            sup.add(s["supplier"].lower())
            items |= {r["ItemName"].lower() for r in s["input_data"]}
    return sup, items


_lock = threading.Lock()


def generate(client: OpenAI, sid: str, typ: tuple, n_items: int, code_base: int,
             banned: tuple[set, set], used: tuple[set, set]) -> dict | None:
    name, labels, instruction = typ
    r = random.Random(sid)
    prompt = (f"{COMMON}\n\nScenario type: {instruction}\n\nThe supplier has {n_items} items in "
              f"{r.choice(LINES)}. Invent a new supplier name starting with the letter {r.choice(LETTERS)} "
              f"(not 'Haedam') and new product names (not generic like 'ItemA').\n\n{SCHEMA}")
    for _ in range(4):
        out = client.chat.completions.create(model=MODEL, reasoning_effort=EFFORT,
                                             response_format={"type": "json_object"},
                                             messages=[{"role": "user", "content": prompt}])
        try:
            d = json.loads(out.choices[0].message.content)
            sup = d["supplier"].strip()
            names = [str(i if isinstance(i, str) else i.get("ItemName", "")).strip() for i in d["items"]][:n_items]
        except Exception as e:
            print("  retry", sid, type(e).__name__, str(e)[:80], flush=True)
            continue
        if len(names) != n_items or not all(names):
            print("  retry", sid, "item count", len(names), flush=True)
            continue
        with _lock:
            # supplier names unique; product names only kept apart from the evaluation set (two
            # suppliers may well sell a product of the same name)
            clash = sup.lower() in banned[0] | used[0] or any(x.lower() in banned[1] for x in names)
            if clash or len(set(names)) != len(names):
                print("  retry", sid, "name clash", flush=True)
                continue
            used[0].add(sup.lower())
            used[1].update(x.lower() for x in names)
        figs = figures(name, n_items, r)
        rows = [{"ItemCode": str(code_base + k), "ItemName": nm, "SupplierName": sup,
                 "RiskLevel": risk_level_from_weeks(figs[k][1]),
                 "AvailableStock": figs[k][0], "WksToOOS": figs[k][1]} for k, nm in enumerate(names)]
        codes = {r["ItemName"]: r["ItemCode"] for r in rows}
        history = [{"supplier": h.get("supplier") or "", "items": h.get("items") or [],
                    "item_codes": [codes[i] for i in (h.get("items") or []) if i in codes],
                    "date": h.get("date") or "", "text": h["text"]} for h in d.get("history") or []]
        return {"scenario_id": sid, "category": name,
                "labels": {**labels, "data_issue": "missing_values" if name == "missing_values" else "none",
                           "history_present": bool(history)},
                "supplier": sup, "input_data": rows, "history": history}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tie-in", action="store_true")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.tie_in:
        out, jobs = OUT_DIR / "generalisation_tie_in.jsonl", [(f"tie_{k:02d}", TIE_IN, 900000 + k * 100) for k in range(1, 8)]
    else:
        out = OUT_DIR / "scenarios.jsonl"
        jobs = [(f"sft_{t[0]}_{k:02d}", t, 800000 + (i * PER_TYPE + k) * 100)
                for i, t in enumerate(TYPES) for k in range(1, PER_TYPE + 1)]
    done = {json.loads(l)["scenario_id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    banned = banned_names()
    used = (set(), set())
    for f in OUT_DIR.glob("*.jsonl"):  # names already used by earlier scenarios
        for line in open(f, encoding="utf-8"):
            s = json.loads(line)
            used[0].add(s["supplier"].lower())
            used[1].update(r["ItemName"].lower() for r in s["input_data"])
    rnd = random.Random(7)
    client = OpenAI(api_key=settings.openai_api_key, timeout=180)
    todo = [(sid, t, base, rnd.choice([2, 2, 3, 3, 3, 4, 4, 5, 6])) for sid, t, base in jobs if sid not in done]
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(generate, client, sid, t, n, base, banned, used): sid for sid, t, base, n in todo}
        for fut in as_completed(futs):
            s = fut.result()
            if s is None:
                print("FAILED", futs[fut], flush=True)
                continue
            with _lock:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                f.flush()
            print(s["scenario_id"], s["supplier"], len(s["input_data"]), "items,", len(s["history"]), "docs", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
