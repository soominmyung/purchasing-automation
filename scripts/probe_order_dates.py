"""
Probe: how gpt-4o works out the latest order date after a lead-time change.

In the Kestrel dev case the analysis wrote the same wrong order dates in two runs (ValveSeat
2025-04-12 for 2025-04-05, GasketKit 2025-04-26 for 2025-04-19, PistonRing and OilSeal
unchanged). Two questions to gpt-4o, recorded in dev/runs/ with every prompt:
  1. the date arithmetic alone, in the prompt's words;
  2. the first answer replayed, and the model asked to show how it got those dates (an
     explanation after the fact, which need not be how it worked them out).

usage: python scripts/probe_order_dates.py <run folder whose first attempt is replayed>
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from services.agents import AGENT_MODELS, _llm, history_for_model, to_model  # noqa: E402
from services.item_grouping import group_by_supplier_and_recommend  # noqa: E402
from services.prompts import ANALYSIS_AGENT_SYSTEM  # noqa: E402
from utils.csv_utils import parse_csv_rows  # noqa: E402

ARITHMETIC = [
    "What is 2025-09-06 minus 22 weeks? Give the date as YYYY-MM-DD.",
    "An item's latest delivery date is 2025-09-06 and its lead time is 22 weeks. Its latest order "
    "date is the latest delivery date minus the lead time in weeks. What is its latest order date? "
    "Give the date as YYYY-MM-DD.",
    "An item's latest delivery date is 2025-09-06. At a lead time of 18 weeks its latest order date "
    "is 2025-05-03. The lead time becomes 22 weeks. Its latest order date is the latest delivery date "
    "minus the lead time in weeks. What is its latest order date now? Give the date as YYYY-MM-DD.",
]
EXPLAIN = ("Show, step by step, how you worked out recommended_latest_po_date in figures_after_changes "
           "for 520002, 520003 and 520004: which date you started from, what you subtracted and the "
           "result. Do not correct anything; explain what you did.")


def main(earlier: Path) -> None:
    record = json.loads((earlier / "record.json").read_text(encoding="utf-8"))
    case = json.loads((ROOT / "dev" / "cases" / f"{record['case']}.json").read_text(encoding="utf-8"))
    group = group_by_supplier_and_recommend(parse_csv_rows(case["csv"], f"{case['name']}.csv"))[0]
    user = json.dumps({"snapshot_date": group["snapshot_date"], "supplier": group["supplier"],
                       "items": group["items"], "history": history_for_model(case["history"])}, ensure_ascii=False)
    first = record["analysis_attempts"][0]["output"]
    llm = _llm(AGENT_MODELS["analysis"])
    out = {"label": "probe: how gpt-4o works out the order date after a lead-time change",
           "replayed_from": earlier.name, "model": AGENT_MODELS["analysis"], "arithmetic": [], "explain": {}}
    for q in ARITHMETIC:
        for n in range(3):
            a = llm.invoke([HumanMessage(content=q)]).content
            out["arithmetic"].append({"question": q, "try": n + 1, "answer": a})
            print(f"[{n + 1}] {q[:70]}… -> {a.strip()[:200]}")
    messages = [SystemMessage(content=ANALYSIS_AGENT_SYSTEM), HumanMessage(content=user),
                AIMessage(content=json.dumps(to_model(first), ensure_ascii=False)), HumanMessage(content=EXPLAIN)]
    answer = llm.invoke(messages).content
    out["explain"] = {"system": ANALYSIS_AGENT_SYSTEM, "user": user,
                      "assistant": json.dumps(to_model(first), ensure_ascii=False), "question": EXPLAIN,
                      "answer": answer}
    print("\n--- explanation\n" + answer)
    folder = ROOT / "dev" / "runs" / f"{datetime.now():%Y-%m-%d_%H%M%S}_probe-order-date-arithmetic"
    folder.mkdir(parents=True)
    (folder / "record.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrecord: {folder}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
