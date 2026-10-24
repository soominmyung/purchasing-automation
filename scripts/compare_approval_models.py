"""
The purchase request's narrative (approval agent) written by several models from the same input:
the approval input saved with each of the 25 v5 base outputs (eval_results/v5_base_outputs.jsonl),
so the analysis is not run again and every model gets exactly what the pipeline gave gpt-4o.

Recorded per model and scenario: Purpose and Justification, seconds, tokens and cost, and the code
checks (P2 sentences, P3 citations). The judge's checks (P1a-c) are run later, once the judge has
been validated (docs/judge_validation.md).

Run: python scripts/compare_approval_models.py
Out: eval_results/approval_model_comparison.jsonl (resumes where it stopped)
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402

from services import agents  # noqa: E402
from services import quality_checks as q  # noqa: E402
from services.prompts import APPROVAL_AGENT_SYSTEM, PROMPT_VERSION  # noqa: E402

BASE = ROOT / "eval_results" / "v5_base_outputs.jsonl"
OUT = ROOT / "eval_results" / "approval_model_comparison.jsonl"
MODELS = ["gpt-4o", "gpt-6-luna", "gpt-5.4-mini", "gpt-6.1-sol"]
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-6-luna": (0.10, 0.50), "gpt-5.4-mini": (0.75, 4.50), "gpt-6.1-sol": (2.00, 10.00)}


def write_one(model: str, base: dict) -> dict:
    agents.AGENT_MODELS["approval"] = model  # read when run_approval_agent builds its model
    started = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        approval = agents.run_approval_agent(base["approval_input"])
    seconds = round(time.perf_counter() - started, 2)
    tin = sum(u.get("input_tokens", 0) for u in cb.usage_metadata.values())
    tout = sum(u.get("output_tokens", 0) for u in cb.usage_metadata.values())
    pur, jus = str(approval.get("purpose") or ""), str(approval.get("justification") or "")
    cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", pur + " " + jus)})
    p_in, p_out = PRICES[model]
    return {"scenario_id": base["scenario_id"], "model": model, "approval": approval, "seconds": seconds,
            "input_tokens": tin, "output_tokens": tout, "cost": round((tin * p_in + tout * p_out) / 1e6, 6),
            "purpose_sentences": q._sentences(pur), "justification_sentences": q._sentences(jus),
            "p2_pass": 1 <= q._sentences(pur) <= 2 and 2 <= q._sentences(jus) <= 4,
            "citations": cites, "p3_pass": all(1 <= c <= len(base["history_docs"]) for c in cites),
            "unparsed": "unparsed_output" in approval}


def main() -> None:
    bases = [json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip()]
    bases = [b for b in bases if b.get("approval_input")]
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["model"], json.loads(l)["scenario_id"]) for l in open(OUT, encoding="utf-8") if l.strip()}
    todo = [(m, b) for m in MODELS for b in bases if (m, b["scenario_id"]) not in done]
    print(f"{len(todo)} to write", flush=True)
    # one model at a time: AGENT_MODELS is shared, so the calls for one model run together
    with open(OUT, "a", encoding="utf-8") as f:
        for m in MODELS:
            mine = [b for mm, b in todo if mm == m]
            if not mine:
                continue
            agents.AGENT_MODELS["approval"] = m
            with ThreadPoolExecutor(max_workers=4) as ex:
                for row in ex.map(lambda b: write_one(m, b), mine):
                    row.update({"prompt_version": PROMPT_VERSION, "prompt": APPROVAL_AGENT_SYSTEM,
                                "at": datetime.now().isoformat(timespec="seconds")})
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    print(f"  {m:<14} {row['scenario_id']:<8} {row['seconds']:>6}s just {row['justification_sentences']} "
                          f"{'P2 ok' if row['p2_pass'] else 'P2 FAIL'} ${row['cost']}", flush=True)


def sheet() -> None:
    """eval_results/approval_model_comparison.md: each scenario's input facts and the four models'
    Purpose and Justification side by side, for reading."""
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    bases = {b["scenario_id"]: b for b in (json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip())}
    L = ["# Purchase request narrative: four models, same input", "",
         "Each model got the approval input saved with the v5 base output (no style examples in the store).", ""]
    for sid in bases:
        b = bases[sid]
        if not b.get("approval_input"):
            continue
        p = b["approval_input"]
        L += [f"## {sid} · {b['category']} · {p['supplier']}", "", "**Input given to the agent**", ""]
        L += [f"- {it['item_name']}: {it['risk_level']}, {it['weeks_to_stockout']} weeks to stockout, gap "
              f"{it['expected_stockout_weeks']}, quantity {it['suggested_quantity']}, order by {it['order_by']} "
              f"({it['order_by_timing']})" for it in p["items"]]
        L += [f"- Finding ({f['about']}) {f['text']} {f['documents'] or ''}" for f in p["findings"]]
        L += [f"- Change: {c['item']} {c['what']} {c['from']} → {c['to']}: {c['reason']}" for c in p["changes"]]
        L += [f"- Action: {a['action']}" for a in p["actions"]] + [""]
        for m in MODELS:
            r = next((r for r in rows if r["model"] == m and r["scenario_id"] == sid), None)
            if not r:
                continue
            ap = r["approval"]
            L += [f"### {m} · {r['justification_sentences']} sentences · {r['seconds']} s", "",
                  f"**Purpose.** {ap.get('purpose', '')}", "", f"**Justification.** {ap.get('justification', '')}", ""]
        L += ["---", ""]
    (ROOT / "eval_results" / "approval_model_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("-> eval_results/approval_model_comparison.md")


if __name__ == "__main__":
    sheet() if "--sheet" in sys.argv else main()
