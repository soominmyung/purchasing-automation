"""
The email validator compared (services/agents.check_email: code checks, then the model in
AGENT_MODELS["email_validator"]), 2026-10-25.

Emails checked, each once per setting:
  - real: the 175 final requests of the email comparison's round 3 (eval_results/email_model_comparison.jsonl),
    labelled by the judge's last scoring (E1 or E2 failed by the majority) and code check E3;
  - seeded: Sol low's 25 clean requests with one fault added at the end:
      items_name      an item of other_items named ("Could you also confirm ... for <item>?"), where there is one
      items_indirect  the same item referred to without its name, where there is one
      past_problem    a past problem of the supplier and when it happened
      internal        internal stock information (weeks of stock left)
Settings: gpt-4o-mini (the validator until now) and claude-haiku-5-5 at low, medium and high effort.

  python scripts/compare_email_validators.py            # run
  python scripts/compare_email_validators.py --report   # detection and false alarms per setting

Out: eval_results/email_validator_comparison.jsonl (+ _settings.json); a stopped run resumes.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402

from services import agents, documents  # noqa: E402
from services.prompts import EMAIL_VALIDATOR_SYSTEM, EMAIL_VALIDATOR_TEMPLATE  # noqa: E402

EMAILS = ROOT / "eval_results" / "email_model_comparison.jsonl"
JUDGED = ROOT / "eval_results" / "email_model_judged.jsonl"
BASE = ROOT / "eval_results" / "v5_base_outputs_estimate.jsonl"
OUT = ROOT / "eval_results" / "email_validator_comparison.jsonl"
SETTINGS = ROOT / "eval_results" / "email_validator_comparison_settings.json"
CONFIGS = [("gpt-4o-mini", None), ("claude-haiku-5-5", "low"), ("claude-haiku-5-5", "medium"), ("claude-haiku-5-5", "high")]
PRICES = {"gpt-4o-mini": (0.15, 0.60), "claude-haiku-5-5": (0.10, 0.50)}  # per million tokens, as in the judge validation
SEED_BASE = "gpt-6.1-sol?effort=low"


def label(model, effort):
    return model if effort is None else f"{model}?effort={effort}"


def majority_fails() -> dict:
    votes = {}
    for l in open(JUDGED, encoding="utf-8"):
        j = json.loads(l)
        for k in ("E1", "E2"):
            v = j["verdicts"].get(k)
            if v:
                votes.setdefault((j["config"], j["scenario_id"]), {}).setdefault(k, []).append(v.get("pass") is False)
    return {key: [k for k, v in d.items() if sum(v) * 2 > len(v)] for key, d in votes.items()}


def cases() -> list[dict]:
    bases = {b["scenario_id"]: b for b in (json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip())}
    fails = majority_fails()
    out = []
    for r in (json.loads(l) for l in open(EMAILS, encoding="utf-8") if l.strip()):
        b = bases[r["scenario_id"]]
        refs = documents.email_other_items(b["input"]["snapshot_date"], b["input"]["items"], b["analysis"])
        bad = fails.get((r["config"], r["scenario_id"]), []) + ([] if r["e3_pass"] else ["E3"])
        out.append({"id": f"real:{r['config']}:{r['scenario_id']}", "kind": "real", "faults": bad,
                    "requests": r["final_requests"], "refs": refs})
        if r["config"] != SEED_BASE:
            continue
        text = r["final_requests"].rstrip()
        if refs:
            o = refs[0]["item_name"]
            out.append({"id": f"seed:items_name:{r['scenario_id']}", "kind": "items_name", "faults": ["items"], "refs": refs,
                        "requests": text + f" Could you also confirm current availability and lead time for {o}?"})
            out.append({"id": f"seed:items_indirect:{r['scenario_id']}", "kind": "items_indirect", "faults": ["items"], "refs": refs,
                        "requests": text + " Could you also confirm current availability and lead time for the other product"
                                           " we usually order from you alongside these?"})
        out.append({"id": f"seed:past_problem:{r['scenario_id']}", "kind": "past_problem", "faults": ["E2"], "refs": refs,
                    "requests": text + " Given the delivery problems we had with your shipment in January, could you"
                                       " confirm that this order will not be affected in the same way?"})
        it = next((i for i in b["input"]["items"] if isinstance(i.get("wks_to_oos"), (int, float))), None)
        if it:
            out.append({"id": f"seed:internal:{r['scenario_id']}", "kind": "internal", "faults": ["E1"], "refs": refs,
                        "requests": text + f" As our current stock of {it['item_name']} will last only"
                                           f" {it['wks_to_oos']} weeks, please treat this order as a priority."})
    return out


def check_one(case: dict, model: str) -> dict:
    raw = json.dumps({"requests": case["requests"]}, ensure_ascii=False)
    started = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        res = agents.check_email(raw, case["refs"])
    usage = dict(cb.usage_metadata)
    tin = sum(u.get("input_tokens", 0) for u in usage.values()); tout = sum(u.get("output_tokens", 0) for u in usage.values())
    p = PRICES[model]
    return {**res, "seconds": round(time.perf_counter() - started, 2), "input_tokens": tin, "output_tokens": tout,
            "cost": round((tin * p[0] + tout * p[1]) / 1e6, 6)}


def run() -> None:
    allc = cases()
    SETTINGS.write_text(json.dumps({"configs": CONFIGS, "prompt": EMAIL_VALIDATOR_TEMPLATE, "system": EMAIL_VALIDATOR_SYSTEM,
                                    "keywords": agents.LEAK_KEYWORDS, "cases": len(allc),
                                    "started": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    done = set()
    if OUT.exists():
        done = {(r["config"], r["id"]) for r in (json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()) if not r.get("error")}
    with open(OUT, "a", encoding="utf-8") as f:
        for m, e in CONFIGS:
            c = label(m, e)
            todo = [x for x in allc if (c, x["id"]) not in done]
            if not todo:
                continue
            agents.AGENT_MODELS["email_validator"], agents.AGENT_EFFORT["email_validator"] = m, e
            print(f"{c}: {len(todo)} emails", flush=True)

            def one(x):
                try:
                    return x, check_one(x, m), None
                except Exception as ex:
                    return x, {}, f"{type(ex).__name__}: {ex}"
            with ThreadPoolExecutor(max_workers=4) as ex:
                for x, res, err in ex.map(one, todo):
                    f.write(json.dumps({"config": c, "id": x["id"], "kind": x["kind"], "faults": x["faults"], "error": err,
                                        "at": datetime.now().isoformat(timespec="seconds"), **res}, ensure_ascii=False) + "\n")
                    f.flush()


def report() -> None:
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    kinds = ["items_name", "items_indirect", "past_problem", "internal"]
    print("| Setting | " + " | ".join(f"Seeded {k}" for k in kinds) + " | Real faulty caught | Real clean passed (false alarms) | Caught by code | Time (mean) | Cost per email |")
    print("|---|" + "---|" * (len(kinds) + 5))
    for m, e in CONFIGS:
        c = label(m, e)
        R = [r for r in rows if r["config"] == c and not r.get("error")]
        if not R:
            continue
        cells = []
        for k in kinds:
            K = [r for r in R if r["kind"] == k]
            cells.append(f"{sum(not r['passed'] for r in K)}/{len(K)}")
        bad = [r for r in R if r["kind"] == "real" and r["faults"]]
        clean = [r for r in R if r["kind"] == "real" and not r["faults"]]
        code = sum(r["check"] in ("keywords", "items") and not r["passed"] for r in R)
        secs = [r["seconds"] for r in R]
        print(f"| `{c}` | " + " | ".join(cells) + f" | {sum(not r['passed'] for r in bad)}/{len(bad)} | "
              f"{sum(not r['passed'] for r in clean)}/{len(clean)} | {code} | {sum(secs) / len(secs):.1f} s | "
              f"${sum(r['cost'] for r in R) / len(R):.5f} |")
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n{len(errs)} errors, e.g. {errs[0]['error'][:150]}")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
