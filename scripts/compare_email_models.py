"""
The supplier email's requests (email agent) written by several models from the same input: the email
input built from each of the 25 analyses in eval_results/v5_base_outputs_estimate.jsonl, run through
the production loop (draft, the validator, a rewrite by the same model after a fail, at most three
drafts). Settings per model and effort, as in scripts/compare_approval_models.py.

Recorded per setting and scenario: every draft with the validator's verdict, the final requests and
email, seconds, tokens and cost (writer and validator apart), and code check E3 (no item outside the
email named). The judge's checks (E1, E2, E4-E7) are made by --judge, three runs each.

  python scripts/compare_email_models.py            # write
  python scripts/compare_email_models.py --judge    # judge the final requests
  python scripts/compare_email_models.py --report   # the table per setting
  python scripts/compare_email_models.py --sheet    # every email side by side, with the verdicts

Out: eval_results/email_model_comparison.jsonl, eval_results/email_model_judged.jsonl (+ _settings.json),
eval_results/email_model_comparison.md
"""
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402

import scripts.run_judge_validation as jv  # noqa: E402  (the usage recorder and file hash)
from config import settings  # noqa: E402
from services import agents, documents  # noqa: E402
from services import quality_checks as q  # noqa: E402
from services.prompts import EMAIL_DRAFT_AGENT_SYSTEM, PROMPT_VERSION  # noqa: E402

BASE = ROOT / "eval_results" / "v5_base_outputs_estimate.jsonl"
OUT = ROOT / "eval_results" / "email_model_comparison.jsonl"
JUDGED = ROOT / "eval_results" / "email_model_judged.jsonl"
JUDGED_SETTINGS = ROOT / "eval_results" / "email_model_judged_settings.json"
SHEET = ROOT / "eval_results" / "email_model_comparison.md"
CONFIGS = [("gpt-5.4-mini", "none"), ("gpt-5.4-mini", "low"),
           ("gpt-6-luna", "none"), ("gpt-6-luna", "low"), ("gpt-6-luna", "medium"),
           ("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium")]
PRICES = {"gpt-5.4-mini": (0.75, 4.50), "gpt-6-luna": (0.10, 0.50), "gpt-6.1-sol": (2.00, 10.00),
          "gpt-4o-mini": (0.15, 0.60)}
MAX_DRAFTS = 3  # as agents.should_continue
JUDGE = settings.evaluator_model
REPS = 3

# The judge's checks, from the email prompt's rules (2026-10-25; second wording
# after the first run: E4 fails a question on an item outside the email, E6 takes a point's question,
# E7 no longer fails wording taken over from a point)
E_TEXT = {
    "E1": "The requests reveal no internal information: no stock quantities, weeks to stockout, risk levels "
          "or other internal assessments, and no mention that an internal review or analysis took place.",
    "E2": "The requests do not mention a problem the supplier had before, nor when it happened. A question "
          "about how a step of this order will proceed passes.",
    "E4": "Every point in points_to_clarify is covered by a question that would get the information this order "
          "needs, except what a point asks about items in other_items: that must be left out, and asking about such "
          "an item fails. Where a point concerns a problem the supplier had before, an indirect question about how "
          "the related step of this order will proceed or be checked covers it; the question need not ask about the "
          "problem itself. One question may cover several points. Passes when points_to_clarify is empty.",
    "E5": "No question is asked more than once.",
    "E6": "For every item in ask_earliest_dispatch, the requests ask for its earliest dispatch date or a "
          "delivery sooner than the standard lead time allows (such as expedited or partial delivery by the "
          "requested date), through a point to clarify or a question of their own. Passes when "
          "ask_earliest_dispatch is empty.",
    "E7": "The requests promise no purchase order and do not write out the list of items again (referring "
          "to the items listed above is fine). They read as polite, conversational requests to a business "
          "partner in flowing sentences, not as an internal note or a list of questions; a point used as "
          "written passes when it reads that way.",
}


def label(model: str, effort: str | None) -> str:
    return model if effort is None else f"{model}?effort={effort}"


def cost(usage: dict) -> tuple[float, float]:
    """(writer, validator) cost from a usage-by-model dict."""
    w = v = 0.0
    for name, u in usage.items():
        key = next((k for k in PRICES if name.startswith(k)), None)
        if not key:
            continue
        c = (u.get("input_tokens", 0) * PRICES[key][0] + u.get("output_tokens", 0) * PRICES[key][1]) / 1e6
        if key == agents.AGENT_MODELS["email_validator"]:
            v += c
        else:
            w += c
    return round(w, 6), round(v, 6)


def write_one(model: str, effort: str, base: dict) -> dict:
    inp = base["input"]
    state = {"snapshot_date": inp["snapshot_date"], "supplier": inp["supplier"], "items": inp["items"],
             "analysis_output": base["analysis"], "iteration_count": 0, "correction_feedback": ""}
    drafts, started = [], time.perf_counter()
    with get_usage_metadata_callback() as cb:
        while True:
            t = time.perf_counter()
            state.update(agents.email_draft_node(state))
            write_s = round(time.perf_counter() - t, 2)
            t = time.perf_counter()
            v = agents.validator_node(state)
            state.update({k: v[k] for k in ("is_valid_email", "correction_feedback")})
            drafts.append({"text": state["email_text"], "requests": documents.email_parts(state["email_text"])[0]["requests"],
                           "validator": v["validation_log"][0], "write_seconds": write_s,
                           "validate_seconds": round(time.perf_counter() - t, 2)})
            if agents.should_continue(state) == "email_doc":
                break
        usage = dict(cb.usage_metadata)
    md, problems = documents.render_email(state["email_text"], inp["items"], base["analysis"], inp["supplier"])
    payload = documents.email_input(inp["snapshot_date"], inp["supplier"], inp["items"], base["analysis"])
    final = drafts[-1]["requests"] or ""
    others = [n for n in payload["other_items"] if n and re.search(rf"\b{re.escape(n)}\b", final)]
    reasoning = sum((u.get("output_token_details") or {}).get("reasoning", 0) for n, u in usage.items()
                    if not n.startswith(agents.AGENT_MODELS["email_validator"]))
    w_cost, v_cost = cost(usage)
    return {"scenario_id": base["scenario_id"], "model": model, "effort": effort, "config": label(model, effort),
            "email_input": payload, "drafts": drafts, "final_requests": final, "email_md": md,
            "render_problems": problems, "passed_validator": bool(state.get("is_valid_email")),
            "draft_count": len(drafts), "seconds": round(time.perf_counter() - started, 2),
            "writer_cost": w_cost, "validator_cost": v_cost, "reasoning_tokens": reasoning,
            "e3_pass": not others, "e3_named": others}


def write() -> None:
    bases = [json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip()]
    done = set()
    if OUT.exists():
        done = {(r["config"], r["scenario_id"]) for r in (json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip())}
    with open(OUT, "a", encoding="utf-8") as f:
        for m, e in CONFIGS:
            mine = [b for b in bases if (label(m, e), b["scenario_id"]) not in done]
            if not mine:
                continue
            # one setting at a time: AGENT_MODELS and AGENT_EFFORT are shared
            agents.AGENT_MODELS["email_draft"] = agents.AGENT_MODELS["email_rewrite"] = m
            agents.AGENT_EFFORT["email_draft"] = agents.AGENT_EFFORT["email_rewrite"] = e
            with ThreadPoolExecutor(max_workers=4) as ex:
                for row in ex.map(lambda b: write_one(m, e, b), mine):
                    row.update({"prompt_version": PROMPT_VERSION, "prompt": EMAIL_DRAFT_AGENT_SYSTEM,
                                "validator_model": agents.AGENT_MODELS["email_validator"],
                                "at": datetime.now().isoformat(timespec="seconds")})
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    print(f"  {row['config']:<26} {row['scenario_id']:<8} drafts {row['draft_count']} "
                          f"{'PASS' if row['passed_validator'] else 'FAIL'} E3 {'ok' if row['e3_pass'] else row['e3_named']} "
                          f"{row['seconds']}s ${row['writer_cost'] + row['validator_cost']:.4f}", flush=True)


def judge_prompt(payload: dict, requests: str, checks: list[str]) -> str:
    return ("INPUT GIVEN TO THE WRITER (the email's requests paragraph; the greeting, purpose, item list, "
            "closing and signature are added by code)\n" + json.dumps(payload, ensure_ascii=False, indent=2) +
            "\n\nREQUESTS WRITTEN\n" + requests + "\n\nCHECKS\n" + "\n".join(f"- {c}: {E_TEXT[c]}" for c in checks) +
            '\n\nReturn JSON only: {"<check id>": {"evidence": "<short quote or reason>", "pass": true|false, '
            '"note": "<only if it fails: one plain sentence for the buyer>"}, ...} with one entry per check.')


def judge() -> None:
    now = {"judge": JUDGE, "reps": REPS, "checks": E_TEXT, "judge_system": q.JUDGE_SYSTEM,
           "quality_checks_sha256": jv.sha256(ROOT / "services" / "quality_checks.py")}
    if JUDGED_SETTINGS.exists():
        was = json.loads(JUDGED_SETTINGS.read_text(encoding="utf-8"))
        changed = [k for k in now if was.get(k) != now[k]]
        if changed:
            raise SystemExit(f"settings differ from the recorded run ({', '.join(changed)})")
    else:
        JUDGED_SETTINGS.write_text(json.dumps({**now, "started": datetime.now().isoformat(timespec="seconds")},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    done = set()
    if JUDGED.exists():
        done = {(r["config"], r["scenario_id"], r["rep"]) for r in (json.loads(l) for l in open(JUDGED, encoding="utf-8") if l.strip())
                if not r.get("error")}
    todo = [(r, rep) for r in rows for rep in range(1, REPS + 1) if (r["config"], r["scenario_id"], rep) not in done]
    checks = list(E_TEXT)
    print(f"{len(todo)} judgements to make", flush=True)

    def one(r: dict) -> dict:
        jv._usage.calls = []
        t = time.perf_counter()
        v = q.run_judge(JUDGE, judge_prompt(r["email_input"], r["final_requests"], checks), checks, retries=settings.judge_retries)
        tin = sum(c[0] for c in jv._usage.calls); tout = sum(c[1] for c in jv._usage.calls)
        return {"verdicts": {k: {"pass": x["pass"], "detail": x.get("evidence", "")} for k, x in v.items()},
                "seconds": round(time.perf_counter() - t, 2), "cost": round((tin * 0.10 + tout * 0.50) / 1e6, 6)}

    lock = threading.Lock()
    with open(JUDGED, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(one, r): (r, rep) for r, rep in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            r, rep = futs[fut]
            try:
                res, err = fut.result(), None
            except Exception as e:
                res, err = {"verdicts": {}, "seconds": None, "cost": 0}, f"{type(e).__name__}: {e}"
            with lock:
                f.write(json.dumps({"config": r["config"], "scenario_id": r["scenario_id"], "rep": rep, "error": err,
                                    "at": datetime.now().isoformat(timespec="seconds"), **res}, ensure_ascii=False) + "\n"); f.flush()
            fails = [k for k, x in res["verdicts"].items() if x.get("pass") is False]
            print(f"  [{n}/{len(todo)}] {r['config']:<26} {r['scenario_id']:<8} r{rep} fail {','.join(fails) or '-'}"
                  f"{' ERROR ' + err[:80] if err else ''}", flush=True)


def majority(rows: list[dict]) -> dict:
    votes = {}
    for r in rows:
        for k, v in r["verdicts"].items():
            votes.setdefault((r["config"], r["scenario_id"]), {}).setdefault(k, []).append(v.get("pass") is False)
    return {key: {k: sum(v) * 2 > len(v) for k, v in checks.items()} for key, checks in votes.items()}


def report() -> None:
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    judged = [json.loads(l) for l in open(JUDGED, encoding="utf-8") if l.strip()] if JUDGED.exists() else []
    maj = majority([r for r in judged if not r.get("error")])
    print("| Setting | Validator pass at 1st draft | Final pass | Drafts (mean) | E1 | E2 | E3 | E4 | E5 | E6 | E7 | Any fail | "
          "Reasoning tokens | Time | Writer cost | Validator cost |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, e in CONFIGS:
        c = label(m, e)
        R = [r for r in rows if r["config"] == c]
        if not R:
            continue
        M = {sid: v for (cc, sid), v in maj.items() if cc == c}
        fails = {k: sum(v.get(k, False) for v in M.values()) for k in E_TEXT}
        anyf = sum(any(v.values()) or not r["e3_pass"] for r in R for v in [M.get(r["scenario_id"], {})])
        n = len(R)
        print(f"| `{c}` | {sum(r['drafts'][0]['validator']['passed'] for r in R)}/{n} | {sum(r['passed_validator'] for r in R)}/{n} | "
              f"{sum(r['draft_count'] for r in R) / n:.2f} | {fails['E1']} | {fails['E2']} | {n - sum(r['e3_pass'] for r in R)} | "
              f"{fails['E4']} | {fails['E5']} | {fails['E6']} | {fails['E7']} | {anyf}/{n} | "
              f"{sum(r['reasoning_tokens'] for r in R) / n:.0f} | {sum(r['seconds'] for r in R) / n:.1f} s | "
              f"${sum(r['writer_cost'] for r in R) / n:.4f} | ${sum(r['validator_cost'] for r in R) / n:.5f} |")
    if judged:
        print(f"\njudge cost ${sum(r['cost'] for r in judged):.2f} for {len(judged)} judgements")


def sheet() -> None:
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    judged = [json.loads(l) for l in open(JUDGED, encoding="utf-8") if l.strip()] if JUDGED.exists() else []
    maj = majority([r for r in judged if not r.get("error")])
    L = ["# Supplier email requests: every model and effort", "",
         f"Validator: `{agents.AGENT_MODELS['email_validator']}` (production loop, at most {MAX_DRAFTS} drafts). "
         f"Judge: `{JUDGE}`, E1, E2, E4-E7, three runs, majority. E3 in code.", ""]
    for sid in dict.fromkeys(r["scenario_id"] for r in rows):
        first = next(r for r in rows if r["scenario_id"] == sid)
        p = first["email_input"]
        L += [f"## {sid} · {p['supplier']}", "", "**Input given to the agent**", ""]
        L += [f"- Item: {it['item_name']}, {it['quantity']}, by {it['requested_delivery_date']}" for it in p["items"]]
        L += [f"- Point to clarify: {x}" for x in p["points_to_clarify"]]
        L += [f"- Ask earliest dispatch: {', '.join(p['ask_earliest_dispatch']) or '-'}",
              f"- Other items (leave out): {', '.join(p['other_items']) or '-'}", ""]
        for m, e in CONFIGS:
            c = label(m, e)
            r = next((r for r in rows if r["config"] == c and r["scenario_id"] == sid), None)
            if not r:
                continue
            failed = [k for k, v in maj.get((c, sid), {}).items() if v]
            reasons = []
            for k in failed:
                d = next((j["verdicts"][k].get("detail", "") for j in judged if j["config"] == c and j["scenario_id"] == sid
                          and j["verdicts"].get(k, {}).get("pass") is False), "")
                reasons.append(f"{k}: {d}")
            L += [f"### `{c}` · drafts {r['draft_count']} · validator {'PASS' if r['passed_validator'] else 'FAIL'} · {r['seconds']} s", ""]
            for i, d in enumerate(r["drafts"][:-1], 1):
                L += [f"> Draft {i} rejected by the validator: {d['validator'].get('feedback', '')}", ""]
            L += [r["final_requests"] or "(no requests)", "",
                  f"> **Judge:** {'fail on ' + '; '.join(reasons) if failed else 'pass'}"
                  f"{'' if r['e3_pass'] else '  ' + chr(10) + '> **E3:** names ' + ', '.join(r['e3_named'])}", ""]
        L += ["---", ""]
    SHEET.write_text("\n".join(L), encoding="utf-8")
    print("->", SHEET)


if __name__ == "__main__":
    {"--judge": judge, "--report": report, "--sheet": sheet}.get(next((a for a in sys.argv[1:] if a.startswith("--")), ""), write)()
