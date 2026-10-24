"""
The approval agent comparison, the judge's checks: P1a-c (nothing added, no judging words, no
overstatement) on every purchase request in eval_results/approval_model_comparison.jsonl (each model
and effort, 25 scenarios), three runs each, with the production judge (settings.evaluator_model,
chosen in docs/judge_selection.md) and the production P1 wording. P2 and P3 were checked in code
when the text was written.

Settings are written once to eval_results/approval_model_judged_settings.json; verdicts to
eval_results/approval_model_judged.jsonl; a stopped run resumes.

  python scripts/judge_approval_models.py            # run
  python scripts/judge_approval_models.py --report   # the table per setting
  --revised: the same for the texts written with the revised prompt
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_judge_validation as jv  # noqa: E402  (the usage recorder and file hash)
from config import settings  # noqa: E402
from services import quality_checks as q  # noqa: E402
from scripts.compare_approval_models import CONFIGS, label, settings_text  # noqa: E402

TEXTS = ROOT / "eval_results" / "approval_model_comparison.jsonl"
BASE = ROOT / "eval_results" / "v5_base_outputs.jsonl"
OUT = ROOT / "eval_results" / "approval_model_judged.jsonl"
SETTINGS = ROOT / "eval_results" / "approval_model_judged_settings.json"
if "--revised" in sys.argv:  # the prompt revised on 2026-10-24 (scripts/compare_approval_models.py --revised)
    TEXTS = ROOT / "eval_results" / "approval_model_comparison_revised.jsonl"
    OUT = ROOT / "eval_results" / "approval_model_judged_revised.jsonl"
    SETTINGS = ROOT / "eval_results" / "approval_model_judged_revised_settings.json"
if "--timing" in sys.argv:  # the texts written from the analyses rebuilt after the action-timing rule
    TEXTS = ROOT / "eval_results" / "approval_model_comparison_timing.jsonl"
    BASE = ROOT / "eval_results" / "v5_base_outputs_timing.jsonl"
    OUT = ROOT / "eval_results" / "approval_model_judged_timing.jsonl"
    SETTINGS = ROOT / "eval_results" / "approval_model_judged_timing_settings.json"
if "--estimate" in sys.argv:  # the chosen setting on the analyses rebuilt after changelog item 25
    TEXTS = ROOT / "eval_results" / "approval_model_comparison_estimate.jsonl"
    BASE = ROOT / "eval_results" / "v5_base_outputs_estimate.jsonl"
    OUT = ROOT / "eval_results" / "approval_model_judged_estimate.jsonl"
    SETTINGS = ROOT / "eval_results" / "approval_model_judged_estimate_settings.json"
JUDGE = settings.evaluator_model
PRICE = (0.10, 0.50)  # claude-haiku-5-5, per million tokens in / out, as in the judge validation
REPS = 3


def judge_one(payload: dict, approval: dict) -> dict:
    jv._usage.calls = []
    started = time.perf_counter()
    res = q.approval_checks(payload, approval, [], judge=JUDGE, retries=settings.judge_retries)
    tin = sum(c[0] for c in jv._usage.calls); tout = sum(c[1] for c in jv._usage.calls)
    return {"verdicts": {k: {"pass": v["pass"], "detail": v.get("detail", ""), "note": v.get("note", "")}
                         for k, v in res["checks"].items() if k.startswith("P1")},
            "seconds": round(time.perf_counter() - started, 2), "cost": round((tin * PRICE[0] + tout * PRICE[1]) / 1e6, 6)}


def fixed_settings() -> None:
    now = {"judge": JUDGE, "reps": REPS, "p1_wording": q._P1_TEXT, "judge_system": q.JUDGE_SYSTEM,
           "quality_checks_sha256": jv.sha256(ROOT / "services" / "quality_checks.py")}
    if SETTINGS.exists():
        was = json.loads(SETTINGS.read_text(encoding="utf-8"))
        changed = [k for k in now if was.get(k) != now[k]]
        if changed:
            raise SystemExit(f"settings differ from the recorded run ({', '.join(changed)})")
        return
    SETTINGS.write_text(json.dumps({**now, "started": datetime.now().isoformat(timespec="seconds")},
                                   ensure_ascii=False, indent=2), encoding="utf-8")


def run() -> None:
    fixed_settings()
    inputs = {b["scenario_id"]: b["approval_input"] for b in (json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip())
              if b.get("approval_input")}
    texts = [json.loads(l) for l in open(TEXTS, encoding="utf-8") if l.strip()]
    done = set()
    if OUT.exists():
        done = {(r["config"], r["scenario_id"], r["rep"]) for r in (json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip())
                if not r.get("error")}
    todo = [(t, rep) for t in texts for rep in range(1, REPS + 1)
            if (label(t["model"], t.get("effort")), t["scenario_id"], rep) not in done]
    print(f"{len(todo)} judgements to make", flush=True)
    lock = threading.Lock()
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(judge_one, inputs[t["scenario_id"]], t["approval"]): (t, rep) for t, rep in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            t, rep = futs[fut]
            try:
                res, err = fut.result(), None
            except Exception as e:
                res, err = {"verdicts": {}, "seconds": None, "cost": 0}, f"{type(e).__name__}: {e}"
            row = {"config": label(t["model"], t.get("effort")), "scenario_id": t["scenario_id"], "rep": rep, "error": err,
                   "at": datetime.now().isoformat(timespec="seconds"), **res}
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            fails = [k for k, v in res["verdicts"].items() if v.get("pass") is False]
            print(f"  [{n}/{len(todo)}] {row['config']:<26} {t['scenario_id']:<8} r{rep} fail {','.join(fails) or '-'}"
                  f"{' ERROR ' + err[:80] if err else ''}", flush=True)


def majority(rows: list[dict]) -> dict:
    """(config, scenario) -> {check: failed by the majority of runs}"""
    votes = {}
    for r in rows:
        for k, v in r["verdicts"].items():
            votes.setdefault((r["config"], r["scenario_id"]), {}).setdefault(k, []).append(v.get("pass") is False)
    return {key: {k: sum(v) * 2 > len(v) for k, v in checks.items()} for key, checks in votes.items()}


def report() -> None:
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip() and not json.loads(l).get("error")]
    texts = [json.loads(l) for l in open(TEXTS, encoding="utf-8") if l.strip()]
    maj = majority(rows)
    print("| Setting | Settings | P1a | P1b | P1c | Any P1 fail | P2 | P3 | Reasoning tokens (mean) | Time (mean) | Cost per request |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m, e in CONFIGS:
        c = label(m, e)
        T = [t for t in texts if label(t["model"], t.get("effort")) == c]
        if not T:
            continue
        M = {sid: v for (cc, sid), v in maj.items() if cc == c}
        fails = {k: sum(v.get(k, False) for v in M.values()) for k in ("P1a", "P1b", "P1c")}
        anyf = sum(any(v.values()) for v in M.values())
        rt = [t.get("reasoning_tokens") for t in T if t.get("reasoning_tokens") is not None]
        print(f"| `{c}` | {settings_text(m, e)} | {fails['P1a']} | {fails['P1b']} | {fails['P1c']} | {anyf}/{len(M)} | "
              f"{sum(t['p2_pass'] for t in T)}/{len(T)} | {sum(t['p3_pass'] for t in T)}/{len(T)} | "
              f"{(sum(rt) / len(rt)) if rt else float('nan'):.0f} | {sum(t['seconds'] for t in T) / len(T):.1f} s | "
              f"${sum(t['cost'] for t in T) / len(T):.4f} |")
    print(f"\njudge cost ${sum(r['cost'] for r in rows):.2f} for {len(rows)} judgements")


def reading_notes() -> dict:
    """(setting, scenario) -> the issue note from reading the text: the first run's notes from
    eval_results/approval_model_comparison.md, the effort runs' from approval_model_issues_effort.json."""
    if "--timing" in sys.argv:  # read in full, no issue found for either Sol setting (docs/approval_model_comparison.md, round 5)
        return {}
    if "--revised" in sys.argv:  # the texts written with the revised prompt have notes of their own
        revised = json.loads((ROOT / "eval_results" / "approval_model_issues_revised.json").read_text(encoding="utf-8"))
        return {(c, s): n for c, per in revised.items() if not c.startswith("_") for s, n in per.items()}
    notes, sid, model = {}, None, None
    for line in (ROOT / "eval_results" / "approval_model_comparison.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and " · " in line:
            sid = line[3:].split(" · ")[0]
        elif line.startswith("### "):
            model = line[4:].split(" · ")[0]
        elif line.startswith("> ") and sid and model:
            notes[(model, sid)] = "" if "No issue found" in line else line[2:].replace("**Issues:** ", "")
    effort = json.loads((ROOT / "eval_results" / "approval_model_issues_effort.json").read_text(encoding="utf-8"))
    notes.update({(c, s): n for c, per in effort.items() if not c.startswith("_") for s, n in per.items()})
    return notes


def sheet() -> None:
    """eval_results/approval_model_judged.md: per scenario, the input and every setting's text, with
    the judge's majority verdict on P1a-c (and its reason when it fails) and the issue found by reading."""
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip() and not json.loads(l).get("error")]
    texts = [json.loads(l) for l in open(TEXTS, encoding="utf-8") if l.strip()]
    bases = [b for b in (json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip()) if b.get("approval_input")]
    maj, notes = majority(rows), reading_notes()
    L = ["# Purchase request narrative: every model and effort, judged", "",
         f"Judge: `{JUDGE}`, P1a-c, three runs, majority verdict. \"Reading\": the issue found by reading the text "
         "(Claude, before the judge's verdicts). Settings: `scripts/compare_approval_models.py`.", ""]
    for b in bases:
        sid, p = b["scenario_id"], b["approval_input"]
        L += [f"## {sid} · {b['category']} · {p['supplier']}", "", "**Input given to the agent**", ""]
        L += [f"- {it['item_name']}: {it['risk_level']}, {it['weeks_to_stockout']} weeks to stockout, gap "
              f"{it['expected_stockout_weeks']}, quantity {it['suggested_quantity']}, order by {it['order_by']} "
              f"({it['order_by_timing']})" for it in p["items"]]
        L += [f"- Finding ({f['about']}) {f['text']} {f['documents'] or ''}" for f in p["findings"]]
        L += [f"- Change: {c['item']} {c['what']} {c['from']} → {c['to']}: {c['reason']}" for c in p["changes"]]
        L += [f"- Action: {a['action']}" for a in p["actions"]] + [""]
        for m, e in CONFIGS:
            c = label(m, e)
            t = next((t for t in texts if label(t["model"], t.get("effort")) == c and t["scenario_id"] == sid), None)
            if not t:
                continue
            ap = t["approval"]
            L += [f"### `{c}` · {settings_text(m, e)} · {t['justification_sentences']} sentences · {t['seconds']} s", ""]
            if "unparsed_output" in ap:
                L += ["**Not parsed** (the output is not valid JSON):", "", "> " + str(ap["unparsed_output"])[:1500], ""]
            else:
                L += [f"**Purpose.** {ap.get('purpose', '')}", "", f"**Justification.** {ap.get('justification', '')}", ""]
            failed = [k for k, v in maj.get((c, sid), {}).items() if v]
            reasons = []
            for k in failed:
                d = next((r["verdicts"][k].get("detail", "") for r in rows if r["config"] == c and r["scenario_id"] == sid
                          and r["verdicts"].get(k, {}).get("pass") is False), "")
                reasons.append(f"{k}: {d}")
            L += [f"> **Judge:** {'fail on ' + '; '.join(reasons) if failed else 'P1a-c pass'}  ",
                  f"> **Reading:** {notes.get((c, sid)) or 'no issue found'}", ""]
        L += ["---", ""]
    name = ("approval_model_judged_timing.md" if "--timing" in sys.argv else
            "approval_model_judged_revised.md" if "--revised" in sys.argv else "approval_model_judged.md")
    (ROOT / "eval_results" / name).write_text("\n".join(L), encoding="utf-8")
    print("-> eval_results/" + name)


if __name__ == "__main__":
    report() if "--report" in sys.argv else sheet() if "--sheet" in sys.argv else run()
