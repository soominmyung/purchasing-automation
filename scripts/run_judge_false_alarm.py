"""
Judge validation, the false-alarm test (docs/judge_validation.md): judge configurations score the
six hand-scored v5 base outputs (eval_data/hand_scores_v5.jsonl: 89 verdicts, 88 pass and 1 fail),
each output three times, with the production prompt and wording. A judge's "fail" on a check the
person passed is a false alarm.

Per output and run: one analysis judgement (every applicable check, single mode) and one purchase
request judgement (P1a-c), as production makes them. Settings are written once to
eval_results/judge_false_alarm_v5_settings.json; verdicts to eval_results/judge_false_alarm_v5_runs.jsonl;
a stopped run resumes.

  python scripts/run_judge_false_alarm.py            # run
  python scripts/run_judge_false_alarm.py --report   # metrics
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

import scripts.run_judge_validation as jv  # noqa: E402  (the usage recorder and settings helpers)
from config import settings  # noqa: E402
from services import quality_checks as q  # noqa: E402

BASE = ROOT / "eval_results" / "v5_base_outputs.jsonl"
SCORES = ROOT / "eval_data" / "hand_scores_v5.jsonl"
SETTINGS = ROOT / "eval_results" / "judge_false_alarm_v5_settings.json"
RUNS = ROOT / "eval_results" / "judge_false_alarm_v5_runs.jsonl"
CONFIGS = ["anthropic:claude-haiku-5-5?effort=high&format=schema",   # the provisional choice
           "anthropic:claude-haiku-5-5?effort=medium&format=schema",
           "anthropic:claude-haiku-5-5",                              # forced tool, no thinking: production until now
           "openai:gpt-6-luna"]                                       # reference, the analysis model's family
PRICES = {"anthropic:claude-haiku-5-5": (0.10, 0.50), "openai:gpt-6-luna": (0.10, 0.50)}
REPS = 3


def judge_output(b: dict, spec: str) -> dict:
    jv._usage.calls = []
    started = time.perf_counter()
    items, docs, a = b["input"]["items"], b["history_docs"], b["analysis"]
    final = q.figures_after_changes(items, a, b["input"]["snapshot_date"])
    checks = q.applicable_judge_checks(items, docs, analysis=a)
    verdicts = q.run_judge(spec, q.judge_prompt(b["input"]["supplier"], items, docs, a, checks, labelled=False, after=final),
                           checks, retries=settings.judge_retries)
    verdicts = {k: {"pass": v["pass"], "detail": v.get("evidence", "")} for k, v in verdicts.items()}
    if b.get("approval"):
        res = q.approval_checks(b["approval_input"], b["approval"], [], judge=spec, retries=settings.judge_retries)
        verdicts.update({k: {"pass": v["pass"], "detail": v.get("detail", "")} for k, v in res["checks"].items() if k.startswith("P1")})
    tin = sum(c[0] for c in jv._usage.calls); tout = sum(c[1] for c in jv._usage.calls)
    p_in, p_out = PRICES[spec.split("?")[0]]
    return {"verdicts": verdicts, "seconds": round(time.perf_counter() - started, 2), "calls": len(jv._usage.calls),
            "input_tokens": tin, "output_tokens": tout, "cost": round((tin * p_in + tout * p_out) / 1e6, 6)}


def fixed_settings() -> None:
    now = {k: v for k, v in jv.current_settings().items() if k not in ("plan", "groups", "calibration_file", "calibration_sha256")}
    now.update({"configs": CONFIGS, "reps": REPS, "hand_scores_sha256": jv.sha256(SCORES), "base_sha256": jv.sha256(BASE)})
    if SETTINGS.exists():
        was = json.loads(SETTINGS.read_text(encoding="utf-8"))
        changed = [k for k in ("configs", "reps", "hand_scores_sha256", "base_sha256", "judge_system", "check_wording",
                               "quality_checks_sha256") if was.get(k) != now.get(k)]
        if changed:
            raise SystemExit(f"settings differ from the recorded run ({', '.join(changed)})")
        return
    now["started"] = datetime.now().isoformat(timespec="seconds")
    SETTINGS.write_text(json.dumps(now, ensure_ascii=False, indent=2), encoding="utf-8")


def run() -> None:
    fixed_settings()
    scores = [json.loads(l) for l in open(SCORES, encoding="utf-8") if l.strip()]
    sids = sorted({s["scenario_id"] for s in scores})
    bases = {b["scenario_id"]: b for b in (json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip()) if b["scenario_id"] in sids}
    done = set()
    if RUNS.exists():
        done = {(r["scenario_id"], r["config"], r["rep"]) for r in (json.loads(l) for l in open(RUNS, encoding="utf-8") if l.strip())}
    todo = [(sid, c, rep) for c in CONFIGS for rep in range(1, REPS + 1) for sid in sids if (sid, c, rep) not in done]
    print(f"{len(todo)} judgements to make", flush=True)
    lock = threading.Lock()
    with open(RUNS, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(judge_output, bases[sid], c): (sid, c, rep) for sid, c, rep in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            sid, c, rep = futs[fut]
            try:
                res, err = fut.result(), None
            except Exception as e:
                res, err = {"verdicts": {}, "seconds": None, "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0}, f"{type(e).__name__}: {e}"
            with lock:
                f.write(json.dumps({"scenario_id": sid, "config": c, "rep": rep, "error": err,
                                    "at": datetime.now().isoformat(timespec="seconds"), "ids": jv.IDS, **res}, ensure_ascii=False) + "\n"); f.flush()
            fails = sum(v.get("pass") is False for v in res["verdicts"].values())
            print(f"  [{n}/{len(todo)}] {c.split(':')[1]:<40} r{rep} {sid:<8} checks {len(res['verdicts'])} judged fail {fails} "
                  f"{res['seconds']}s ${res['cost']}{' ERROR ' + err[:80] if err else ''}", flush=True)


def report() -> None:
    # the hand scores, and rows written before 2026-10-29, keep the check ids of that time
    truth = {(s["scenario_id"], q.current_id(s["check"])): s["verdict"]
             for s in (json.loads(l) for l in open(SCORES, encoding="utf-8") if l.strip())}
    rows = [json.loads(l) for l in open(RUNS, encoding="utf-8") if l.strip()]
    for r in rows:
        if r.get("ids") != jv.IDS:
            r["verdicts"] = {q.current_id(k): v for k, v in r["verdicts"].items()}
    for c in CONFIGS:
        R = [r for r in rows if r["config"] == c]
        if not R:
            continue
        fa = n_pass = missing = caught = n_fail = 0
        per_cell, alarms = {}, {}
        for r in R:
            for chk, v in r["verdicts"].items():
                t = truth.get((r["scenario_id"], chk))
                if t is None:
                    continue
                per_cell.setdefault((r["scenario_id"], chk), []).append(v.get("pass"))
                if v.get("pass") is None:
                    missing += 1
                elif t == "pass":
                    n_pass += 1
                    if v["pass"] is False:
                        fa += 1
                        alarms[(r["scenario_id"], chk)] = alarms.get((r["scenario_id"], chk), 0) + 1
                else:
                    n_fail += 1
                    caught += v["pass"] is False
        flips = sum(len(set(v)) > 1 for v in per_cell.values()) / max(1, len(per_cell))
        secs = [r["seconds"] for r in R if r["seconds"]]
        print(f"{c:<55} false alarms {fa}/{n_pass} = {fa / max(1, n_pass):.1%} | eval_14 P1a caught {caught}/{n_fail} | "
              f"flip {flips:.1%} | missing {missing} | ${sum(r['cost'] for r in R) / len(R):.4f} per output | {sum(secs) / max(1, len(secs)):.1f}s")
        if alarms:
            print("    false alarms on: " + ", ".join(f"{s} {k} ×{n}" for (s, k), n in sorted(alarms.items())))


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
