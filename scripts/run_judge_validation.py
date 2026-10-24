"""
Judge validation, prompt v5 (docs/judge_validation.md): each candidate judges the calibration set
(eval_data/judge_calibration_v5.jsonl) with the production judge prompt and wording
(services/quality_checks.py), and every verdict is recorded with its tokens and time.

Modes:
  single  every applicable check of an output in one call, as production does
  split   one call per group of checks, the calls made at the same time:
          J1-J2 (history and questions), J6-J7a (changes and figures), J8-J11 (the rest)
Purchase request samples (P1a-c) are judged as production does, one call for P1a-c, in single mode.

The settings of a run are written once to eval_results/judge_validation_v5_settings.json (models,
modes, repetitions, the judge's system prompt and every check's wording, the calibration file's
hash, the code's commit) and the run refuses to continue under different settings. Verdicts go to
eval_results/judge_validation_v5_runs.jsonl, one line per (sample, model, mode, repetition); a
stopped run resumes where it stopped.

  python scripts/run_judge_validation.py --smoke     # two samples per mode and model, then stop
  python scripts/run_judge_validation.py             # the full plan
  python scripts/run_judge_validation.py --report    # the metrics, from what has been recorded
"""
import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from services import quality_checks as q  # noqa: E402

CALIBRATION = ROOT / "eval_data" / "judge_calibration_v5.jsonl"
SETTINGS = ROOT / "eval_results" / "judge_validation_v5_settings.json"
RUNS = ROOT / "eval_results" / "judge_validation_v5_runs.jsonl"

# The plan agreed on 2026-10-24 (docs/judge_validation.md, Candidates): Haiku is the production
# candidate, Luna a reference of the analysis model's family, Sonnet the accuracy reference in
# single mode once; Gemini Flash left out on cost and availability.
PLAN = [
    {"model": "anthropic:claude-haiku-5-5", "modes": ["single", "split"], "reps": 3},
    {"model": "openai:gpt-6-luna", "modes": ["single", "split"], "reps": 3},
    {"model": "anthropic:claude-sonnet-5-5", "modes": ["single"], "reps": 1},
]
PRICES = {"anthropic:claude-haiku-5-5": (0.10, 0.50), "openai:gpt-6-luna": (0.10, 0.50),
          "anthropic:claude-sonnet-5-5": (2.00, 10.00)}  # $ per 1M tokens, input / output
GROUPS = [["J1a", "J1b", "J1c", "J2a", "J2b", "J2c"], ["J6a", "J6b", "J6c", "J7a"], ["J8", "J9", "J10", "J11"]]
WORKERS = 4

_usage = threading.local()
_original_call = q._judge_call


def _recording_call(*args, **kwargs):
    out = _original_call(*args, **kwargs)
    u = out.get("usage_metadata") or {}
    calls = getattr(_usage, "calls", None)
    if calls is not None:
        calls.append((u.get("input_tokens", 0), u.get("output_tokens", 0)))
    return out


q._judge_call = _recording_call  # run_judge looks it up at call time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_settings() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "services/quality_checks.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return {
        "plan": PLAN, "groups": GROUPS, "prices_per_million": PRICES,
        "judge_retries": settings.judge_retries, "judge_timeout_seconds": settings.judge_timeout_seconds,
        "anthropic_max_tokens": 4000,
        "judge_system": q.JUDGE_SYSTEM, "check_wording": {**q._JUDGE_TEXT, **q._J2_UNLABELLED, **q._P1_TEXT},
        "calibration_file": str(CALIBRATION.relative_to(ROOT)), "calibration_sha256": sha256(CALIBRATION),
        "code_commit": commit, "quality_checks_uncommitted_changes": bool(dirty),
        "quality_checks_sha256": sha256(ROOT / "services" / "quality_checks.py"),
    }


def fixed_settings() -> dict:
    now = current_settings()
    if SETTINGS.exists():
        was = json.loads(SETTINGS.read_text(encoding="utf-8"))
        keys = ("plan", "groups", "judge_system", "check_wording", "calibration_sha256", "quality_checks_sha256",
                "judge_retries", "judge_timeout_seconds", "anthropic_max_tokens")
        changed = [k for k in keys if was.get(k) != now.get(k)]
        if changed:
            raise SystemExit(f"settings differ from the recorded run ({', '.join(changed)}): start a new run file")
        return was
    now["started"] = datetime.now().isoformat(timespec="seconds")
    SETTINGS.write_text(json.dumps(now, ensure_ascii=False, indent=2), encoding="utf-8")
    return now


def judge_sample(s: dict, model: str, mode: str) -> dict:
    """One sample judged once; returns the verdicts, tokens and time."""
    _usage.calls = []
    started = time.perf_counter()
    if s["target_check"].startswith("P"):
        res = q.approval_checks(s["approval_input"], s["approval"], [], judge=model, retries=settings.judge_retries)
        verdicts = {k: v for k, v in res["checks"].items() if k.startswith("P1")}
    else:
        final = q.figures_after_changes(s["items"], s["analysis"], s["snapshot_date"])
        checks = q.applicable_judge_checks(s["items"], s["history_docs"], analysis=s["analysis"])
        def run(cs):
            p = q.judge_prompt(s["supplier"], s["items"], s["history_docs"], s["analysis"], cs, labelled=False, after=final)
            return q.run_judge(model, p, cs, retries=settings.judge_retries)
        if mode == "single":
            judged = run(checks)
        else:
            parts = [[c for c in g if c in checks] for g in GROUPS]
            parts = [p for p in parts if p]
            calls = _usage.calls
            judged = {}
            def run_part(cs):
                _usage.calls = calls
                return q.run_judge(model, q.judge_prompt(s["supplier"], s["items"], s["history_docs"], s["analysis"],
                                                         cs, labelled=False, after=final), cs, retries=settings.judge_retries)
            with ThreadPoolExecutor(max_workers=len(parts)) as ex:
                for r in ex.map(run_part, parts):
                    judged.update(r)
        verdicts = {k: {"pass": v["pass"], "detail": v.get("evidence", ""), "note": v.get("note", "")} for k, v in judged.items()}
    seconds = round(time.perf_counter() - started, 2)
    tin = sum(c[0] for c in _usage.calls); tout = sum(c[1] for c in _usage.calls)
    p_in, p_out = PRICES[model]
    target = verdicts.get(s["target_check"], {})
    return {"verdicts": verdicts, "target_pass": target.get("pass"), "seconds": seconds,
            "calls": len(_usage.calls), "input_tokens": tin, "output_tokens": tout,
            "cost": round((tin * p_in + tout * p_out) / 1e6, 6)}


def jobs(samples, smoke: bool):
    out = []
    for p in PLAN:
        for mode in p["modes"]:
            chosen = samples
            if smoke:  # one analysis and one purchase request sample, once
                chosen = [next(s for s in samples if s["target_check"] == "J7a" and s["base"].startswith("dev:")),
                          next(s for s in samples if s["target_check"].startswith("P"))]
            for rep in range(1, (1 if smoke else p["reps"]) + 1):
                for s in chosen:
                    if mode == "split" and s["target_check"].startswith("P"):
                        continue  # P1 is one group either way
                    out.append((s, p["model"], mode, rep))
    return out


def run(smoke: bool) -> None:
    fixed_settings()
    samples = [json.loads(l) for l in open(CALIBRATION, encoding="utf-8") if l.strip()]
    done = set()
    if RUNS.exists():
        for l in open(RUNS, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["sample_id"], r["model"], r["mode"], r["rep"], r.get("smoke", False)))
    todo = [j for j in jobs(samples, smoke) if (j[0]["id"], j[1], j[2], j[3], smoke) not in done]
    print(f"{len(todo)} judgements to make ({len(done)} recorded)", flush=True)
    lock = threading.Lock()
    with open(RUNS, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(judge_sample, s, model, mode): (s, model, mode, rep) for s, model, mode, rep in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            s, model, mode, rep = futures[fut]
            try:
                res, error = fut.result(), None
            except Exception as e:  # recorded, never dropped
                res, error = {"verdicts": {}, "target_pass": None, "seconds": None, "calls": 0,
                              "input_tokens": 0, "output_tokens": 0, "cost": 0}, f"{type(e).__name__}: {e}"
            row = {"sample_id": s["id"], "target_check": s["target_check"], "expected": s["expected"],
                   "base": s["base"], "model": model, "mode": mode, "rep": rep, "smoke": smoke,
                   "at": datetime.now().isoformat(timespec="seconds"), "error": error, **res}
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            mark = {True: "pass", False: "fail", None: "none"}[res["target_pass"]]
            print(f"  [{n}/{len(todo)}] {model.split(':')[1]:<18} {mode:<6} r{rep} {s['id']} {s['target_check']:<4} "
                  f"expected {s['expected']:<4} got {mark:<4} {res['seconds']}s ${res['cost']}{' ERROR ' + error[:80] if error else ''}",
                  flush=True)


def report() -> None:
    rows = [json.loads(l) for l in open(RUNS, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("smoke")]
    for p in PLAN:
        for mode in p["modes"]:
            R = [r for r in rows if r["model"] == p["model"] and r["mode"] == mode]
            if not R:
                continue
            fails = [r for r in R if r["expected"] == "fail"]
            passes = [r for r in R if r["expected"] == "pass"]
            det = sum(r["target_pass"] is False for r in fails) / max(1, len(fails))
            fa = sum(r["target_pass"] is False for r in passes) / max(1, len(passes))
            noans = sum(r["target_pass"] is None for r in R) / len(R)
            by_sample = {}
            for r in R:
                by_sample.setdefault(r["sample_id"], set()).add(r["target_pass"])
            flips = sum(len(v) > 1 for v in by_sample.values()) / max(1, len(by_sample)) if p["reps"] > 1 else None
            secs = [r["seconds"] for r in R if r["seconds"]]
            print(f"{p['model']:<30} {mode:<6} n={len(R):>4} detection {det:6.1%} false-alarm(real passes) {fa:6.1%} "
                  f"flip {('%5.1f%%' % (flips * 100)) if flips is not None else '  n/a'} no-answer {noans:5.1%} "
                  f"mean ${sum(r['cost'] for r in R) / len(R):.5f} mean {sum(secs) / max(1, len(secs)):.1f}s")
            per = {}
            for r in fails:
                per.setdefault(r["target_check"], []).append(r["target_pass"] is False)
            print("    detection by check: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(per.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.smoke)
