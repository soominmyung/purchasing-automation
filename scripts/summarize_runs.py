"""
Summarise run records (runs/*.json, written by services/run_record.py).

One row per prompt version: supplier groups run, how many were marked for review, which
checks failed, checks the judge could not answer (counted apart from real failures), email
checks that failed in the end, average rewrites, tokens per model and time.

Production rates are for spotting regressions. Whether a prompt change helped is decided
on the fixed evaluation scenarios, since the data differs from week to week.

  python scripts/summarize_runs.py
  python scripts/summarize_runs.py --since 2026-10-01 --csv summary.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(runs_dir: Path, since: str | None):
    for f in sorted(runs_dir.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if since and rec.get("started_at", "") < since:
            continue
        yield rec


def summarise(records):
    by_version = defaultdict(lambda: {"runs": 0, "failed_runs": 0, "groups": 0, "review": 0,
                                      "check_fail": Counter(), "unanswered": 0, "email_failed": 0,
                                      "rewrites": 0, "tokens": Counter(), "seconds": 0.0})
    for rec in records:
        v = by_version[rec.get("prompt_version") or "unknown"]
        v["runs"] += 1
        v["failed_runs"] += rec.get("status") == "failed"
        for g in rec.get("groups", []):
            v["groups"] += 1
            ev = g.get("evaluation") or {}
            v["review"] += ev.get("verdict") == "needs_review"
            for cid, res in (ev.get("checks") or {}).items():
                if res.get("pass") is None:
                    v["unanswered"] += 1
                elif res.get("pass") is False:
                    v["check_fail"][cid] += 1
            em = g.get("email_validation") or {}
            v["email_failed"] += not em.get("passed", False)
            v["rewrites"] += max(0, (em.get("attempts") or 1) - 1)
            totals = g.get("totals") or {}
            v["seconds"] += totals.get("seconds") or 0
            for model, u in (totals.get("tokens") or {}).items():
                v["tokens"][model] += u.get("input_tokens", 0) + u.get("output_tokens", 0)
    return by_version


def pct(n, d):
    return f"{n} ({n / d:.0%})" if d else str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--csv", help="also write the table to this CSV file")
    args = ap.parse_args()

    table = summarise(load(Path(args.runs), args.since))
    if not table:
        print("No run records found.")
        return
    rows = []
    for version, v in sorted(table.items()):
        g = v["groups"]
        rows.append({
            "prompt_version": version,
            "runs": v["runs"], "failed_runs": v["failed_runs"], "supplier_groups": g,
            "needs_review": pct(v["review"], g),
            "check_failures": ", ".join(f"{c} {n}" for c, n in sorted(v["check_fail"].items())) or "-",
            "judge_unanswered": v["unanswered"],
            "email_failed_in_the_end": pct(v["email_failed"], g),
            "avg_email_rewrites": f"{v['rewrites'] / g:.1f}" if g else "-",
            "tokens_per_group": "; ".join(f"{m} {t // g:,}" for m, t in sorted(v["tokens"].items())) if g else "-",
            "seconds_per_group": f"{v['seconds'] / g:.0f}" if g else "-",
        })
    for r in rows:
        print(f"\nprompt {r['prompt_version']}: {r['runs']} runs ({r['failed_runs']} failed), {r['supplier_groups']} supplier groups")
        for k in ("needs_review", "check_failures", "judge_unanswered", "email_failed_in_the_end",
                  "avg_email_rewrites", "tokens_per_group", "seconds_per_group"):
            print(f"  {k:<24} {r[k]}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
