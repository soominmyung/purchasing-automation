"""
How the analysis agent's recommended actions state their timing, per base-output file
(docs/approval_model_comparison.md, round 5). Per file: actions; those saying they come before the
order, alongside it, after it; and dates in actions. "Overdue + before": an action that puts
something before the order while concerning an item whose latest order date has passed or falls on
the snapshot date, and does not also say the overdue order goes ahead now; read each one, since an
action naming several items may be about another of them.

  python scripts/count_action_timing.py eval_results/v5_base_outputs.jsonl [more files ...]
"""
import json
import re
import sys

BEFORE = re.compile(r"before (?:placing|place|order placement|releasing|release|committing|ordering|finalis|the order)", re.I)
ALONG = re.compile(r"alongside|in parallel|without delay|rather than delaying|while placing", re.I)
AFTER = re.compile(r"once the order|after (?:placing|the order)|once .* placed", re.I)
NOW = re.compile(r"immediately|now\b|without delay", re.I)
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) 20\d\d")


def main() -> None:
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        n = before = along = after = dates = clash = 0
        clashes = []
        for b in rows:
            p = b.get("approval_input") or {}
            overdue = {str(it.get("item_name")) for it in p.get("items", [])
                       if str(it.get("order_by_timing")) in ("overdue", "immediately")}
            for a in (b.get("analysis") or {}).get("recommended_actions") or []:
                t = str(a.get("action") or "")
                n += 1
                before += bool(BEFORE.search(t)); along += bool(ALONG.search(t)); after += bool(AFTER.search(t))
                dates += len(DATE.findall(t))
                names = {x for x in overdue if x in t} or (overdue if not a.get("items") else set())
                if BEFORE.search(t) and names and not ALONG.search(t) and not NOW.search(t):
                    clash += 1
                    clashes.append(f"{b['scenario_id']}: {t[:200]}")
        print(f"{path}: {len(rows)} analyses, {n} actions; before {before}, alongside {along}, after {after}; "
              f"dates {dates}; overdue + before {clash}")
        for c in clashes:
            print("   ", c)


if __name__ == "__main__":
    main()
