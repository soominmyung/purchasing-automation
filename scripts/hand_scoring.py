"""
Hand scoring of unmodified analyses: the false-alarm half of the judge calibration set
(docs/judge_validation.md).

  python scripts/hand_scoring.py sheet     -> eval_data/hand_scoring_sheet.md
  python scripts/hand_scoring.py collect   -> eval_data/hand_scores.jsonl

The sheet shows, for each chosen scenario, the inputs, GPT-4o's output and the judge
checks that apply, worded exactly as the judges receive them (plus a Korean gloss). The
scorer writes pass or fail, and optionally a note, in each row. `collect` reads the filled
sheet back and refuses to write anything while a row is blank or unreadable.
"""
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from services.quality_checks import applicable_judge_checks, check_text

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "eval_results" / "eval_v2_gpt4o_outputs.jsonl"
SCENARIOS = [ROOT / "eval_data" / "eval_scenarios.jsonl", ROOT / "eval_data" / "holdout_scenarios.jsonl"]
SHEET = ROOT / "eval_data" / "hand_scoring_sheet.md"
OUT_FILE = ROOT / "eval_data" / "hand_scores.jsonl"

# Every scenario type, and no-Critical, mixed and all-Critical suppliers.
CHOSEN = ["eval_01", "eval_03", "eval_06", "eval_08", "eval_11",
          "eval_14", "eval_16", "eval_18", "eval_19", "H3"]

GLOSS = {
    "J1": "히스토리에 없는 사건이나 추세를 지어내지 않았고, 있는 내용을 왜곡하지도 않았다 (예: 급증을 소폭 증가로).",
    "J1b": "히스토리가 있는 품목에 \"히스토리 없음\"이라고 쓰지 않았다. 공급사 히스토리는 모든 품목에 해당한다.",
    "J2a": "공급사 히스토리에 적힌 문제를 다루는 질문이 하나 이상 있다.",
    "J2b": "품목 히스토리에 적힌 문제를 다루는 질문이 하나 이상 있다.",
    "J2c": "데이터에 없는 문제를 전제한 질문이 없다. 중립적인 질문과 실제 문제에 대한 질문은 통과.",
    "J3": "히스토리가 없다고 밝혔고, 지어낸 히스토리가 없다.",
    "J5": "Critical 품목마다 요약에 품목명, 주어진 예상 품절 주수, 대응책(긴급 배송, 대체 공급처 등)이 있다.",
}

ROW = re.compile(r"^\|\s*(J\w+)\s*\|.*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$")
HEADING = re.compile(r"^## (\S+)")


def load():
    scen = {}
    for path in SCENARIOS:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                s = json.loads(line)
                scen[s["scenario_id"]] = s
    outs = {r["scenario_id"]: r for r in map(json.loads, SOURCE.read_text(encoding="utf-8").splitlines()) if r}
    return scen, outs


def checks_for(out: dict, scenario: dict) -> list[str]:
    return applicable_judge_checks(out["input"]["items"], out["supplier_history"], out["item_history"],
                                   labels=scenario["labels"])


def write_sheet():
    scen, outs = load()
    L = [
        "# Hand scoring sheet",
        "",
        "For each check, write `pass` or `fail` in the **Verdict** column; a note is optional.",
        "Judge the output against the inputs shown, and only on what the check asks.",
        "When done, run `python scripts/hand_scoring.py collect`.",
        "",
        "각 체크의 **Verdict** 칸에 `pass` 또는 `fail`을 적어 주세요. Note는 선택입니다.",
        "체크 문구(영어)는 judge가 받는 문장 그대로이고, 아래 한국어는 설명입니다.",
        "",
        f"Source: `{SOURCE.relative_to(ROOT).as_posix()}` (GPT-4o, unmodified)",
        "",
    ]
    for sid in CHOSEN:
        s, o = scen[sid], outs[sid]
        items = o["input"]["items"]
        out = o["output"]
        lab = s["labels"]
        L += [f"## {sid}", "",
              f"**Supplier:** {o['input']['supplier']}  ·  **Type:** `{s['category']}`  ·  "
              f"**Labels:** supplier_issue={lab['supplier_issue']}, item_issue={lab['item_issue']}", ""]
        L += ["| ItemCode | ItemName | Available stock | WksToOOS | Risk | Expected stockout (wks) |",
              "|---|---|---:|---:|---|---:|"]
        for it in items:
            gap = it["expected_stockout_weeks"] if it["risk_level"] == "Critical" else "-"
            L.append(f"| {it['item_code']} | {it['item_name']} | {it['available_stock']} | {it['wks_to_oos']} "
                     f"| {it['risk_level']} | {gap} |")
        if o.get("data_issue_lines"):
            L += ["", "Excluded before analysis: " + " ".join(o["data_issue_lines"])]
        L += ["", f"**Supplier history:** {o['supplier_history']}", "",
              f"**Item history:** {o['item_history']}", "",
              "<details open><summary><b>GPT-4o report</b></summary>", "",
              out["purchasing_report_markdown"].strip(), "", "</details>", "",
              "**Critical questions**", ""]
        qs = out.get("critical_questions") or []
        L += [f"{i}. {q.get('question')}  *(target: {q.get('target')}, reason: {q.get('reason')})*"
              for i, q in enumerate(qs, 1)] or ["(none)"]
        L += ["", "**Notes per item**", ""]
        L += [f"- `{e.get('item_code')}`: {e.get('notes')}" for e in out["replenishment_timeline"]]
        L += ["", "| Check | What it asks | Verdict | Note |", "|---|---|---|---|"]
        for c in checks_for(o, s):
            text = check_text(c, items, labelled=True).replace("|", "/")
            L.append(f"| {c} | {text}<br>*{GLOSS[c]}* |  |  |")
        L += ["", "---", ""]
    SHEET.write_text("\n".join(L), encoding="utf-8")
    n = sum(len(checks_for(outs[sid], scen[sid])) for sid in CHOSEN)
    print(f"{len(CHOSEN)} scenarios, {n} checks -> {SHEET}")


def collect():
    scen, outs = load()
    found, problems, sid = {}, [], None
    for line in SHEET.read_text(encoding="utf-8").splitlines():
        h = HEADING.match(line)
        if h:
            sid = h.group(1)
            continue
        m = ROW.match(line)
        if m and sid in CHOSEN and m.group(1) in GLOSS:
            verdict = m.group(2).strip().lower()
            if verdict not in ("pass", "fail"):
                problems.append(f"{sid} {m.group(1)}: verdict is {verdict!r}")
            found.setdefault(sid, {})[m.group(1)] = {"verdict": verdict, "note": m.group(3).strip()}
    for s in CHOSEN:
        missing = set(checks_for(outs[s], scen[s])) - set(found.get(s, {}))
        problems += [f"{s} {c}: row missing" for c in sorted(missing)]
    if problems:
        raise SystemExit("Not written; fix these rows first:\n  " + "\n  ".join(problems))
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in CHOSEN:
            f.write(json.dumps({
                "id": f"hand_{s}", "kind": "hand_scored", "scenario_id": s,
                "category": scen[s]["category"], "labels": scen[s]["labels"],
                "expected": {c: v["verdict"] for c, v in found[s].items()},
                "notes": {c: v["note"] for c, v in found[s].items() if v["note"]},
                "source": {"file": SOURCE.name, "model": outs[s]["model"]},
                "input": outs[s]["input"], "supplier_history": outs[s]["supplier_history"],
                "item_history": outs[s]["item_history"], "output": outs[s]["output"],
            }, ensure_ascii=False) + "\n")
    fails = sum(v == "fail" for s in CHOSEN for v in (x["verdict"] for x in found[s].values()))
    print(f"{len(CHOSEN)} scenarios, {sum(len(found[s]) for s in CHOSEN)} verdicts ({fails} fail) -> {OUT_FILE}")


if __name__ == "__main__":
    {"sheet": write_sheet, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "sheet"]()
