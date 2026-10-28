"""
Hand scoring for judge validation, prompt v5 (docs/judge_validation.md, the false-alarm test).

A person scores unmodified v5 base outputs (eval_results/v5_base_outputs.jsonl) on every judge check
that applies, worded exactly as the judge receives it and chosen by the same code the judge uses,
plus P1a-c on the purchase request's text. Those verdicts are the answer key for false alarms: a
judge that fails a check the person passed is a false alarm.

  python scripts/hand_scoring_v5.py sheet     # writes eval_data/hand_scoring_sheet_v5.md
  python scripts/hand_scoring_v5.py collect   # reads the filled sheet -> eval_data/hand_scores_v5.jsonl
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import quality_checks as q  # noqa: E402

BASE = ROOT / "eval_results" / "v5_base_outputs.jsonl"
SHEET = ROOT / "eval_data" / "hand_scoring_sheet_v5.md"
SCORES = ROOT / "eval_data" / "hand_scores_v5.jsonl"
# one of each kind of scenario: clean with a Critical item, no history, ambiguous attribution with a
# lead-time change, a supplier incident with a lead-time change, an item incident, a holdout
CHOSEN = ["eval_02", "eval_06", "eval_08", "eval_11", "eval_14", "H2"]
# Check ids as they were when the sheet was scored (before 2026-10-29); readers map them with
# quality_checks.current_id.
ORDER = ["J1a", "J1b", "J1c", "J2a", "J2b", "J2c", "J6a", "J6b", "J6c", "J7a", "J8", "J9", "J10", "J11"]
KO = {
    "J1a": "이력에 대한 모든 서술이 주어진 문서에 있다. 지어낸 사건이나 추세가 없다.",
    "J1b": "문서 내용을 더 강하게, 약하게, 다르게 바꾸지 않았다.",
    "J1c": "문서가 지정하지 않은 공급사나 품목의 일로 쓰지 않았다.",
    "J2a": "이력에 발주에 중요한 공급사 문제가 있으면, 그걸 다루는 질문이 있다. 없으면 통과.",
    "J2b": "이력에 발주에 중요한 품목 문제가 있으면, 그걸 다루는 질문이 있다. 없으면 통과.",
    "J2c": "데이터에 없는 문제를 전제한 질문이 없다. 중립적 질문과 실제 문제에 대한 질문은 통과.",
    "J6a": "각 변경(리드타임, 수량)이 인용한 finding이 그 방향의 근거가 된다.",
    "J6b": "변경의 크기가 근거에 맞고 이유가 그 크기를 설명한다. 기간이 없는 지연에 추정 크기를 쓰고 추정이라고 밝혔으면, 묘사에 걸맞을 때 통과.",
    "J6c": "같은 이유가 적용되는 품목이 빠지지 않았다. 빠졌다면 findings가 뒷받침하는 이유가 있다.",
    "J7a": "글(summary, findings, 질문, 액션, 변경 이유)의 재고 수치가 그 품목의 변경 후 수치와 정확히 같다.",
    "J8": "공급사 질문이 내부 정보(재고 수준, 소진 주수, 부족 정도, 예산, 분석)를 드러내지 않는다. 긴급·분할 납품 문의는 허용.",
    "J9": "각 질문과 액션이 인용한 finding에서 나온다.",
    "J10": "어디에도 쓰이지 않은 문서(문구에 번호가 있음) 중, 이번 발주에 중요한 문제를 담은 것이 없다.",
    "J11": "findings가 판단 없이 문서나 재고 수치의 사실만 쓴다.",
    "P1a": "Purpose와 Justification이 입력에 없는 사실, 위험, 평가를 보태지 않는다. 액션이 왜 중요한지 말하는 것과 발주 전 확인을 조건으로 쓰는 것은 허용.",
    "P1b": "수치를 입력 그대로 쓰고, 수치를 평가하는 말을 붙이지 않는다.",
    "P1c": "입력과 다른 단어를 써도 뜻이 바뀌지 않는다 (예: 'about 60%'를 '60%'로 쓰면 실패).",
}


def cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def sheet() -> None:
    bases = {b["scenario_id"]: b for b in (json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip())}
    L = ["# Hand scoring sheet, prompt v5", "",
         "For each check write `pass` or `fail` in the **Verdict** column, and the reason in **Reason** "
         "(which part of the output decides it). Judge the output against the inputs shown, and only on what "
         "the check asks.", "",
         "각 검사의 **Verdict** 칸에 `pass` 또는 `fail`을, **Reason** 칸에 판정 이유(출력의 어느 부분 때문인지)를 "
         "적어 주세요. 검사 문구(영어)는 judge가 받는 문장 그대로이고, 아래 한국어는 설명입니다. 끝나면 "
         "`python scripts/hand_scoring_v5.py collect`를 실행합니다.", "",
         "Source: `eval_results/v5_base_outputs.jsonl` (analysis gpt-6.1-sol, purchase request gpt-4o; first "
         "answers, unmodified).", ""]
    for sid in CHOSEN:
        b = bases[sid]
        a, items, docs = b["analysis"], b["input"]["items"], b["history_docs"]
        after = {str(i["item_code"]): i for i in q.figures_after_changes(items, a, b["input"]["snapshot_date"])}
        L += ["---", "", f"## {sid}", "",
              f"**Supplier:** {b['input']['supplier']} · **Type:** `{b['category']}` · **Snapshot:** {b['input']['snapshot_date']}", "",
              "| Item | Available stock | Weeks to stockout | Lead time | Risk | Gap (weeks) | Order by | Deliver by | Suggested qty |",
              "|---|---:|---:|---|---|---:|---|---|---:|"]
        for it in items:
            f = after[str(it["item_code"])]
            ch = lambda x, y: f"{x} → **{y}**" if x != y else f"{x}"
            L.append(f"| {it['item_name']} ({it['item_code']}) | {it['available_stock']} | {it['wks_to_oos']} | "
                     f"{ch(18, f['lead_time_weeks'])} | {ch(it['risk_level'], f['risk_level'])} | "
                     f"{ch(it['expected_stockout_weeks'], f['expected_stockout_weeks'])} | "
                     f"{ch(it['recommended_latest_po_date'], f['recommended_latest_po_date'])} | "
                     f"{it['recommended_latest_delivery_date']} | {ch(it['suggested_quantity'], f['suggested_quantity'])} |")
        L += ["", "Bold: the figure after the proposed changes (worked out by the pipeline).", "",
              "**History documents** (each applies only to what it names)", ""]
        if not docs:
            L.append("(none)")
        for n, d in enumerate(docs, 1):
            names = ", ".join(d.get("items") or [])
            scope = (f"{d['supplier']}, all items" if d.get("supplier") and not names else
                     f"{names}, {d['supplier']}" if d.get("supplier") else f"{names}, any supplier")
            L.append(f"{n}. *[{scope}]* {d['text']}")
        L += ["", "### Analysis", "", "*Findings*", ""]
        for f in a.get("findings") or []:
            src = "stock figures" if (f.get("basis") or {}).get("stock") else f"documents {(f.get('basis') or {}).get('documents')}"
            L.append(f"- **F{f['id']}** ({src}; about {', '.join(f.get('items') or []) or 'the supplier'}): {f['text']}")
        L += ["", "*Proposed changes*", ""]
        sl = a.get("supplier_lead_time")
        changes = ([f"- Supplier lead time {sl['lead_time_weeks']} weeks: {sl['reason']} (F{sl.get('findings')})"] if sl else []) + \
                  [f"- Lead time of {', '.join(e['items'])} → {e['lead_time_weeks']} weeks: {e['reason']} (F{e.get('findings')})"
                   for e in a.get("lead_time_changes") or []] + \
                  [f"- Quantity of {c['item_code']} {c['original_quantity']} × {c['factor']} → {c['proposed_quantity']}: {c['reason']} (F{c.get('findings')})"
                   for c in a.get("quantity_changes") or []]
        L += changes or ["(none)"]
        L += ["", "*Critical questions*", ""] + ([f"- [{x['ask']}] {x['question']} (F{x.get('findings')})" for x in a.get("critical_questions") or []] or ["(none)"])
        L += ["", "*Recommended actions*", ""] + ([f"- {x['action']} (F{x.get('findings')})" for x in a.get("recommended_actions") or []] or ["(none)"])
        L += ["", f"*Summary:* {a.get('summary', '')}", "", "| Check | What it asks | Verdict | Reason |", "|---|---|---|---|"]
        applicable = q.applicable_judge_checks(items, docs, analysis=a)
        for c in [c for c in ORDER if c in applicable]:
            L.append(f"| {c} | {cell(q.check_text(c, items, False, a, docs))}<br>*{KO[c]}* |  |  |")
        if b.get("approval"):
            ap = b["approval"]
            L += ["", "### Purchase request text (gpt-4o)", "",
                  "*Its input: the items above, the findings above (with their documents), the proposed changes and the actions.*", "",
                  f"**Purpose.** {ap.get('purpose', '')}", "", f"**Justification.** {ap.get('justification', '')}", "",
                  "| Check | What it asks | Verdict | Reason |", "|---|---|---|---|"]
            for c, t in q._P1_TEXT.items():
                L.append(f"| {c} | {cell(t)}<br>*{KO[c]}* |  |  |")
        L.append("")
    SHEET.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {SHEET}")


def collect() -> None:
    rows, sid = [], None
    for line in SHEET.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## (\S+)$", line)
        if m:
            sid = m.group(1)
        m = re.match(r"\|\s*(J\d+[a-c]?|P1[a-c])\s*\|(.*)\|(.*)\|(.*)\|\s*$", line)
        if m and sid:
            verdict = m.group(3).strip().lower()
            if verdict in ("pass", "fail"):
                rows.append({"scenario_id": sid, "check": m.group(1), "verdict": verdict, "reason": m.group(4).strip()})
    with open(SCORES, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} verdicts -> {SCORES}")


if __name__ == "__main__":
    {"sheet": sheet, "collect": collect}[sys.argv[1]]()
