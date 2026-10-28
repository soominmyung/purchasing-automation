"""
The documents the pipeline produces, assembled in code from the analysis agent's fields
(prompt v5; layout and reasons in docs/changelog_2026-10.md, item 16):

- the analysis report, for the buyer: facts first (Findings), then the judgements that cite
  them (Critical Questions, Proposals, Recommended Actions), then the figures per item
- the purchase request, for the approvers: a draft with the open questions and actions in a
  red note, the approval agent's Purpose and Justification, the items table and the
  documents Justification cites
- the supplier email: the email agent's text with the item lines written here

Also the inputs of the approval and email agents, chosen here so that each receives only
what its part of the document needs.

Inline marks (**bold**, *italic*, ==highlight==, [[red]]…[[/red]]) are rendered by
utils/docx_utils.py.
"""
import json
import math
import re
from datetime import date, timedelta
from typing import Any

from config import run_date
from services.item_grouping import (DATA_ISSUES_HEADING, RISK_ORDER, TOTAL_LEAD_WEEKS,
                                    _build_timing_label_from_weeks, risk_level_from_weeks)

REQUESTED_RISKS = ("Critical", "High", "Medium")  # Low is not requested; order dates are months away
APPROVERS = ("Purchasing Manager", "Operations Director", "Finance Officer")
PR_NOTE = ("Draft — to be revised once the questions and actions below are resolved, with current "
           "quantities, dates and figures. Delete this note and everything in red before submitting.")
EMAIL_NOTE = "Supplier email draft — review and edit internally before use, then delete this note."


# ----------------------------------------------------------------------------- small helpers

def _num(v: Any) -> str:
    if v is None or v == "":
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"
    return str(v)


def _risk(level: str) -> str:
    """Critical in bold red in tables and headings (not inside the model's sentences)."""
    return "[[red]]**Critical**[[/red]]" if level == "Critical" else (level or "N/A")


def _changed(original: Any, proposed: Any) -> str:
    """A proposed change: pipeline value → proposed value, highlighted as a whole."""
    return f"=={_num(original)} → {_num(proposed)}=="


def _red(text: str) -> str:
    return f"[[red]]*{text}*[[/red]]"


def _cell(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _codes(entry: dict) -> list[str]:
    return [str(c) for c in _list(entry.get("items"))]


def _risk_of(it: dict) -> str:
    """The item's risk level after any lead-time change (its pipeline level otherwise)."""
    return it.get("risk_new") or it.get("risk_level")


def ordered_items(items: list[dict]) -> list[dict]:
    """Most urgent first: by risk level (after any lead-time change), then by weeks to stockout."""
    def key(it: dict):
        level = _risk_of(it)
        wks = it.get("wks_to_oos")
        return (RISK_ORDER.index(level) if level in RISK_ORDER else len(RISK_ORDER),
                wks if isinstance(wks, (int, float)) else math.inf)
    return sorted(items, key=key)


def requested_items(items: list[dict]) -> list[dict]:
    return [it for it in ordered_items(items) if _risk_of(it) in REQUESTED_RISKS]


# Written into the analysis report's Summary when every item is still Low after the analysis; the
# run ends with the report (the user, 2026-10-25)
NOTHING_TO_REQUEST = ("**No purchase request or supplier email was prepared:** every item is Low risk after this "
                      "analysis, with its latest order date months away.")


def items_to_request(items: list[dict], analysis: dict, snapshot_date: str | None) -> list[dict]:
    """The items due for a purchase request, by their risk level after the analysis's lead-time
    changes (an item Low at 18 weeks may not be at the analysis's lead time)."""
    return requested_items(with_lead_times(items, analysis if isinstance(analysis, dict) else {}, snapshot_date))


def changes_by_item(analysis: dict) -> dict[str, dict[str, dict]]:
    """{item_code: {"qty": change}}; the first quantity change per item counts."""
    out: dict[str, dict[str, dict]] = {}
    for ch in _list(analysis.get("quantity_changes")):
        if isinstance(ch, dict) and ch.get("item_code") is not None:
            out.setdefault(str(ch["item_code"]), {}).setdefault("qty", ch)
    return out


def _proposed_quantity(item: dict, ch: dict) -> Any:
    return ch.get("qty", {}).get("proposed_quantity", item.get("suggested_quantity")) if ch.get("qty") else item.get("suggested_quantity")


# ----------------------------------------------------------------------------- lead times

def lead_time_entries(analysis: dict) -> dict[str, dict]:
    """{item_code: its lead_time_changes entry}; an item's first entry counts (two are a check failure)."""
    out: dict[str, dict] = {}
    for e in _list((analysis or {}).get("lead_time_changes")):
        if isinstance(e, dict):
            for code in _codes(e):
                out.setdefault(code, e)
    return out


def _weeks(v: Any) -> float | None:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


def with_lead_times(items: list[dict], analysis: dict, snapshot_date: str | None) -> list[dict]:
    """Each item with its final lead time and what follows from it, calculated here: the risk
    level (Critical below the lead time, High below lead time + 8, Medium below lead time + 17)
    and the latest order date (latest delivery date minus the lead time). Weeks to stockout, the
    latest delivery date and the suggested quantity do not change. An order date that would fall
    on or before the snapshot date is set to the snapshot date and marked, with the date it should
    have been placed by."""
    entries = lead_time_entries(analysis)
    out = []
    for it in items:
        e = dict(it)
        entry = entries.get(str(it.get("item_code")))
        lead = _weeks((entry or {}).get("lead_time_weeks")) or TOTAL_LEAD_WEEKS
        e["lead_time_weeks"] = lead
        if lead != TOTAL_LEAD_WEEKS:
            e["lead_time_changed"] = True
            new_risk = risk_level_from_weeks(it.get("wks_to_oos"), lead)
            if new_risk != it.get("risk_level") and new_risk != "N/A":
                e["risk_new"] = new_risk
            delivery = it.get("recommended_latest_delivery_date")
            if snapshot_date and delivery and delivery != "N/A":
                new_po = date.fromisoformat(delivery) - timedelta(days=round(lead * 7))
                snap = date.fromisoformat(snapshot_date)
                if new_po <= snap:
                    e["po_new"], e["po_should_have"] = snap.isoformat(), new_po.isoformat()
                    e["po_timing_new"] = "overdue"
                else:
                    e["po_new"] = new_po.isoformat()
                    e["po_timing_new"] = _build_timing_label_from_weeks((new_po - snap).days / 7)
        out.append(e)
    return out


def items_after_changes(items: list[dict], changes: dict, snapshot_date: str | None) -> list[dict]:
    """The items as the second analysis step sees them: the figures after the first step's
    changes, in the same fields as the pipeline's (lead time, risk level, latest order date and
    its timing, the gap, and the proposed quantity where one was proposed)."""
    qty = changes_by_item(changes)
    out = []
    for it in with_lead_times(items, changes, snapshot_date):
        e = {k: v for k, v in it.items()
             if k not in ("risk_new", "po_new", "po_should_have", "po_timing_new", "lead_time_changed")}
        e["risk_level"] = _risk_of(it)
        if it.get("po_new"):
            e["recommended_latest_po_date"] = it["po_new"]
            e["recommended_latest_po_timing"] = it.get("po_timing_new") or it.get("recommended_latest_po_timing")
        wks = it.get("wks_to_oos")
        if isinstance(wks, (int, float)) and not isinstance(wks, bool):
            e["expected_stockout_weeks"] = round(max(0.0, it["lead_time_weeks"] - wks), 1)
        ch = qty.get(str(it.get("item_code")))
        if ch:
            e["suggested_quantity"] = _proposed_quantity(it, ch)
        out.append(e)
    return out


# The asterisk that marks an order date set to the snapshot date. A plain "*" is the italic mark
# of utils/docx_utils.py, so the asterisk operator (U+2217), which looks the same, is used.
OVERDUE_MARK = "∗"


def _risk_cell(it: dict) -> str:
    if it.get("risk_new"):
        return f"=={_risk(it.get('risk_level'))} → {_risk(it['risk_new'])}=="
    return _risk(it.get("risk_level"))


def _lead_cell(it: dict) -> str:
    lead = it.get("lead_time_weeks") or TOTAL_LEAD_WEEKS
    if it.get("lead_time_changed"):
        return f"=={_num(TOTAL_LEAD_WEEKS)} → {_num(lead)} weeks=="
    return f"{_num(lead)} weeks"


def _overdue_notes(items: list[dict]) -> list[str]:
    """In red, below a table: the date an order should have been placed by, when the lead time
    makes it fall on or before the snapshot date."""
    return [_red(f"{OVERDUE_MARK} {it.get('item_name')} should have been ordered by {it['po_should_have']} "
                 f"(lead time {_num(it.get('lead_time_weeks'))} weeks); it is overdue.")
            for it in items if it.get("po_should_have")]


# ----------------------------------------------------------------------------- findings

def number_findings(analysis: dict, items: list[dict], history_docs: list[dict], supplier: str) -> list[dict]:
    """
    The findings as the report shows them, numbered F1, F2, … in two groups: those from the
    documents first, then those from the stock figures; in each, findings about the supplier as a
    whole first, then items most urgent first (by risk after any lead-time change), each item's
    findings in the order written.
    Findings are the facts worth reporting; an item no document names gets no line saying so
    (removed 2026-10-24: it repeated one sentence per item and was not a finding).
    Each entry: {label, id (the model's id), items, text, documents, stock, code}.
    """
    order = {str(it.get("item_code")): n for n, it in enumerate(ordered_items(with_lead_times(items, analysis, None)))}
    names = {str(it.get("item_code")): str(it.get("item_name")) for it in items}
    entries = []
    for n, f in enumerate(_list(analysis.get("findings"))):
        if not isinstance(f, dict):
            continue
        basis = f.get("basis") if isinstance(f.get("basis"), dict) else {}
        codes = [c for c in _codes(f) if c in names]
        entries.append({"id": f.get("id"), "items": codes, "text": str(f.get("text") or "").strip(),
                        "documents": [d for d in _list(basis.get("documents")) if isinstance(d, int)],
                        "stock": bool(basis.get("stock")), "code": False,
                        "_group": min((order[c] for c in codes), default=-1), "_n": n})
    entries.sort(key=lambda e: (e["stock"], e["_group"], e["_n"]))
    for k, e in enumerate(entries, 1):
        e["label"] = f"F{k}"
        subject = ", ".join(names[c] for c in e["items"]) if e["items"] else supplier
        text = e["text"]
        if text.lower().startswith(subject.lower() + " "):
            text = text[len(subject) + 1:]
        source = "".join(f"[{d}]" for d in e["documents"])  # the group heading says when it is the stock figures
        text = _highlight_changes(text, analysis)
        e["subject"], e["body"] = subject, text
        e["line"] = f"{subject} — {text}" + (f" {source}" if source else "")
        del e["_group"], e["_n"]
    return entries


def _findings_lines(numbered: list[dict], analysis: dict) -> list[str]:
    """The report's Findings: the facts from the documents, then those from the stock figures (after
    the proposed changes, when there are any), each under its own heading; an empty group is left out."""
    changed = bool(_list(analysis.get("lead_time_changes")) or _list(analysis.get("quantity_changes")))
    groups = (("From the documents", [e for e in numbered if not e["stock"]]),
              ("From the stock figures" + (" (after the proposed changes)" if changed else ""),
               [e for e in numbered if e["stock"]]))
    out = []
    for title, entries in groups:
        if entries:
            out += [f"### {title}", ""] + [f"{e['label']}. {e['line']}" for e in entries] + [""]
    return out or ["No findings.", ""]


def _labels(numbered: list[dict]) -> dict[Any, str]:
    return {e["id"]: e["label"] for e in numbered if e["id"] is not None}


def _highlight_changes(text: str, analysis: dict) -> str:
    """Highlight each "pipeline value → proposed value" in the model's text that is one of its
    proposed changes, as the tables do (an arrow that is not a change is left as written; the
    quality check C9 reports it)."""
    from services.quality_checks import ARROW, _figure, change_pairs
    pairs = change_pairs(analysis)
    def mark(m: re.Match) -> str:
        if (_figure(m.group(1)), _figure(m.group(2))) not in pairs:
            return m.group(0)
        start, end = m.start(), m.end()
        if text[max(0, start - 2):start] == "==" and text[end:end + 2] == "==":
            return m.group(0)
        return f"=={m.group(1)} → {m.group(2)}=="  # one arrow form, as in the tables
    return ARROW.sub(mark, text or "")


def _cites(entry: dict, labels: dict[Any, str]) -> str:
    cited = [labels.get(i, f"F? (id {i})") for i in _list(entry.get("findings"))]
    return f" ({', '.join(cited)})" if cited else ""


# ----------------------------------------------------------------------------- report sections

def _questions_lines(analysis: dict, supplier: str, labels: dict, keep=None, cite: bool = True) -> list[str]:
    qs = [q for q in _list(analysis.get("critical_questions"))
          if isinstance(q, dict) and str(q.get("question") or "").strip() and (keep is None or keep(q))]
    lines, n = [], 0
    for ask, title in (("supplier", f"Ask {supplier}:"), ("internal", "Check internally:")):
        group = [q for q in qs if (q.get("ask") == "supplier") == (ask == "supplier")]
        if not group:
            continue
        lines.append(title)
        for q in group:
            n += 1
            lines.append(f"{n}. {_highlight_changes(q['question'].strip(), analysis)}" + (_cites(q, labels) if cite else ""))
    return lines


def _actions_lines(analysis: dict, labels: dict, keep=None, cite: bool = True) -> list[str]:
    acts = [a for a in _list(analysis.get("recommended_actions"))
            if isinstance(a, dict) and str(a.get("action") or "").strip() and (keep is None or keep(a))]
    return [f"{n}. {_highlight_changes(a['action'].strip(), analysis)}" + (_cites(a, labels) if cite else "")
            for n, a in enumerate(acts, 1)]


def _proposal_lines(items: list[dict], analysis: dict, labels: dict, supplier: str = "") -> list[str]:
    """Lead times (the supplier's expected one, then each item's final one) and quantity changes."""
    lines, names = [], {str(it.get("item_code")): it.get("item_name") for it in items}

    def add(text: str) -> None:
        lines.append(f"{len(lines) + 1}. {text}")

    sl = analysis.get("supplier_lead_time")
    if isinstance(sl, dict) and _weeks(sl.get("lead_time_weeks")):
        add(f"Lead time, {supplier} overall (expected): =={_num(TOTAL_LEAD_WEEKS)} → {_num(sl['lead_time_weeks'])} weeks==: "
            f"{_highlight_changes(str(sl.get('reason') or '').strip(), analysis)}{_cites(sl, labels)}")
    for e in _list(analysis.get("lead_time_changes")):
        if not isinstance(e, dict) or not _weeks(e.get("lead_time_weeks")):
            continue
        who = ", ".join(str(names.get(c, c)) for c in _codes(e))
        weeks = e["lead_time_weeks"]
        shown = (f"{_num(weeks)} weeks (unchanged)" if weeks == TOTAL_LEAD_WEEKS
                 else f"=={_num(TOTAL_LEAD_WEEKS)} → {_num(weeks)} weeks==")
        add(f"Lead time, {who}: {shown}: {_highlight_changes(str(e.get('reason') or '').strip(), analysis)}{_cites(e, labels)}")
    for it in ordered_items(items):
        ch = changes_by_item(analysis).get(str(it.get("item_code")), {})
        if ch.get("qty"):
            c = ch["qty"]
            factor = c.get("factor")
            times = f" (×{factor:g})" if isinstance(factor, (int, float)) else ""
            add(f"{it.get('item_name')} quantity {_changed(c.get('original_quantity'), c.get('proposed_quantity'))}"
                f"{times}: {_highlight_changes(str(c.get('reason') or '').strip(), analysis)}{_cites(c, labels)}")
    return lines


def _order_by(it: dict, ch: dict | None = None) -> str:
    """The latest order date: after a lead-time change "old → new", highlighted; a date on or
    before the snapshot date shown as the snapshot date with an asterisk (see _overdue_notes)."""
    old = it.get("recommended_latest_po_date") or "N/A"
    if it.get("po_new"):
        star = OVERDUE_MARK if it.get("po_should_have") else ""
        if old == it["po_new"]:
            return f"=={old}{star}=="
        return f"=={old} → {it['po_new']}{star}=="
    return f"{old} (overdue)" if it.get("recommended_latest_po_timing") == "overdue" else old


def _quantity(it: dict, ch: dict) -> str:
    if ch.get("qty"):
        return _changed(ch["qty"].get("original_quantity"), ch["qty"].get("proposed_quantity"))
    return _num(it.get("suggested_quantity"))


def _doc_about(doc: dict) -> str:
    items = ", ".join(doc.get("items") or doc.get("item_codes") or [])
    if doc.get("supplier"):
        return f"{doc['supplier']} — {items}" if items else f"{doc['supplier']} (supplier-wide)"
    return f"{items} (any supplier)" if items else "-"


def _docs_table(rows: list[tuple[int, dict]]) -> list[str]:
    lines = ["| # | File | Date | About |", "| --- | --- | --- | --- |"]
    for n, d in rows:
        lines.append(f"| {n} | {_cell(d.get('source') or '(file name not recorded)')} | "
                     f"{d.get('date') or '-'} | {_cell(_doc_about(d))} |")
    return lines


CHECK_NOTES = "<!-- check notes -->"  # replaced by fill_check_notes once every check has run


# For the buyer, in plain words and without check ids: what a failed code check means. The
# check's own detail (items and figures) follows.
_PLAIN_CODE = {
    "C1": "The analysis could not be read in full, so parts of this draft may be missing",
    "C2": "An item that is not part of this order is named",
    "C3": "A proposed change does not start from the calculated figure",
    "C4": "A statement rests on a source that does not support it",
    "C5": "A proposed change is incomplete or does not add up",
    "C6": "No action is given for an item that will run out before a new order can arrive",
    "C7": "The summary does not name every item that will run out before a new order can arrive",
    "C8": "The summary is longer than intended",
    "C9": "A changed figure in the text does not match the proposed change",
    "C10": "A figure worked out after a lead-time change is wrong",
    "P2": "Purpose or Justification is longer than intended",
    "P3": "Justification cites a document that was not given",
}
# What a judge check looks at, for when the judge could not answer.
_PLAIN_AREA = {
    "J1a": "What the analysis says about past documents",
    "J1b": "Whether the analysis reports past documents as they are",
    "J1c": "Whether past documents are applied to the right supplier and items",
    "J2a": "Whether the supplier's problems are raised as questions",
    "J2b": "Whether the items' problems are raised as questions",
    "J2c": "Whether a question assumes a problem the documents do not show",
    "J6a": "Whether each proposed change goes the way its reasons point",
    "J6b": "Whether each proposed change is the right size",
    "J6c": "Whether the same change was left out for an item it also applies to",
    "J7a": "The stock figures quoted in the text",
    "J8": "Whether the questions for the supplier reveal anything internal",
    "J9": "Whether each question and action follows from its reasons",
    "J10": "Whether a document with a problem for this order was left unused",
    "J11": "Whether the findings state facts only",
    "P1a": "Purpose and Justification",
    "P1b": "The figures in Purpose and Justification",
    "P1c": "The wording of Purpose and Justification",
}


def _plain(cid: str, v: dict) -> str:
    if v.get("pass") is None:
        return f"{_PLAIN_AREA.get(cid, 'One part')} could not be checked automatically; please review it."
    if cid in _PLAIN_CODE:
        return f"{_PLAIN_CODE[cid]}: {v.get('detail') or ''}".rstrip(": ")
    return v.get("note") or f"{_PLAIN_AREA.get(cid, 'One part')} may be wrong; please review it."


def check_note_lines(results: list[dict]) -> list[str]:
    """In red, for the draft, in plain words: what must be corrected (blocking checks not passed
    after the analysis rewrites) and what to check (warnings). Each result is {"checks": {id:
    {pass, detail, severity, note}}}; check ids stay in the quality-check document."""
    from services.quality_checks import BLOCKING
    blocking, warnings = [], []
    for result in results:
        for cid, v in ((result or {}).get("checks") or {}).items():
            if v.get("pass") is True:
                continue
            (blocking if v.get("severity") == BLOCKING else warnings).append(f"- {_plain(cid, v)}")
    out = []
    if blocking:
        out += [_red("**Must be corrected before use**")] + [_red(x) for x in blocking] + [""]
    if warnings:
        out += [_red("**Please check**")] + [_red(x) for x in warnings] + [""]
    return out


def fill_check_notes(markdown: str, results: list[dict]) -> str:
    lines = check_note_lines(results)
    return (markdown or "").replace(CHECK_NOTES + "\n", "\n".join(lines) + ("\n" if lines else ""))


def _date_header(snapshot_date: str) -> list[str]:
    today = run_date()
    age = (today - date.fromisoformat(snapshot_date)).days if snapshot_date else 0
    line = f"Snapshot date: {snapshot_date}"
    if age > 0:
        line += f" ({age} day{'s' if age != 1 else ''} before this report)"
    return [line, "", f"Report date: {today.isoformat()}"]


# ----------------------------------------------------------------------------- analysis report

def render_analysis_report(supplier: str, snapshot_date: str, items: list[dict], analysis: dict,
                           history_docs: list[dict], window: dict | None,
                           data_issue_lines: list[str], run_id: str) -> str:
    analysis = analysis if isinstance(analysis, dict) else {}
    numbered = number_findings(analysis, items, history_docs, supplier)
    labels = _labels(numbered)
    changes = changes_by_item(analysis)
    items = with_lead_times(items, analysis, snapshot_date)
    # CHECK_NOTES: the analysis checks not passed, filled in by agents.finish_node (empty when all
    # passed), so a buyer sees at the top whether the conclusion needs checking (the user, 2026-10-28)
    out = [f"# Purchasing Analysis — {supplier}", "", *_date_header(snapshot_date), "", CHECK_NOTES]
    if data_issue_lines:
        out += [DATA_ISSUES_HEADING, *[f"- {ln}" for ln in data_issue_lines], ""]
    out += ["## Summary", ""] + ([NOTHING_TO_REQUEST, ""] if not requested_items(items) else []) + [
        _highlight_changes(str(analysis.get("summary") or "(no summary was returned)").strip(), analysis), ""]

    out += ["## Items", "", "| Item | Risk | Weeks to stockout | Lead time | Order by | Quantity |",
            "| --- | --- | --- | --- | --- | --- |"]
    for it in ordered_items(items):
        ch = changes.get(str(it.get("item_code")), {})
        out.append(f"| {_cell(it.get('item_name'))} ({it.get('item_code')}) | {_risk_cell(it)} | "
                   f"{_num(it.get('wks_to_oos'))} | {_lead_cell(it)} | {_order_by(it)} | {_quantity(it, ch)} |")
    out += [""] + _overdue_notes(ordered_items(items)) + ([""] if _overdue_notes(items) else [])

    out += ["## Findings", ""] + _findings_lines(numbered, analysis)
    out += ["## Critical Questions", ""] + (_questions_lines(analysis, supplier, labels)
                                            or ["No open questions for this order."]) + [""]
    out += ["## Proposals", ""] + (_proposal_lines(items, analysis, labels, supplier) or ["No changes proposed."]) + [""]
    out += ["## Recommended Actions", ""] + (_actions_lines(analysis, labels)
                                             or ["No action needed beyond ordering by the recommended dates."]) + [""]

    out += ["## Item Details", ""]
    for it in ordered_items(items):
        ch = changes.get(str(it.get("item_code")), {})
        out += [f"### {it.get('item_name')} ({it.get('item_code')}) — {_risk_cell(it)}", "",
                "| Measure | Value |", "| --- | --- |",
                f"| Available stock | {_num(it.get('available_stock'))} |",
                f"| Weeks to stockout | {_num(it.get('wks_to_oos'))} |",
                f"| Lead time | {_lead_cell(it)} |",
                f"| Order by | {_order_by(it)} |",
                f"| Deliver by | {it.get('recommended_latest_delivery_date') or 'N/A'} |",
                f"| Suggested quantity | {_quantity(it, ch)} |", ""] + _overdue_notes([it]) + (
                [""] if it.get("po_should_have") else [])

    out += ["## Referenced Documents", ""]
    if window:
        rule = (f"History searched: documents dated {window['from']} to {window['to']} "
                f"(last {window['window_months']} months), at most {window['per_scope']} per supplier and per item.")
        if window.get("left_out"):
            rule += (f" {window['in_window']} documents fell in that period; the {window['left_out']} oldest "
                     "beyond that limit are not included.")
        out += [rule, ""]
    out += (_docs_table(list(enumerate(history_docs, 1))) if history_docs
            else ["No history was found for this supplier or these items."]) + [""]
    out += [f"Run id: {run_id}"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- approval agent

def _concerns(entry_codes: list[str], codes: set[str]) -> bool:
    """An entry about the supplier as a whole (no items) or about one of these items."""
    return not entry_codes or bool(set(entry_codes) & codes)


def approval_input(supplier: str, items: list[dict], analysis: dict, history_docs: list[dict],
                   snapshot_date: str | None = None) -> dict:
    """What the approval agent receives: the requested items with their figures, and the
    findings, changes and actions about them or about the supplier as a whole. Findings carry
    the numbers of their documents, which Justification cites. Figures are after any lead-time
    change."""
    numbered = number_findings(analysis, items, history_docs, supplier)
    items = with_lead_times(items, analysis, snapshot_date)
    req = requested_items(items)
    codes = {str(it.get("item_code")) for it in req}
    findings = [{"about": e["subject"], "text": e["body"], "documents": e["documents"]}
                for e in numbered if not e["code"] and _concerns(e["items"], codes)]
    changes = []
    entries = lead_time_entries(analysis)
    for it in req:
        ch = changes_by_item(analysis).get(str(it.get("item_code")), {})
        if it.get("lead_time_changed"):
            changes.append({"item": it.get("item_name"), "what": "lead time (weeks)",
                            "from": TOTAL_LEAD_WEEKS, "to": it.get("lead_time_weeks"),
                            "reason": entries.get(str(it.get("item_code")), {}).get("reason"),
                            "order_by": {"from": it.get("recommended_latest_po_date"), "to": it.get("po_new")}})
        if ch.get("qty"):
            changes.append({"item": it.get("item_name"), "what": "quantity",
                            "from": ch["qty"].get("original_quantity"), "to": ch["qty"].get("proposed_quantity"),
                            "reason": ch["qty"].get("reason")})
    actions = [{"action": a.get("action"), "items": _codes(a)} for a in _list(analysis.get("recommended_actions"))
               if isinstance(a, dict) and _concerns(_codes(a), codes)]
    return {
        "supplier": supplier,
        "items": [{"item_name": it.get("item_name"), "item_code": it.get("item_code"),
                   "risk_level": _risk_of(it), "weeks_to_stockout": it.get("wks_to_oos"),
                   "lead_time_weeks": it.get("lead_time_weeks"),
                   "expected_stockout_weeks": max(0, it.get("lead_time_weeks") - it["wks_to_oos"])
                   if isinstance(it.get("wks_to_oos"), (int, float)) else it.get("expected_stockout_weeks"),
                   "suggested_quantity": it.get("suggested_quantity"),
                   "order_by": it.get("po_new") or it.get("recommended_latest_po_date"),
                   "order_by_timing": it.get("po_timing_new") or it.get("recommended_latest_po_timing"),
                   "deliver_by": it.get("recommended_latest_delivery_date")} for it in req],
        "findings": findings,
        "changes": changes,
        "actions": actions,
    }


_DOC_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def renumber_citations(texts: list[str], n_docs: int) -> tuple[list[str], list[int], list[str]]:
    """Document numbers in the approval text renumbered 1, 2, … in order of first citation.
    Returns (texts, document ids in their new order, problems)."""
    order: list[int] = []
    problems: list[str] = []

    def sub(m: re.Match) -> str:
        new = []
        for raw in re.split(r"\s*,\s*", m.group(1)):
            d = int(raw)
            if not 1 <= d <= n_docs:
                problems.append(f"cited document [{d}] was not given")
                new.append("?")
                continue
            if d not in order:
                order.append(d)
            new.append(str(order.index(d) + 1))
        return "[" + ", ".join(new) + "]"

    return [_DOC_CITE.sub(sub, t or "") for t in texts], order, problems


def render_purchase_request(supplier: str, snapshot_date: str, items: list[dict], analysis: dict,
                            history_docs: list[dict], approval: dict | None, run_id: str) -> tuple[str, list[str]]:
    """The purchase request draft. Returns (markdown, problems found in the approval text)."""
    analysis = analysis if isinstance(analysis, dict) else {}
    labels = _labels(number_findings(analysis, items, history_docs, supplier))
    items = with_lead_times(items, analysis, snapshot_date)
    req = requested_items(items)
    codes = {str(it.get("item_code")) for it in req}
    keep = lambda e: _concerns(_codes(e), codes)
    out = [_red(PR_NOTE), "", CHECK_NOTES]
    questions = _questions_lines(analysis, supplier, labels, keep=keep, cite=False)
    actions = _actions_lines(analysis, labels, keep=keep, cite=False)
    if questions:
        out += [_red("**Critical Questions**")] + [_red(q) for q in questions] + [""]
    if actions:
        out += [_red("**Recommended Actions**")] + [_red(a) for a in actions] + [""]
    out += [f"# Purchase Request — {supplier}", "", f"Request date: {run_date().isoformat()}", "",
            "Requested by: *[Your name here]*", "", f"Analysis reference: run {run_id}", ""]
    if not req:
        out += ["No item is due for a purchase request: every item is Low risk, with order dates months away."]
        return "\n".join(out) + "\n", []

    approval = approval if isinstance(approval, dict) else {}
    (purpose, justification), cited, problems = renumber_citations(
        [str(approval.get("purpose") or ""), str(approval.get("justification") or "")], len(history_docs))
    out += ["## Purpose", "", purpose.strip() or "(not written)", "",
            "## Justification", "", justification.strip() or "(not written)", ""]
    changes = changes_by_item(analysis)
    out += [f"## Items Requested (snapshot {snapshot_date})", "",
            "| Item | Code | Risk | Weeks to stockout | Lead time | Quantity | Order by | Deliver by |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for it in req:
        ch = changes.get(str(it.get("item_code")), {})
        out.append(f"| {_cell(it.get('item_name'))} | {it.get('item_code')} | {_risk_cell(it)} | "
                   f"{_num(it.get('wks_to_oos'))} | {_lead_cell(it)} | {_quantity(it, ch)} | {_order_by(it)} | "
                   f"{it.get('recommended_latest_delivery_date') or 'N/A'} |")
    out += [""] + _overdue_notes(req) + ([""] if _overdue_notes(req) else [])
    out += ["Unit price and total: to be confirmed with the supplier.", ""]
    if cited:
        out += ["## Supporting Documents", ""] + _docs_table([(n, history_docs[d - 1]) for n, d in enumerate(cited, 1)]) + [""]
    out += ["## Approval", "", "| Role | Name | Signature | Date |", "| --- | --- | --- | --- |"]
    out += [f"| {role} |  |  |  |" for role in APPROVERS]
    return "\n".join(out) + "\n", problems


# ----------------------------------------------------------------------------- supplier email

def email_input(snapshot_date: str, supplier: str, items: list[dict], analysis: dict) -> dict:
    """What the email agent receives: the items being enquired about (name, quantity after any
    proposed change, requested delivery date), the questions for the supplier, and the
    Critical items for which it also asks the earliest dispatch date. No stock figures, risk
    levels, findings or internal questions."""
    items = with_lead_times(items, analysis, snapshot_date)
    req = requested_items(items)
    changes = changes_by_item(analysis)
    return {  # no snapshot date: when the company reviewed its stock is internal
        "supplier": supplier,
        "items": [{"item_name": it.get("item_name"),
                   "quantity": _proposed_quantity(it, changes.get(str(it.get("item_code")), {})),
                   "requested_delivery_date": it.get("recommended_latest_delivery_date")} for it in req],
        "points_to_clarify": [str(q.get("question")).strip() for q in _list(analysis.get("critical_questions"))
                               if isinstance(q, dict) and q.get("ask") == "supplier" and q.get("question")],
        "ask_earliest_dispatch": [it.get("item_name") for it in req if _risk_of(it) == "Critical"],
        # named so the agent can leave them out where a point mentions them
        "other_items": [it.get("item_name") for it in ordered_items(items) if it not in req],
    }


def email_other_items(snapshot_date: str | None, items: list[dict], analysis: dict) -> list[dict]:
    """The supplier's items that are not part of the email (as email_input's other_items), with
    their codes, for the validator's check that none of them is named."""
    items = with_lead_times(items, analysis, snapshot_date)
    req = requested_items(items)
    return [{"item_name": str(it.get("item_name") or ""), "item_code": str(it.get("item_code") or "")}
            for it in ordered_items(items) if it not in req]


def items_named(text: str, refs: list[dict]) -> list[str]:
    """The items of refs that text names, by name or by code. A name matches whatever its case and
    whether its parts are joined, spaced or hyphenated ("PlasticCap", "plastic cap", "Plastic-Cap"),
    as a whole word only."""
    found = []
    for r in refs:
        name, code = r.get("item_name") or "", r.get("item_code") or ""
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name) or ([name] if name else [])
        pats = []
        if parts:
            pats.append(r"\b" + r"[\s\-_]?".join(re.escape(p) for p in parts) + r"(?:e?s)?\b")  # plural too
        if code:
            pats.append(r"\b" + re.escape(code) + r"\b")
        if any(re.search(p, text or "", re.I) for p in pats):
            found.append(name or code)
    return found


def email_item_lines(items: list[dict], analysis: dict) -> list[str]:
    changes = changes_by_item(analysis)
    items = with_lead_times(items, analysis, None)  # which items are requested; dates do not change
    lines = []
    for it in requested_items(items):
        ch = changes.get(str(it.get("item_code")), {})
        qty = _num(_proposed_quantity(it, ch))
        if ch.get("qty"):
            qty = f"=={qty}=="
        lines.append(f"- {it.get('item_name')} — purchase quantity: {qty}; "
                     f"requested delivery date: {it.get('recommended_latest_delivery_date') or 'N/A'}")
    return lines


# The email's fixed parts, written in code: the agent writes only the requests (2026-10-23).
EMAIL_PURPOSE = ("We are planning a possible order of the following items and would be grateful for your "
                 "input before we issue a formal purchase order.")
EMAIL_CLOSING = "Thank you in advance; we look forward to your reply."
SIGNATURE = ("Best regards,", "Company K Purchasing Team")
EMAIL_PARTS = ("requests",)


def email_parts(email_text: str) -> tuple[dict[str, str], list[str]]:
    """The email agent's part (requests) from its JSON answer. Returns (parts, problems); text
    that is not that JSON is kept whole as the requests."""
    raw = (email_text or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    try:
        data = json.loads(m.group(0), strict=False) if m else None  # line breaks inside the string too
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return {"requests": raw}, ["the email agent did not return its JSON; its text was kept as it is"]
    parts = {k: str(data.get(k) or "").strip() for k in EMAIL_PARTS}
    return parts, [f"the email agent left {k} empty" for k in EMAIL_PARTS if not parts[k]]


def email_for_check(email_text: str) -> str:
    """What the validator reads: the agent's own text, without the item lines."""
    parts, _ = email_parts(email_text)
    return "\n\n".join(parts[k] for k in EMAIL_PARTS if parts[k])


def render_email(email_text: str, items: list[dict], analysis: dict, supplier: str,
                 check_failure: str = "", drafts: int = 0) -> tuple[str, list[str]]:
    """The email draft, assembled in a fixed order: the draft note, the greeting, the purpose,
    the item lines, the agent's requests, the closing and the signature. Only the requests come
    from the agent. check_failure: the validator's reason when the last draft still failed it
    (after the last rewrite), shown in the draft note for the buyer. Returns (markdown, problems)."""
    parts, problems = email_parts(email_text)
    note = _red(EMAIL_NOTE)
    if check_failure:
        note += "  \n" + _red(f"The email check still failed after {drafts} drafts; check before sending: {check_failure}")
    blocks = [note, f"Dear {supplier} Team,", EMAIL_PURPOSE, "\n".join(email_item_lines(items, analysis)),
              parts["requests"], EMAIL_CLOSING, "\n".join(SIGNATURE)]
    return "\n\n".join(b for b in blocks if b) + "\n", problems
