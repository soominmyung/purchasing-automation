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
import math
import re
from datetime import date, timedelta
from typing import Any

from config import run_date
from services.item_grouping import DATA_ISSUES_HEADING, RISK_ORDER, TOTAL_LEAD_WEEKS
from services.vector_store import history_key

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


def ordered_items(items: list[dict]) -> list[dict]:
    """Most urgent first: by risk level, then by weeks to stockout."""
    def key(it: dict):
        level = it.get("risk_level")
        wks = it.get("wks_to_oos")
        return (RISK_ORDER.index(level) if level in RISK_ORDER else len(RISK_ORDER),
                wks if isinstance(wks, (int, float)) else math.inf)
    return sorted(items, key=key)


def requested_items(items: list[dict]) -> list[dict]:
    return [it for it in ordered_items(items) if it.get("risk_level") in REQUESTED_RISKS]


def changes_by_item(analysis: dict) -> dict[str, dict[str, dict]]:
    """{item_code: {"po": change, "qty": change}}; the first change per item and kind counts."""
    out: dict[str, dict[str, dict]] = {}
    for kind, field in (("po", "po_date_changes"), ("qty", "quantity_changes")):
        for ch in _list(analysis.get(field)):
            if isinstance(ch, dict) and ch.get("item_code") is not None:
                out.setdefault(str(ch["item_code"]), {}).setdefault(kind, ch)
    return out


def _proposed_quantity(item: dict, ch: dict) -> Any:
    return ch.get("qty", {}).get("proposed_quantity", item.get("suggested_quantity")) if ch.get("qty") else item.get("suggested_quantity")


# ----------------------------------------------------------------------------- findings

def _doc_names_item(doc: dict, item: dict) -> bool:
    keys = {history_key(v) for v in (doc.get("items") or []) + (doc.get("item_codes") or [])}
    return bool(keys & {history_key(item.get("item_code")), history_key(item.get("item_name"))})


def number_findings(analysis: dict, items: list[dict], history_docs: list[dict], supplier: str) -> list[dict]:
    """
    The findings as the report shows them, numbered F1, F2, …: findings about the supplier as
    a whole first, then items most urgent first, each item's findings in the order written.
    Code adds "<item> — no documents on this item." for an item no document names.
    Each entry: {label, id (the model's id, or None), items, text, documents, stock, code}.
    """
    order = {str(it.get("item_code")): n for n, it in enumerate(ordered_items(items))}
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
    for it in items:
        if not any(_doc_names_item(d, it) for d in history_docs):
            code = str(it.get("item_code"))
            entries.append({"id": None, "items": [code], "text": "no documents on this item.",
                            "documents": [], "stock": False, "code": True,
                            "_group": order[code], "_n": 10_000})
    entries.sort(key=lambda e: (e["_group"], e["_n"]))
    for k, e in enumerate(entries, 1):
        e["label"] = f"F{k}"
        subject = ", ".join(names[c] for c in e["items"]) if e["items"] else supplier
        text = e["text"]
        if text.lower().startswith(subject.lower() + " "):
            text = text[len(subject) + 1:]
        source = "".join(f"[{d}]" for d in e["documents"])
        if e["stock"]:
            source = (source + " " if source else "") + "(stock position)"
        e["subject"], e["body"] = subject, text
        e["line"] = f"{subject} — {text}" + (f" {source}" if source else "")
        del e["_group"], e["_n"]
    return entries


def _labels(numbered: list[dict]) -> dict[Any, str]:
    return {e["id"]: e["label"] for e in numbered if e["id"] is not None}


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
            lines.append(f"{n}. {q['question'].strip()}" + (_cites(q, labels) if cite else ""))
    return lines


def _actions_lines(analysis: dict, labels: dict, keep=None, cite: bool = True) -> list[str]:
    acts = [a for a in _list(analysis.get("recommended_actions"))
            if isinstance(a, dict) and str(a.get("action") or "").strip() and (keep is None or keep(a))]
    return [f"{n}. {a['action'].strip()}" + (_cites(a, labels) if cite else "") for n, a in enumerate(acts, 1)]


def _proposal_lines(items: list[dict], analysis: dict, labels: dict) -> list[str]:
    lines, n = [], 0
    for it in ordered_items(items):
        ch = changes_by_item(analysis).get(str(it.get("item_code")), {})
        name = it.get("item_name")
        if ch.get("po"):
            c = ch["po"]
            shift = c.get("shift_weeks")
            way = ("earlier" if shift < 0 else "later") if isinstance(shift, (int, float)) else ""
            weeks = f"{abs(shift):g} week{'s' if abs(shift) != 1 else ''} {way}" if way else "shift not given"
            n += 1
            lines.append(f"{n}. {name} order date {_changed(c.get('original_po_date'), c.get('proposed_po_date'))} "
                         f"({weeks}): {str(c.get('reason') or '').strip()}{_cites(c, labels)}")
        if ch.get("qty"):
            c = ch["qty"]
            factor = c.get("factor")
            times = f" (×{factor:g})" if isinstance(factor, (int, float)) else ""
            n += 1
            lines.append(f"{n}. {name} quantity {_changed(c.get('original_quantity'), c.get('proposed_quantity'))}"
                         f"{times}: {str(c.get('reason') or '').strip()}{_cites(c, labels)}")
    return lines


def _order_by(it: dict, ch: dict) -> str:
    if ch.get("po"):
        return _changed(ch["po"].get("original_po_date"), ch["po"].get("proposed_po_date"))
    value = it.get("recommended_latest_po_date") or "N/A"
    return f"{value} (overdue)" if it.get("recommended_latest_po_timing") == "overdue" else value


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
    out = [f"# Purchasing Analysis — {supplier}", "", *_date_header(snapshot_date), ""]
    if data_issue_lines:
        out += [DATA_ISSUES_HEADING, *[f"- {ln}" for ln in data_issue_lines], ""]
    out += ["## Summary", "", str(analysis.get("summary") or "(no summary was returned)").strip(), ""]

    out += ["## Items", "", "| Item | Risk | Weeks to stockout | Order by | Quantity |", "| --- | --- | --- | --- | --- |"]
    for it in ordered_items(items):
        ch = changes.get(str(it.get("item_code")), {})
        out.append(f"| {_cell(it.get('item_name'))} ({it.get('item_code')}) | {_risk(it.get('risk_level'))} | "
                   f"{_num(it.get('wks_to_oos'))} | {_order_by(it, ch)} | {_quantity(it, ch)} |")
    out.append("")

    out += ["## Findings", ""] + [f"{e['label']}. {e['line']}" for e in numbered] + [""]
    out += ["## Critical Questions", ""] + (_questions_lines(analysis, supplier, labels)
                                            or ["No open questions for this order."]) + [""]
    out += ["## Proposals", ""] + (_proposal_lines(items, analysis, labels) or ["No changes proposed."]) + [""]
    out += ["## Recommended Actions", ""] + (_actions_lines(analysis, labels)
                                             or ["No action needed beyond ordering by the recommended dates."]) + [""]

    out += ["## Item Details", ""]
    for it in ordered_items(items):
        ch = changes.get(str(it.get("item_code")), {})
        out += [f"### {it.get('item_name')} ({it.get('item_code')}) — {_risk(it.get('risk_level'))}", "",
                "| Measure | Value |", "| --- | --- |",
                f"| Available stock | {_num(it.get('available_stock'))} |",
                f"| Weeks to stockout | {_num(it.get('wks_to_oos'))} |",
                f"| Lead time | {TOTAL_LEAD_WEEKS} weeks |",
                f"| Order by | {_order_by(it, ch)} |",
                f"| Deliver by | {it.get('recommended_latest_delivery_date') or 'N/A'} |",
                f"| Suggested quantity | {_quantity(it, ch)} |", ""]

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


def approval_input(supplier: str, items: list[dict], analysis: dict, history_docs: list[dict]) -> dict:
    """What the approval agent receives: the requested items with their figures, and the
    findings, changes and actions about them or about the supplier as a whole. Findings carry
    the numbers of their documents, which Justification cites."""
    req = requested_items(items)
    codes = {str(it.get("item_code")) for it in req}
    numbered = number_findings(analysis, items, history_docs, supplier)
    findings = [{"about": e["subject"], "text": e["body"], "documents": e["documents"]}
                for e in numbered if not e["code"] and _concerns(e["items"], codes)]
    changes = []
    for it in req:
        ch = changes_by_item(analysis).get(str(it.get("item_code")), {})
        if ch.get("po"):
            changes.append({"item": it.get("item_name"), "what": "order date",
                            "from": ch["po"].get("original_po_date"), "to": ch["po"].get("proposed_po_date"),
                            "reason": ch["po"].get("reason")})
        if ch.get("qty"):
            changes.append({"item": it.get("item_name"), "what": "quantity",
                            "from": ch["qty"].get("original_quantity"), "to": ch["qty"].get("proposed_quantity"),
                            "reason": ch["qty"].get("reason")})
    actions = [{"action": a.get("action"), "items": _codes(a)} for a in _list(analysis.get("recommended_actions"))
               if isinstance(a, dict) and _concerns(_codes(a), codes)]
    return {
        "supplier": supplier,
        "items": [{"item_name": it.get("item_name"), "item_code": it.get("item_code"),
                   "risk_level": it.get("risk_level"), "weeks_to_stockout": it.get("wks_to_oos"),
                   "expected_stockout_weeks": it.get("expected_stockout_weeks"),
                   "suggested_quantity": it.get("suggested_quantity"),
                   "order_by": it.get("recommended_latest_po_date"),
                   "order_by_timing": it.get("recommended_latest_po_timing"),  # e.g. "overdue"
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
    req = requested_items(items)
    codes = {str(it.get("item_code")) for it in req}
    labels = _labels(number_findings(analysis, items, history_docs, supplier))
    keep = lambda e: _concerns(_codes(e), codes)
    out = [_red(PR_NOTE), ""]
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
            "| Item | Code | Risk | Weeks to stockout | Quantity | Order by | Deliver by |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for it in req:
        ch = changes.get(str(it.get("item_code")), {})
        out.append(f"| {_cell(it.get('item_name'))} | {it.get('item_code')} | {_risk(it.get('risk_level'))} | "
                   f"{_num(it.get('wks_to_oos'))} | {_quantity(it, ch)} | {_order_by(it, ch)} | "
                   f"{it.get('recommended_latest_delivery_date') or 'N/A'} |")
    out += ["", "Unit price and total: to be confirmed with the supplier.", ""]
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
    req = requested_items(items)
    changes = changes_by_item(analysis)
    return {  # no snapshot date: when the company reviewed its stock is internal
        "supplier": supplier,
        "items": [{"item_name": it.get("item_name"),
                   "quantity": _proposed_quantity(it, changes.get(str(it.get("item_code")), {})),
                   "requested_delivery_date": it.get("recommended_latest_delivery_date")} for it in req],
        "points_to_clarify": [str(q.get("question")).strip() for q in _list(analysis.get("critical_questions"))
                               if isinstance(q, dict) and q.get("ask") == "supplier" and q.get("question")],
        "ask_earliest_dispatch": [it.get("item_name") for it in req if it.get("risk_level") == "Critical"],
        # named so the agent can leave them out where a point mentions them
        "other_items": [it.get("item_name") for it in ordered_items(items) if it not in req],
    }


def email_item_lines(items: list[dict], analysis: dict) -> list[str]:
    changes = changes_by_item(analysis)
    lines = []
    for it in requested_items(items):
        ch = changes.get(str(it.get("item_code")), {})
        qty = _num(_proposed_quantity(it, ch))
        if ch.get("qty"):
            qty = f"=={qty}=="
        lines.append(f"- {it.get('item_name')} — purchase quantity: {qty}; "
                     f"requested delivery date: {it.get('recommended_latest_delivery_date') or 'N/A'}")
    return lines


def render_email(email_text: str, items: list[dict], analysis: dict) -> tuple[str, list[str]]:
    """The email draft with its item lines, which code puts right after the purpose: the
    paragraph after the greeting. Returns (markdown, problems)."""
    lines = email_item_lines(items, analysis)
    text = (email_text or "").strip().replace("\r\n", "\n")
    text = re.sub(r"(?m)^[ \t]*\{?ITEMS\}?[ \t]*\n?", "", text)  # a placeholder from earlier prompts
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    problems = []
    greeting = next((n for n, p in enumerate(paras) if p.lstrip().lower().startswith("dear")), None)
    if greeting is None:
        problems.append("no greeting found; the item lines were put after the first paragraph")
        at = 1
    else:
        first_line, _, rest = paras[greeting].partition("\n")
        if rest.strip():  # greeting and purpose in one paragraph
            paras[greeting:greeting + 1] = [first_line, rest.strip()]
        at = greeting + 2
    paras = paras[:at] + ["\n".join(lines)] + paras[at:]
    return _red(EMAIL_NOTE) + "\n\n" + "\n\n".join(paras) + "\n", problems
