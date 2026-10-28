"""
Quality checks on the analysis agent's output, defined once and used in two places:

- the evaluation node, which checks every analysis the pipeline produces
- scripts/eval_v2_score.py, which scores systems offline (see docs/eval_rubric.md)

Code checks are deterministic and run on the structured output. Judge checks are yes/no
questions put to a model that did not write the analysis. Which judge checks apply is
decided from the inputs alone in production; offline, scenario labels decide it instead,
so a scenario designed to contain a supplier problem is always checked for one.
"""
import json
import math
import os
import re
import time
from datetime import date, timedelta
from typing import Any

from langsmith import traceable

REQUIRED_FIELDS = {"summary": str, "findings": list, "critical_questions": list,
                   "lead_time_changes": list, "quantity_changes": list, "figures_after_changes": list,
                   "recommended_actions": list}
DEFAULT_LEAD_WEEKS = 18  # services/item_grouping.py, TOTAL_LEAD_WEEKS
NO_HISTORY_SENTENCE = "No historical supplier or item data available"

CHECK_NAMES = {
    "C1": "Output format",
    "C2": "Items match the input",
    "C3": "Figures match the input",
    "C4": "Every judgement cites findings that apply",
    "C5": "Proposed changes well formed and add up",
    "C6": "Each Critical item has an action on its stock position",
    "C7": "Summary names every Critical item",
    "C8": "Summary within about 100 words",
    "C9": "Arrows in the text match the proposed changes",
    "C10": "Figures after the lead-time changes follow the rules",
    "J1a": "History statements found in the documents",
    "J1b": "History not distorted",
    "J1c": "History not attributed to what it does not name",
    "J2a": "Supplier problem raised as a question",
    "J2b": "Item problem raised as a question",
    "J2c": "No question presumes a problem the data does not show",
    "J3a": "Each proposed change goes the way its findings point",
    "J3b": "Each proposed change is the size its findings support",
    "J3c": "No item the same reason applies to is left out",
    "J4": "Stock figures quoted match the input",
    "J5": "Questions marked for the supplier reveal nothing internal",
    "J6": "Each question and action follows from its findings",
    "J7": "No document with a problem for this order left unused",
    "J8": "Findings state facts only",
}

# Blocking: the analysis must not be used as it is if this fails; the verdict follows these
# alone. Warning: usable, but a person should look. (docs/eval_rubric.md, decided 2026-10-23)
BLOCKING, WARNING = "blocking", "warning"
SEVERITY = {
    "C1": BLOCKING, "C2": BLOCKING, "C3": BLOCKING, "C4": BLOCKING, "C5": BLOCKING, "C6": BLOCKING,
    "C7": WARNING, "C8": WARNING, "C9": BLOCKING, "C10": BLOCKING,
    "J1a": BLOCKING, "J1b": BLOCKING, "J1c": BLOCKING,
    "J2a": WARNING, "J2b": WARNING,  # until J7 has shown it can replace them
    "J2c": WARNING,
    "J3a": BLOCKING, "J3b": BLOCKING, "J3c": BLOCKING,  # J3: whether the proposed changes hold
    "J4": BLOCKING,
    "J5": BLOCKING,
    "J6": WARNING, "J7": WARNING, "J8": WARNING,
    "P1a": WARNING, "P1b": WARNING, "P1c": WARNING, "P2": WARNING, "P3": BLOCKING,
    # the email's E1-E3 (exposure) are the validator's, which sends it back to be rewritten
    "E4": WARNING, "E5": WARNING, "E6": WARNING, "E7": WARNING,
}
# Judge checks renumbered 2026-10-29 so the ids run without gaps (J3-J5 and J7b had been dropped
# earlier). Records, calibration files and hand scores written before keep the old ids; read them
# through current_id, and only them: J8 is an id in both numberings. Mapping also in docs/eval_rubric.md.
LEGACY_IDS = {"J6a": "J3a", "J6b": "J3b", "J6c": "J3c", "J7a": "J4", "J8": "J5", "J9": "J6", "J10": "J7", "J11": "J8"}


def current_id(cid: str) -> str:
    return LEGACY_IDS.get(cid, cid)


SUMMARY_WORDS = 100

_JUDGE_TEXT = {
    # J1: the history as the documents give it
    "J1a": "Every statement the analysis makes about supplier or item history (incidents, trends, "
           "reliability) is found in the history documents given; none is invented.",
    "J1b": "No statement about the history distorts what a document says: nothing is made stronger, "
           "weaker or different (for example, 'high demand' reported as 'medium demand').",
    "J1c": "No document's content is attributed to a supplier or item the document does not name (for "
           "example, a supplier-wide quality note presented as a problem of one particular item).",
    "J2c": "No critical question presumes a supplier or item problem that the inputs do not show. "
           "Neutral questions that assume no problem, such as asking whether anything is about to "
           "change, pass. Questions about problems the inputs do show also pass.",
    # J3: whether the proposed order-date and quantity changes hold (applies when there are any)
    "J3a": "Each proposed change follows from the findings it cites: what those findings say is a reason "
           "for the lead time in supplier_lead_time and lead_time_changes to be longer or shorter than "
           "18 weeks, or for the quantity in quantity_changes to be larger or smaller, as proposed.",
    "J3b": "The size of each proposed change fits what the cited findings say, and its reason explains that "
           "size. For an item's lead time this includes whether the supplier-wide reason and the item's own "
           "reasons were rightly added together or rightly taken as one. Where the findings report a delay or "
           "a change without giving its size, a size passes if the reason makes clear, in whatever words, that "
           "it is the analysis's own estimate rather than a figure from the documents and is to be confirmed, "
           "as long as it is in keeping with how the findings describe it (a slight increase, a significant "
           "delay).",
    "J3c": "No item that a reason applies to is left without it: an item whose final lead time stays at "
           "18 weeks while supplier_lead_time changes, or that misses a change other items get for the same "
           "reason, has a reason for that which the findings support. If there is no such item, this check "
           "passes.",
    # J4: stock figures in the prose (the summary, the findings' text, the questions, the actions and
    # the changes' reasons; not the values in the change fields)
    "J4": "Every figure from the stock data quoted in the prose (the summary, the findings' text, the "
           "questions, the actions and the changes' reasons; not the values in the change fields) matches "
           "that item's figures after the proposed changes exactly (the pipeline's figures for an item with no "
           "change): available stock, weeks to stockout, lead time, expected stockout weeks (the gap), "
           "suggested quantity, recommended order and delivery dates, risk level. Figures that come from "
           "the history documents are not judged here.",
    "J5": "No critical question marked 'supplier' reveals anything internal to the company: its stock "
          "levels or weeks to stockout, how short its stock is, its budget, or its own analysis or plans. "
          "What the supplier itself did or announced is not internal. Asking what expedited shipping, "
          "earlier or partial delivery the supplier can offer for an item, with quantities and dates, passes "
          "even though it shows the item is needed soon.",
    "J6": "Each critical question and recommended action follows from the findings it cites: what those "
          "findings say is a reason to ask that question or take that action. (Proposed changes are "
          "judged in J3.)",
    "J7": "These documents are behind no question, proposed change or action: {unused}. None of them "
           "describes a problem that matters for this order and calls for a question, a change or an "
           "action; if one does, this check fails.",
    "J8": "Every finding states only facts from its documents or from the stock figures, with no "
           "judgement of what they mean for this order (that belongs in the questions, changes and "
           "actions).",
}
# J2a / J2b: offline the label guarantees a problem exists; in production the judge decides.
_J2_LABELLED = {
    "J2a": "At least one critical question addresses the supplier problem described in the history.",
    "J2b": "At least one critical question addresses an item problem described in the history.",
}
_J2_UNLABELLED = {
    "J2a": "If the history describes a problem with the supplier that matters for ordering, at least "
           "one critical question addresses it. If it describes no such problem, this check passes.",
    "J2b": "If the history describes a problem with one of these items that matters for ordering, at "
           "least one critical question addresses it. If it describes no such problem, this check passes.",
}


# ----------------------------------------------------------------------------- helpers

def has_history(text: str | None) -> bool:
    """For free-text history: False for empty text and for 'no history' placeholders."""
    t = (text or "").strip()
    return bool(t) and not re.match(r"(?i)^no (supplier |item )?history (found|was provided|available)", t)


def critical_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [it for it in items if it.get("risk_level") == "Critical"]


def _as_obj(output: Any) -> dict | None:
    if isinstance(output, dict):
        return output
    try:
        return json.loads(output)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", output or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- code checks

def _entries(obj: dict, field: str) -> list[dict]:
    return [e for e in (obj.get(field) or []) if isinstance(e, dict)] if isinstance(obj.get(field), list) else []


def _result(problems: list[str], ok: str) -> dict[str, Any]:
    return {"pass": not problems, "detail": "; ".join(problems) or ok}


def _doc_applies(doc: dict, codes: list[str], items: list[dict]) -> bool:
    """A document can support a finding about these items (or, with no items, about the
    supplier): it is supplier-wide, or it names one of the items."""
    if not codes:
        return bool(doc.get("supplier"))
    if doc.get("supplier") and not (doc.get("items") or doc.get("item_codes")):
        return True
    named = {_norm(v) for v in (doc.get("items") or []) + (doc.get("item_codes") or [])}
    return any(_norm(c) in named or _norm(it.get("item_name")) in named
               for c in codes for it in items if str(it.get("item_code")) == c)


def _norm(v: Any) -> str:
    return re.sub(r"[\W_]+", "", str(v or "").lower())


def code_checks(items: list[dict[str, Any]], analysis: Any,
                history_docs: list[dict[str, Any]] | None = None,
                snapshot_date: str | None = None) -> dict[str, dict[str, Any]]:
    """C1-C10 against the structured output (prompt v5). Each entry: {"pass": bool, "detail": str}.
    items are the pipeline's figures; C6, C7 and C10 use the figures after the lead-time changes."""
    obj = _as_obj(analysis) or {}
    docs = history_docs or []
    out: dict[str, dict[str, Any]] = {}
    expected = {str(it.get("item_code")): it for it in items}
    final = figures_after_changes(items, obj, snapshot_date)

    # C1 the fields are there, with the right types
    sl = obj.get("supplier_lead_time")
    out["C1"] = _result([f"{f} missing or not a {t.__name__}" for f, t in REQUIRED_FIELDS.items()
                         if not isinstance(obj.get(f), t)]
                        + (["supplier_lead_time is neither an object nor null"] if sl is not None and not isinstance(sl, dict) else []),
                        "all required fields present")

    findings = _entries(obj, "findings")
    questions = _entries(obj, "critical_questions")
    actions = _entries(obj, "recommended_actions")
    lt_changes, qty_changes = _entries(obj, "lead_time_changes"), _entries(obj, "quantity_changes")
    sl = sl if isinstance(sl, dict) else None
    judgements = ([("question", q, [str(c) for c in q.get("items") or []]) for q in questions]
                  + [("action", a, [str(c) for c in a.get("items") or []]) for a in actions]
                  + [("lead-time change", c, [str(x) for x in c.get("items") or []]) for c in lt_changes]
                  + ([("supplier lead time", sl, [])] if sl else [])
                  + [("quantity change", c, [str(c.get("item_code"))]) for c in qty_changes])

    # C2 every item named is in the input
    named = [str(c) for f in findings for c in (f.get("items") or [])] + [c for _, _, codes in judgements for c in codes]
    out["C2"] = _result([f"not in input: {c}" for c in sorted(set(named)) if c not in expected],
                        "every item named is in the input")

    # C3 the quantity changes start from the pipeline's figure (order dates are calculated by code
    # from the lead time, so there is nothing to check for them)
    wrong = []
    for c in qty_changes:
        src = expected.get(str(c.get("item_code")), {})
        if src and _num(c.get("original_quantity")) != _num(src.get("suggested_quantity")):
            wrong.append(f"{c.get('item_code')} original quantity {c.get('original_quantity')}, "
                         f"pipeline {src.get('suggested_quantity')}")
    out["C3"] = _result(wrong, "changes start from the pipeline's figures")

    # C4 citations: findings rest on documents that apply or on the stock position; every
    # question, change and action cites findings that exist and concern its items
    by_id = {f.get("id"): f for f in findings}
    ids = [f.get("id") for f in findings]
    problems = [f"finding id {i} is used twice" for i in sorted({i for i in ids if ids.count(i) > 1}, key=str)]
    for f in findings:
        basis = f.get("basis") if isinstance(f.get("basis"), dict) else {}
        doc_ids = basis.get("documents") or []
        codes = [str(c) for c in f.get("items") or []]
        if not doc_ids and not basis.get("stock"):
            problems.append(f"finding {f.get('id')} has no basis")
        for d in doc_ids:
            if not isinstance(d, int) or not 1 <= d <= len(docs):
                problems.append(f"finding {f.get('id')} cites document {d}, which was not given")
            elif not _doc_applies(docs[d - 1], codes, items):
                problems.append(f"finding {f.get('id')} cites document {d}, which does not apply to "
                                f"{', '.join(codes) or 'the supplier'}")
    for kind, e, codes in judgements:
        cited = e.get("findings") or []
        if not cited:
            problems.append(f"{kind} cites no finding: {str(e.get('question') or e.get('action') or e.get('item_code'))[:60]}")
        for i in cited:
            f = by_id.get(i)
            if f is None:
                problems.append(f"{kind} cites finding {i}, which does not exist")
            elif f.get("items") and codes and not set(map(str, f["items"])) & set(codes):
                problems.append(f"{kind} about {', '.join(codes)} cites finding {i}, which is about "
                                f"{', '.join(map(str, f['items']))}")
    out["C4"] = _result(problems, "every judgement cites findings that apply")

    # C5 the changes are well formed: a lead time in whole weeks, each item at most once, every item
    # given a final lead time when the supplier's changes, no change that changes nothing; quantity
    # arithmetic, never below zero
    problems = []
    seen: dict[str, int] = {}
    for c in lt_changes:
        weeks = c.get("lead_time_weeks")
        if not isinstance(weeks, (int, float)) or isinstance(weeks, bool) or weeks <= 0:
            problems.append(f"lead time for {', '.join(map(str, c.get('items') or [])) or 'no item'} is not a number of weeks")
        elif weeks == DEFAULT_LEAD_WEEKS and not sl:
            problems.append(f"lead time for {', '.join(map(str, c.get('items') or []))} is 18 weeks, which changes nothing")
        if not c.get("items"):
            problems.append("a lead-time change names no item")
        for code in map(str, c.get("items") or []):
            seen[code] = seen.get(code, 0) + 1
    problems += [f"{code} has more than one final lead time" for code, n in seen.items() if n > 1]
    if sl:
        w = sl.get("lead_time_weeks")
        if not isinstance(w, (int, float)) or isinstance(w, bool) or w <= 0 or w == DEFAULT_LEAD_WEEKS:
            problems.append("supplier_lead_time is not a lead time other than 18 weeks")
        problems += [f"{code} has no final lead time although the supplier's lead time changes"
                     for code in expected if code not in seen]
    for c in qty_changes:
        orig, factor = _num(c.get("original_quantity")), _num(c.get("factor"))
        if orig is None or factor is None:
            problems.append(f"{c.get('item_code')} quantity change is incomplete")
            continue
        if factor <= 0 or factor == 1:
            problems.append(f"{c.get('item_code')} quantity factor {c.get('factor')} is not a change"
                            if factor == 1 else f"{c.get('item_code')} quantity would fall to zero or below")
            continue
        want = math.ceil(round(orig * factor, 6))
        if _num(c.get("proposed_quantity")) != want:
            problems.append(f"{c.get('item_code')} quantity {c.get('original_quantity')} × {c.get('factor')} rounded up "
                            f"is {want}, not {c.get('proposed_quantity')}")
    out["C5"] = _result(problems, "the changes are well formed and add up" if lt_changes or qty_changes or sl
                        else "no changes proposed")

    # C6 every Critical item (after the lead-time changes) has an action that cites a stock finding about it
    missing = []
    for it in critical_items(final):
        code = str(it.get("item_code"))
        stock_ids = {f.get("id") for f in findings
                     if isinstance(f.get("basis"), dict) and f["basis"].get("stock") and code in map(str, f.get("items") or [])}
        if not any(code in map(str, a.get("items") or []) and set(a.get("findings") or []) & stock_ids for a in actions):
            missing.append(f"{it.get('item_name')} ({code})")
    out["C6"] = _result([f"no action on the stock position of {m}" for m in missing],
                        "each Critical item has one" if critical_items(final) else "no Critical items")

    # C7 the summary names every Critical item; C8 its length
    summary = str(obj.get("summary") or "").strip()
    unnamed = [f"{it.get('item_name')} ({it.get('item_code')})" for it in critical_items(final)
               if _norm(it.get("item_name")) not in _norm(summary)]
    out["C7"] = _result([f"the summary does not name {u}" for u in unnamed],
                        "names every Critical item" if critical_items(final) else "no Critical items")
    words = len(summary.split())
    out["C8"] = _result([f"{words} words"] if words > SUMMARY_WORDS else [], f"{words} words")

    # C9 every "original -> proposed" written in the text is one of the proposed changes
    allowed = change_pairs(obj)
    wrong = [f"{a} → {b}" for a, b in arrows_in(analysis_text(obj)) if (a, b) not in allowed]
    out["C9"] = _result([f"not a proposed change: {w}" for w in wrong],
                        "every arrow matches a change" if allowed else "no changes proposed")

    # C10 the figures the analysis worked out at each item's final lead time are those the rules
    # give (code recalculates them the same way; the report's tables use code's)
    by_code = {str(it.get("item_code")): it for it in final}
    changed = {str(c) for e in lt_changes for c in e.get("items") or []}
    stated = {str(e.get("item_code")): e for e in _entries(obj, "figures_after_changes")}
    problems = [f"no figures after the change for {c}" for c in sorted(changed) if c not in stated and c in by_code]
    for code, e in stated.items():
        want = by_code.get(code)
        if want is None or code not in changed:
            problems.append(f"figures given for {code}, which has no lead-time change")
            continue
        # the analysis's names (new_*) are kept apart from the input's, which hold the 18-week figures
        for field, rule in (("new_lead_time_weeks", "lead_time_weeks"), ("new_risk_level", "risk_level"),
                            ("new_gap_weeks", "expected_stockout_weeks"), ("new_latest_po_date", "recommended_latest_po_date")):
            got, exp = e.get(field), want.get(rule)
            if (_figure(got) if field != "new_risk_level" else got) != (_figure(exp) if field != "new_risk_level" else exp):
                problems.append(f"{code} {field} {got}, by the rules {exp}")
    out["C10"] = _result(problems, "the figures follow the rules" if changed else "no lead-time change")

    return out


def figures_after_changes(items: list[dict[str, Any]], analysis: dict, snapshot_date: str | None) -> list[dict[str, Any]]:
    """Each item's figures after the proposed lead-time and quantity changes, by the pipeline's
    rules (services/documents.py); the pipeline's figures when there are none or no snapshot date."""
    if not snapshot_date or not isinstance(analysis, dict):
        return items
    from services.documents import items_after_changes
    return items_after_changes(items, analysis, snapshot_date)


ARROW = re.compile(r"(\d{4}-\d{2}-\d{2}|\d[\d,]*(?:\.\d+)?)\s*(?:→|->)\s*(\d{4}-\d{2}-\d{2}|\d[\d,]*(?:\.\d+)?)")


def _figure(v: Any) -> str:
    """A date or a number, written the same way whatever its format (2,600 / 2600 / 2600.0)."""
    t = str(v).strip().replace(",", "")
    try:
        f = float(t)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return t


def change_pairs(obj: dict) -> set[tuple[str, str]]:
    """(pipeline value, proposed value) of every proposed change."""
    pairs = {(_figure(DEFAULT_LEAD_WEEKS), _figure(c.get("lead_time_weeks"))) for c in _entries(obj, "lead_time_changes")}
    if isinstance(obj.get("supplier_lead_time"), dict):
        pairs.add((_figure(DEFAULT_LEAD_WEEKS), _figure(obj["supplier_lead_time"].get("lead_time_weeks"))))
    pairs |= {(_figure(c.get("original_quantity")), _figure(c.get("proposed_quantity")))
              for c in _entries(obj, "quantity_changes")}
    return pairs


def arrows_in(text: str) -> list[tuple[str, str]]:
    return [(_figure(a), _figure(b)) for a, b in ARROW.findall(text or "")]


def analysis_text(obj: dict) -> str:
    """All the model's prose: summary, findings, questions, actions and the changes' reasons."""
    parts = [str(obj.get("summary") or "")]
    parts += [str(f.get("text") or "") for f in _entries(obj, "findings")]
    parts += [str(q.get("question") or "") for q in _entries(obj, "critical_questions")]
    parts += [str(a.get("action") or "") for a in _entries(obj, "recommended_actions")]
    parts += [str(c.get("reason") or "") for f in ("lead_time_changes", "quantity_changes") for c in _entries(obj, f)]
    if isinstance(obj.get("supplier_lead_time"), dict):
        parts.append(str(obj["supplier_lead_time"].get("reason") or ""))
    return "\n".join(parts)


# ----------------------------------------------------------------------------- judge checks

def applicable_judge_checks(items: list[dict[str, Any]], history_docs: list[dict[str, Any]],
                            labels: dict | None = None, analysis: Any = None) -> list[str]:
    """
    Which judge checks apply. Offline, labels decide J2a / J2b; in production they apply when
    some document concerns the supplier (J2a) or names an item (J2b), and the judge decides
    whether that document describes a problem.
    """
    docs = history_docs or []
    hs = any(d.get("supplier") for d in docs)
    hi = any(d.get("items") or d.get("item_codes") for d in docs)
    obj = _as_obj(analysis) or {}
    checks = ["J1a", "J1b", "J1c", "J2c", "J4"]
    if _entries(obj, "lead_time_changes") or _entries(obj, "quantity_changes") or isinstance(obj.get("supplier_lead_time"), dict):
        checks += ["J3a", "J3b", "J3c"]
    if any(q.get("ask") == "supplier" for q in _entries(obj, "critical_questions")):
        checks.append("J5")
    if _entries(obj, "critical_questions") or _entries(obj, "recommended_actions"):
        checks.append("J6")
    if unused_documents(obj, docs):
        checks.append("J7")
    if _entries(obj, "findings"):
        checks.append("J8")
    if labels is not None:
        if labels.get("supplier_issue") == "material":
            checks.append("J2a")
        if labels.get("item_issue") == "material":
            checks.append("J2b")
    else:
        if hs:
            checks.append("J2a")
        if hi:
            checks.append("J2b")
    return checks


def unused_documents(obj: dict, docs: list[dict[str, Any]]) -> list[int]:
    """History documents (by number) behind no question, proposed change or action: cited by no
    finding, or only by findings that no judgement cites."""
    cited = {i for kind in ("critical_questions", "recommended_actions", "lead_time_changes", "quantity_changes")
             for e in _entries(obj, kind) for i in (e.get("findings") or [])}
    if isinstance(obj.get("supplier_lead_time"), dict):
        cited |= set(obj["supplier_lead_time"].get("findings") or [])
    used = {d for f in _entries(obj, "findings") if f.get("id") in cited
            for d in ((f.get("basis") or {}).get("documents") or []) if isinstance(f.get("basis"), dict)}
    return [n for n in range(1, len(docs or []) + 1) if n not in used]


def check_text(check: str, items: list[dict[str, Any]], labelled: bool, analysis: Any = None,
               history_docs: list[dict[str, Any]] | None = None) -> str:
    if check in ("J2a", "J2b"):
        return (_J2_LABELLED if labelled else _J2_UNLABELLED)[check]
    unused = ", ".join(f"[{n}]" for n in unused_documents(_as_obj(analysis) or {}, history_docs or []))
    return _JUDGE_TEXT[check].format(unused=unused or "none")


JUDGE_SYSTEM = (
    "You audit a purchasing analysis written by an AI for a buyer. You are given the exact inputs the AI "
    "received and its output. For each check, first write the evidence (the part of the output that "
    "decides it), then give the verdict, true or false, that the evidence supports. Judge only what the "
    "check asks; do not reward length or style. When a check fails, also write a note: one plain "
    "sentence for the buyer, saying which part of the output may be wrong and how, with no check ids "
    "or technical terms."
)


def history_block(history_docs: list[dict[str, Any]]) -> str:
    """The history documents as the judge sees them: each with the supplier and items its
    header names, which is what it applies to."""
    if not history_docs:
        return "No history was found for this supplier or these items."
    out = []
    for n, d in enumerate(history_docs, 1):
        items = ", ".join(d.get("items") or d.get("item_codes") or [])
        scope = (f"supplier {d['supplier']}, items {items}" if d.get("supplier") and items else
                 f"supplier {d['supplier']}, all items (supplier-wide)" if d.get("supplier") else
                 f"items {items} (any supplier)")
        out.append(f"[{n}] applies to: {scope}; date: {d.get('date') or 'unknown'}\n{d.get('text', '')}")
    return "\n\n".join(out)


AFTER_FIELDS = ("item_code", "item_name", "available_stock", "wks_to_oos", "lead_time_weeks", "risk_level",
                "expected_stockout_weeks", "suggested_quantity", "recommended_latest_po_date",
                "recommended_latest_delivery_date")  # all the figures J4 compares the text with


def judge_prompt(supplier: str, items: list[dict[str, Any]], history_docs: list[dict[str, Any]],
                 analysis: Any, checks: list[str], labelled: bool, after: list[dict[str, Any]] | None = None) -> str:
    """items are the pipeline's figures (18 weeks), as the analysis received them; after, when the
    analysis proposes changes, each item's figures after them, worked out by the pipeline. Both are
    shown, apart: the changes are judged against the first (J3), the text's figures against the
    second (J4). Shown only after, the judge read the new lead time as the input's and passed a
    lead-time change it should have failed (2026-10-24)."""
    lines = [f"- {c}: {check_text(c, items, labelled, analysis, history_docs)}" for c in checks]
    shown_after = (f"Figures after the proposed changes (worked out by the pipeline from the proposed lead times "
                   f"and quantities): {json.dumps([{k: it.get(k) for k in AFTER_FIELDS} for it in after], ensure_ascii=False)}\n"
                   if after is not None and after != items else "")
    return (
        "INPUTS\n"
        f"Supplier: {supplier}\n"
        f"Items (the pipeline's figures, at a lead time of 18 weeks, as the AI received them): {json.dumps(items, ensure_ascii=False)}\n"
        + shown_after +
        f"History documents:\n{history_block(history_docs)}\n\n"
        "AI OUTPUT\n"
        f"{json.dumps(analysis, ensure_ascii=False, indent=2) if isinstance(analysis, dict) else analysis}\n\n"
        "CHECKS\n" + "\n".join(lines) + "\n\n"
        'Return JSON only: {"<check id>": {"evidence": "<short quote or reason>", "pass": true|false, "note": "<only if it fails: one plain sentence for the buyer>"}, ...} '
        "with one entry per check."
    )


def _key(*names: str) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    try:
        from dotenv import dotenv_values
        for f in (".env", ".env.txt"):
            vals = dotenv_values(f)
            for n in names:
                if vals.get(n):
                    return vals[n]
    except Exception:
        pass
    return None


# Each provider call returns (text, token usage). The usage is in LangSmith's format so the
# judge's cost shows on its trace next to the analysis agent's.

def _judge_timeout() -> float:
    """Seconds one judge request may take (config.judge_timeout_seconds); run_judge retries."""
    try:
        from config import settings
        return float(settings.judge_timeout_seconds)
    except Exception:
        return 90.0


def _usage(input_tokens: int | None, output_tokens: int | None) -> dict[str, int]:
    i, o = int(input_tokens or 0), int(output_tokens or 0)
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def _call_openai(model: str, system: str, user: str, checks: list[str] | None = None,
                 effort: str | None = None, **_) -> tuple[str, dict[str, int]]:
    from openai import OpenAI
    from config import settings
    client = OpenAI(api_key=settings.openai_api_key or _key("OPENAI_API_KEY"),
                    timeout=_judge_timeout(), max_retries=0)
    extra = {"reasoning_effort": effort} if effort else {}
    resp = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
    u = resp.usage
    return resp.choices[0].message.content, _usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))


def _call_anthropic(model: str, system: str, user: str, checks: list[str] | None = None,
                    effort: str | None = None, format: str | None = None, **_) -> tuple[str, dict[str, int]]:
    import anthropic
    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
                                 timeout=_judge_timeout(), max_retries=0)
    # One named property per check, each an object with the evidence first: with open property
    # names Haiku sometimes put every verdict into one field as a string.
    verdict = {"type": "object",
               "properties": {"evidence": {"type": "string"}, "pass": {"type": "boolean"},
                              "note": {"type": "string"}},
               "required": ["evidence", "pass"]}
    tool = {"name": "record_verdicts",
            "description": "Record one verdict for each check id.",
            "input_schema": ({"type": "object", "properties": {c: verdict for c in checks}, "required": list(checks)}
                             if checks else {"type": "object", "additionalProperties": verdict})}
    # 4000: an output with fourteen checks and their evidence came to about 1,800 tokens on average and some
    # answers stopped at the old limit of 2000, cut off mid-verdict (2026-10-24)
    # No temperature: the Messages API takes none (2026-10-24), so a verdict can differ from run to run.
    # format "schema": the answer is one JSON object shaped like the scoring sheet (each check its
    # evidence and its verdict), enforced by structured outputs instead of a forced tool call.
    config = {"effort": effort} if effort else {}
    if format == "schema":
        schema = {"type": "object", "properties": {c: verdict for c in checks}, "required": list(checks),
                  "additionalProperties": False} if checks else {"type": "object", "additionalProperties": verdict}
        verdict.setdefault("additionalProperties", False)
        config["format"] = {"type": "json_schema", "schema": schema}
        # 16000: without a forced tool the model thinks, and thinking counts toward max_tokens; at high
        # effort 4000 cut the answer off mid-JSON (2026-10-24)
        resp = client.messages.create(model=model, max_tokens=16000, system=system,
                                      messages=[{"role": "user", "content": user}], output_config=config)
        if resp.stop_reason == "max_tokens":
            raise ValueError("the judge's answer was cut off at max_tokens")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, _usage(resp.usage.input_tokens, resp.usage.output_tokens)
    extra = {"output_config": config} if config else {}
    resp = client.messages.create(model=model, max_tokens=4000, system=system,
                                  messages=[{"role": "user", "content": user}],
                                  tools=[tool], tool_choice={"type": "tool", "name": "record_verdicts"}, **extra)
    used = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
    text = json.dumps(used.input) if used is not None else "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, _usage(resp.usage.input_tokens, resp.usage.output_tokens)


def _call_gemini(model: str, system: str, user: str, checks: list[str] | None = None, **_) -> tuple[str, dict[str, int]]:
    from google import genai
    client = genai.Client(api_key=_key("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                          http_options={"timeout": int(_judge_timeout() * 1000)})  # milliseconds
    resp = client.models.generate_content(
        model=model, contents=user,
        config={"system_instruction": system, "response_mime_type": "application/json"})
    u = resp.usage_metadata
    # thinking tokens are billed as output
    out = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    return resp.text, _usage(getattr(u, "prompt_token_count", 0), out)


PROVIDERS = {"openai": _call_openai, "anthropic": _call_anthropic, "gemini": _call_gemini}


@traceable(run_type="llm", name="Judge_LLM")
def _judge_call(provider: str, model: str, system: str, user: str, checks: list[str] | None = None,
                options: dict | None = None) -> dict[str, Any]:
    """One judge request. Traced in LangSmith (when tracing is on) with its prompt, answer
    and token usage, under the evaluation node that made it."""
    text, usage = PROVIDERS[provider](model, system, user, checks, **(options or {}))
    from services.run_record import note_usage  # the judge's tokens, in the run record
    note_usage(f"{provider}:{model}", usage["input_tokens"], usage["output_tokens"])
    return {"output": text, "usage_metadata": usage}


def _verdict_from_text(text: str) -> dict:
    """A verdict the judge wrote as a string that is not valid JSON (quotes in the evidence left
    unescaped): its pass, evidence and note read from the text itself; no verdict if pass is absent."""
    m = re.search(r'"pass"\s*:\s*(true|false)', text)
    if not m:
        return {}
    def field(name):
        f = re.search(r'"' + name + r'"\s*:\s*"(.*?)"\s*(?:,\s*"(?:pass|note|evidence)"|\}\s*$)', text, re.S)
        return f.group(1) if f else ""
    return {"pass": m.group(1) == "true", "evidence": field("evidence") or text[:300], "note": field("note")}


@traceable(run_type="chain", name="Judge")
def run_judge(spec: str, prompt: str, checks: list[str], retries: int = 2) -> dict[str, dict[str, Any]]:
    """spec is 'provider:model'. A check the judge could not answer comes back as pass=None."""
    # spec: "provider:model", optionally "?effort=high&format=schema" (judge validation, 2026-10-24)
    provider, model = spec.split(":", 1)
    model, _, query = model.partition("?")
    from urllib.parse import parse_qsl
    options = dict(parse_qsl(query))
    for attempt in range(retries + 1):
        try:
            raw = _judge_call(provider, model, JUDGE_SYSTEM, prompt, checks, options,
                              langsmith_extra={"metadata": {"ls_provider": provider, "ls_model_name": model,
                                                            "checks": checks, "attempt": attempt + 1}})["output"]
            m = re.search(r"\{[\s\S]*\}", raw or "")
            data = json.loads(m.group(0)) if m else {}
            result = {}
            for c in checks:
                v = data.get(c)
                if isinstance(v, str):  # a verdict sent as a JSON string instead of an object
                    try:
                        v = json.loads(v)
                    except ValueError:
                        v = _verdict_from_text(v)
                v = v if isinstance(v, dict) else {}
                if "pass" not in v and isinstance(v.get(c), dict):  # wrapped once more
                    v = v[c]
                if "pass" not in v:  # keep what came back, to see why
                    v = {"evidence": f"no verdict for {c} in the judge's answer: {str(raw)[:300]}"}
                p = v.get("pass")
                result[c] = {"pass": p if isinstance(p, bool) else None, "evidence": v.get("evidence", ""),
                             "note": v.get("note", "") if p is False else ""}
            return result
        except Exception as e:  # record and retry; never drop a check silently
            if attempt == retries:
                return {c: {"pass": None, "evidence": f"judge error: {e}"} for c in checks}
            time.sleep(2 * (attempt + 1))


# ----------------------------------------------------------------------------- purchase request text

APPROVAL_CHECK_NAMES = {
    "P1a": "Purchase request adds nothing to its input",
    "P1b": "Purchase request quotes figures as given",
    "P1c": "Purchase request keeps the input's meaning",
    "P2": "Purchase request paragraphs within length",
    "P3": "Purchase request cites only documents given",
}
_P1_TEXT = {
    "P1a": "Purpose and Justification add no fact, risk or assessment that their input does not give. "
           "Stating a conclusion that the input does not draw, such as saying that a supplier's record "
           "supports the decision to buy from it, adds an assessment. "
           "Saying why an action from the input matters passes when the findings behind it support it, "
           "and so does writing a check to be done before ordering as a condition the request rests on.",
    "P1b": "Figures are quoted exactly as the input gives them, with no word that judges them (for "
           "example calling a stated duration quick or slow).",
    "P1c": "Where a word other than the input's is used, the meaning does not change (for example "
           "'about 60%' written as '60%').",
}


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", (text or "").strip()) if s.strip()])


def approval_checks(payload: dict[str, Any], approval: dict[str, Any], citation_problems: list[str],
                    judge: str, retries: int = 2) -> dict[str, Any]:
    """Checks on the approval agent's text: P1a-c by the judge against the agent's input, P2 and P3
    in code. Same verdict rule as the analysis checks."""
    purpose, justification = str(approval.get("purpose") or ""), str(approval.get("justification") or "")
    p, j = _sentences(purpose), _sentences(justification)
    length = [m for m in (f"purpose has {p} sentences (1-2)" if not 1 <= p <= 2 else "",
                          f"justification has {j} sentences (2-4)" if not 2 <= j <= 4 else "") if m]
    results = {"P2": {"pass": not length, "detail": "; ".join(length) or "within length"},
               "P3": {"pass": not citation_problems, "detail": "; ".join(citation_problems) or "all cited documents given"}}
    prompt = ("INPUT GIVEN TO THE WRITER\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n"
              "TEXT WRITTEN\n" + json.dumps({"purpose": purpose, "justification": justification}, ensure_ascii=False, indent=2)
              + "\n\nCHECKS\n" + "\n".join(f"- {c}: {t}" for c, t in _P1_TEXT.items()) + "\n\n"
              'Return JSON only: {"<check id>": {"evidence": "<short quote or reason>", "pass": true|false, '
              '"note": "<only if it fails: one plain sentence for the buyer>"}, ...} with one entry per check.')
    judged = run_judge(judge, prompt, list(_P1_TEXT), retries=retries)
    results = {**{c: {"pass": v["pass"], "detail": v["evidence"], "note": v.get("note", "")}
                  for c, v in judged.items()}, **results}
    return {**verdict(results), "checks": results, "judge": judge}


def approval_markdown(result: dict[str, Any]) -> str:
    return ("\n## Purchase Request Text\n\n"
            f"**Result:** {verdict_sentence(result)}\n\n" + checks_table(result, APPROVAL_CHECK_NAMES))


# ----------------------------------------------------------------------------- supplier email text

# The checks the email models were compared on (docs/email_model_comparison.md), beyond exposure:
# warnings for the buyer, not rewrites (the user, 2026-10-29). Wording as in
# scripts/compare_email_models.py after its second round.
EMAIL_CHECK_NAMES = {
    "E4": "Email covers every point to clarify",
    "E5": "Email asks no question twice",
    "E6": "Email asks the earliest dispatch where needed",
    "E7": "Email promises no order and reads as a request",
}
_E_TEXT = {
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


def email_checks(payload: dict[str, Any], requests: str, judge: str, retries: int = 2) -> dict[str, Any]:
    """E4-E7 by the judge on the email agent's requests against its input (as compared)."""
    prompt = ("INPUT GIVEN TO THE WRITER (the email's requests paragraph; the greeting, purpose, item list, "
              "closing and signature are added by code)\n" + json.dumps(payload, ensure_ascii=False, indent=2) +
              "\n\nREQUESTS WRITTEN\n" + (requests or "") + "\n\nCHECKS\n" +
              "\n".join(f"- {c}: {t}" for c, t in _E_TEXT.items()) + "\n\n"
              'Return JSON only: {"<check id>": {"evidence": "<short quote or reason>", "pass": true|false, '
              '"note": "<only if it fails: one plain sentence for the buyer>"}, ...} with one entry per check.')
    judged = run_judge(judge, prompt, list(_E_TEXT), retries=retries)
    results = {c: {"pass": v["pass"], "detail": v["evidence"], "note": v.get("note", "")} for c, v in judged.items()}
    return {**verdict(results), "checks": results, "judge": judge}


def email_markdown(result: dict[str, Any], validator_passed: bool | None, drafts: int) -> str:
    exposure = ("passed" if validator_passed else "still failed after the last rewrite") if validator_passed is not None \
        else "not run"
    warnings = result.get("warnings") or []
    return ("\n## Supplier Email\n\n"
            f"**Exposure check (E1-E3, the validator):** {exposure} ({drafts} draft{'s' if drafts != 1 else ''}).  \n"
            f"**Other checks (warnings):** {len(warnings)} of {len(result['checks'])} not passed.\n\n"
            + checks_table(result, EMAIL_CHECK_NAMES))


# ----------------------------------------------------------------------------- production entry point

def evaluate_analysis(supplier: str, items: list[dict[str, Any]], analysis: Any,
                      history_docs: list[dict[str, Any]], judge: str,
                      snapshot_date: str | None = None, retries: int = 2) -> dict[str, Any]:
    """
    All applicable checks for one analysis. The verdict is decided in code: the analysis is
    ready only if every blocking check passed, and a check the judge could not answer counts as
    failed. The judge compares the text with each item's figures after the proposed changes.
    """
    code = code_checks(items, analysis, history_docs, snapshot_date)
    checks = applicable_judge_checks(items, history_docs, analysis=analysis)
    final = figures_after_changes(items, _as_obj(analysis) or {}, snapshot_date)
    judged = run_judge(judge, judge_prompt(supplier, items, history_docs, analysis, checks, labelled=False,
                                           after=final), checks, retries=retries)
    results = {**{k: {"pass": v["pass"], "detail": v["detail"]} for k, v in code.items()},
               **{k: {"pass": v["pass"], "detail": v["evidence"], "note": v.get("note", "")}
                  for k, v in judged.items()}}
    return {**verdict(results), "checks": results, "judge": judge}


def verdict(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Ready only if every blocking check passed; a check the judge could not answer counts as
    not passed. Each result gets its severity; warnings are listed but do not decide."""
    for cid, v in results.items():
        v["severity"] = SEVERITY.get(cid, BLOCKING)
    not_passed = [k for k, v in results.items() if v["pass"] is not True]
    failed = [k for k in not_passed if results[k]["severity"] == BLOCKING]
    return {"verdict": "ready" if not failed else "needs_review", "failed": failed,
            "warnings": [k for k in not_passed if k not in failed]}


def verdict_sentence(result: dict[str, Any]) -> str:
    blocking = [k for k, v in result["checks"].items() if v.get("severity") == BLOCKING]
    warnings = result.get("warnings") or []
    note = f"; {len(warnings)} warning{'s' if len(warnings) != 1 else ''}" if warnings else "; no warnings"
    if not result["failed"]:
        return f"Ready: all {len(blocking)} blocking checks passed{note}."
    return f"Needs review: {len(result['failed'])} of {len(blocking)} blocking checks did not pass{note}."


def checks_table(result: dict[str, Any], names: dict[str, str]) -> str:
    rows = []
    for cid, v in result["checks"].items():
        mark = "Pass" if v["pass"] is True else ("Fail" if v["pass"] is False else "Not checked")
        detail = str(v["detail"] or "").replace("|", "/").replace("\n", " ")
        rows.append(f"| {names.get(cid, cid)} ({cid}) | {v.get('severity', '')} | {mark} | {detail} |")
    return "| Check | Severity | Result | Reason |\n| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n"


def evaluation_markdown(supplier: str, result: dict[str, Any], run_id: str = "", snapshot_date: str = "",
                        attempts: list[dict] | None = None, used_attempt: int | None = None) -> str:
    """The quality-check document: a checklist with the reason for each result, the run it belongs
    to, and how the analysis got there (each attempt, what failed, whether it was sent back)."""
    head = ["# Analysis Quality Check", "", f"**Supplier:** {supplier}  "]
    if run_id:
        head.append(f"**Run id:** {run_id}  ")
    if snapshot_date:
        head.append(f"**Snapshot date:** {snapshot_date}  ")
    head += [f"**Result:** {verdict_sentence(result)}  ",
             f"**Checked by:** code, and `{result['judge']}` for the judgement checks", ""]
    if attempts:
        head += ["## Attempts", "", "| Attempt | Judge run | Blocking checks failed | Sent back for a rewrite |",
                 "| --- | --- | --- | --- |"]
        for a in attempts:
            failed = [f"{k} ({CHECK_NAMES.get(k, k)}): {str(v.get('detail') or '')}" for k, v in a["checks"].items()
                      if SEVERITY.get(k) == BLOCKING and v.get("pass") is False]
            cell = "; ".join(failed).replace("|", "/").replace("\n", " ") or "none"
            judged = "yes" if a.get("judged") else "no (a code check failed first)"
            head.append(f"| {a['attempt']} | {judged} | {cell} | {'yes' if a.get('feedback') else 'no'} |")
        head.append("")
        if used_attempt:
            head += [f"The rewrites were used up; the documents are built from attempt {used_attempt}, the one "
                     "with the fewest blocking failures.", ""]
    head += ["## Analysis Checks", ""]
    return "\n".join(head) + "\n" + checks_table(result, CHECK_NAMES)
