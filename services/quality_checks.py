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
                   "po_date_changes": list, "quantity_changes": list, "recommended_actions": list}
NO_HISTORY_SENTENCE = "No historical supplier or item data available"

CHECK_NAMES = {
    "C1": "Output format",
    "C2": "Items match the input",
    "C3": "Figures match the input",
    "C4": "Every judgement cites findings that apply",
    "C5": "Proposed changes add up",
    "C6": "Each Critical item has an action on its stock position",
    "C7": "Summary names every Critical item",
    "C8": "Summary within about 100 words",
    "C9": "Arrows in the text match the proposed changes",
    "J1": "No invented or distorted history",
    "J2a": "Supplier problem raised as a question",
    "J2b": "Item problem raised as a question",
    "J2c": "No question presumes a problem the data does not show",
    "J7": "Stock figures quoted match the input",
}

# Blocking: the analysis must not be used as it is if this fails; the verdict follows these
# alone. Warning: usable, but a person should look. (docs/eval_rubric.md, decided 2026-10-23)
BLOCKING, WARNING = "blocking", "warning"
SEVERITY = {
    "C1": BLOCKING, "C2": BLOCKING, "C3": BLOCKING, "C4": BLOCKING, "C5": BLOCKING, "C6": BLOCKING,
    "C7": WARNING, "C8": WARNING, "C9": BLOCKING,
    "J1": BLOCKING,
    "J2a": WARNING, "J2b": WARNING,  # until replaced by the check on documents left unused
    "J2c": WARNING,
    "J7": BLOCKING,
    "P1": WARNING, "P2": WARNING, "P3": BLOCKING,
}
SUMMARY_WORDS = 100

_JUDGE_TEXT = {
    "J7": "Every figure from the stock data that the analysis quotes anywhere (summary, findings, "
          "questions, actions, reasons) matches the input for that item exactly: available stock, weeks "
          "to stockout, expected stockout weeks (the gap), suggested quantity, recommended order and "
          "delivery dates. For an item with a proposed change in po_date_changes or quantity_changes, "
          "the changed figure (its order date or its quantity) is written as '<pipeline value> → "
          "<proposed value>' wherever the text mentions it; writing only one of the two fails. Figures "
          "that come from the history documents are not judged here.",
    "J1": "Every statement the analysis makes about supplier or item history (incidents, trends, "
          "reliability) is supported by the history documents given, none of them is distorted (for "
          "example, 'high demand' reported as 'medium demand'), and none is attributed to a supplier or "
          "item its document does not name (for example, a supplier-wide quality note presented as a "
          "problem of one particular item).",
    "J2c": "No critical question presumes a supplier or item problem that the inputs do not show. "
           "Neutral questions that assume no problem, such as asking whether anything is about to "
           "change, pass. Questions about problems the inputs do show also pass.",
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
    """C1-C6 against the structured output (prompt v5). Each entry: {"pass": bool, "detail": str}."""
    obj = _as_obj(analysis) or {}
    docs = history_docs or []
    out: dict[str, dict[str, Any]] = {}
    expected = {str(it.get("item_code")): it for it in items}

    # C1 the fields are there, with the right types
    out["C1"] = _result([f"{f} missing or not a {t.__name__}" for f, t in REQUIRED_FIELDS.items()
                         if not isinstance(obj.get(f), t)], "all required fields present")

    findings = _entries(obj, "findings")
    questions = _entries(obj, "critical_questions")
    actions = _entries(obj, "recommended_actions")
    po_changes, qty_changes = _entries(obj, "po_date_changes"), _entries(obj, "quantity_changes")
    judgements = ([("question", q, [str(c) for c in q.get("items") or []]) for q in questions]
                  + [("action", a, [str(c) for c in a.get("items") or []]) for a in actions]
                  + [("order-date change", c, [str(c.get("item_code"))]) for c in po_changes]
                  + [("quantity change", c, [str(c.get("item_code"))]) for c in qty_changes])

    # C2 every item named is in the input
    named = [str(c) for f in findings for c in (f.get("items") or [])] + [c for _, _, codes in judgements for c in codes]
    out["C2"] = _result([f"not in input: {c}" for c in sorted(set(named)) if c not in expected],
                        "every item named is in the input")

    # C3 the changes start from the pipeline's figures
    wrong = []
    for c in po_changes:
        src = expected.get(str(c.get("item_code")), {})
        if src and c.get("original_po_date") != src.get("recommended_latest_po_date"):
            wrong.append(f"{c.get('item_code')} original order date {c.get('original_po_date')}, "
                         f"pipeline {src.get('recommended_latest_po_date')}")
    for c in qty_changes:
        src = expected.get(str(c.get("item_code")), {})
        if src and _num(c.get("original_quantity")) != _num(src.get("suggested_quantity")):
            wrong.append(f"{c.get('item_code')} original quantity {c.get('original_quantity')}, "
                         f"pipeline {src.get('suggested_quantity')}")
    out["C3"] = _result(wrong, "changes start from the pipeline's figures")

    # C4 citations: findings rest on documents that apply or on the stock position; every
    # question, change and action cites findings that exist and concern its items
    by_id = {f.get("id"): f for f in findings}
    problems = []
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

    # C5 arithmetic of the proposed changes, and no proposed order date already passed
    problems = []
    for c in po_changes:
        try:
            orig, shift = date.fromisoformat(c["original_po_date"]), float(c["shift_weeks"])
            want = (orig + timedelta(days=round(shift * 7))).isoformat()
            if c.get("proposed_po_date") != want:
                problems.append(f"{c.get('item_code')} order date {c.get('original_po_date')} moved by {c.get('shift_weeks')} "
                                f"weeks is {want}, not {c.get('proposed_po_date')}")
            elif snapshot_date and want < snapshot_date:
                problems.append(f"{c.get('item_code')} proposed order date {want} is before the snapshot date")
        except (KeyError, TypeError, ValueError):
            problems.append(f"{c.get('item_code')} order-date change is incomplete")
    for c in qty_changes:
        orig, factor = _num(c.get("original_quantity")), _num(c.get("factor"))
        if orig is None or factor is None:
            problems.append(f"{c.get('item_code')} quantity change is incomplete")
            continue
        want = math.ceil(round(orig * factor, 6))
        if _num(c.get("proposed_quantity")) != want:
            problems.append(f"{c.get('item_code')} quantity {c.get('original_quantity')} × {c.get('factor')} rounded up "
                            f"is {want}, not {c.get('proposed_quantity')}")
    out["C5"] = _result(problems, "the changes add up" if po_changes or qty_changes else "no changes proposed")

    # C6 every Critical item has an action that cites a stock finding about it
    missing = []
    for it in critical_items(items):
        code = str(it.get("item_code"))
        stock_ids = {f.get("id") for f in findings
                     if isinstance(f.get("basis"), dict) and f["basis"].get("stock") and code in map(str, f.get("items") or [])}
        if not any(code in map(str, a.get("items") or []) and set(a.get("findings") or []) & stock_ids for a in actions):
            missing.append(f"{it.get('item_name')} ({code})")
    out["C6"] = _result([f"no action on the stock position of {m}" for m in missing],
                        "each Critical item has one" if critical_items(items) else "no Critical items")

    # C7 the summary names every Critical item; C8 its length
    summary = str(obj.get("summary") or "").strip()
    unnamed = [f"{it.get('item_name')} ({it.get('item_code')})" for it in critical_items(items)
               if _norm(it.get("item_name")) not in _norm(summary)]
    out["C7"] = _result([f"the summary does not name {u}" for u in unnamed],
                        "names every Critical item" if critical_items(items) else "no Critical items")
    words = len(summary.split())
    out["C8"] = _result([f"{words} words"] if words > SUMMARY_WORDS else [], f"{words} words")

    # C9 every "original -> proposed" written in the text is one of the proposed changes
    allowed = change_pairs(obj)
    wrong = [f"{a} → {b}" for a, b in arrows_in(analysis_text(obj)) if (a, b) not in allowed]
    out["C9"] = _result([f"not a proposed change: {w}" for w in wrong],
                        "every arrow matches a change" if allowed else "no changes proposed")
    return out


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
    pairs = {(_figure(c.get("original_po_date")), _figure(c.get("proposed_po_date")))
             for c in _entries(obj, "po_date_changes")}
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
    parts += [str(c.get("reason") or "") for f in ("po_date_changes", "quantity_changes") for c in _entries(obj, f)]
    return "\n".join(parts)


# ----------------------------------------------------------------------------- judge checks

def applicable_judge_checks(items: list[dict[str, Any]], history_docs: list[dict[str, Any]],
                            labels: dict | None = None) -> list[str]:
    """
    Which judge checks apply. Offline, labels decide J2a / J2b; in production they apply when
    some document concerns the supplier (J2a) or names an item (J2b), and the judge decides
    whether that document describes a problem.
    """
    docs = history_docs or []
    hs = any(d.get("supplier") for d in docs)
    hi = any(d.get("items") or d.get("item_codes") for d in docs)
    checks = ["J1", "J2c", "J7"]
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


def check_text(check: str, items: list[dict[str, Any]], labelled: bool) -> str:
    if check in ("J2a", "J2b"):
        return (_J2_LABELLED if labelled else _J2_UNLABELLED)[check]
    crit = ", ".join(f"{it.get('item_name')} ({it.get('item_code')}): expected stockout "
                     f"{it.get('expected_stockout_weeks')} weeks" for it in critical_items(items))
    return _JUDGE_TEXT[check].format(critical=crit or "none")


JUDGE_SYSTEM = (
    "You audit a purchasing analysis written by an AI for a buyer. You are given the exact inputs the AI "
    "received and its output. Answer each check strictly with true or false and cite the part of the "
    "output that decides it. Judge only what the check asks; do not reward length or style."
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


def judge_prompt(supplier: str, items: list[dict[str, Any]], history_docs: list[dict[str, Any]],
                 analysis: Any, checks: list[str], labelled: bool) -> str:
    lines = [f"- {c}: {check_text(c, items, labelled)}" for c in checks]
    return (
        "INPUTS\n"
        f"Supplier: {supplier}\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n"
        f"History documents:\n{history_block(history_docs)}\n\n"
        "AI OUTPUT\n"
        f"{json.dumps(analysis, ensure_ascii=False, indent=2) if isinstance(analysis, dict) else analysis}\n\n"
        "CHECKS\n" + "\n".join(lines) + "\n\n"
        'Return JSON only: {"<check id>": {"pass": true|false, "evidence": "<short quote or reason>"}, ...} '
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


def _call_openai(model: str, system: str, user: str) -> tuple[str, dict[str, int]]:
    from openai import OpenAI
    from config import settings
    client = OpenAI(api_key=settings.openai_api_key or _key("OPENAI_API_KEY"),
                    timeout=_judge_timeout(), max_retries=0)
    resp = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    u = resp.usage
    return resp.choices[0].message.content, _usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))


def _call_anthropic(model: str, system: str, user: str) -> tuple[str, dict[str, int]]:
    import anthropic
    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
                                 timeout=_judge_timeout(), max_retries=0)
    resp = client.messages.create(model=model, max_tokens=2000, system=system,
                                  messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, _usage(resp.usage.input_tokens, resp.usage.output_tokens)


def _call_gemini(model: str, system: str, user: str) -> tuple[str, dict[str, int]]:
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
def _judge_call(provider: str, model: str, system: str, user: str) -> dict[str, Any]:
    """One judge request. Traced in LangSmith (when tracing is on) with its prompt, answer
    and token usage, under the evaluation node that made it."""
    text, usage = PROVIDERS[provider](model, system, user)
    from services.run_record import note_usage  # the judge's tokens, in the run record
    note_usage(f"{provider}:{model}", usage["input_tokens"], usage["output_tokens"])
    return {"output": text, "usage_metadata": usage}


@traceable(run_type="chain", name="Judge")
def run_judge(spec: str, prompt: str, checks: list[str], retries: int = 2) -> dict[str, dict[str, Any]]:
    """spec is 'provider:model'. A check the judge could not answer comes back as pass=None."""
    provider, model = spec.split(":", 1)
    for attempt in range(retries + 1):
        try:
            raw = _judge_call(provider, model, JUDGE_SYSTEM, prompt,
                              langsmith_extra={"metadata": {"ls_provider": provider, "ls_model_name": model,
                                                            "checks": checks, "attempt": attempt + 1}})["output"]
            m = re.search(r"\{[\s\S]*\}", raw or "")
            data = json.loads(m.group(0)) if m else {}
            result = {}
            for c in checks:
                v = data.get(c) if isinstance(data.get(c), dict) else {}
                p = v.get("pass")
                result[c] = {"pass": p if isinstance(p, bool) else None, "evidence": v.get("evidence", "")}
            return result
        except Exception as e:  # record and retry; never drop a check silently
            if attempt == retries:
                return {c: {"pass": None, "evidence": f"judge error: {e}"} for c in checks}
            time.sleep(2 * (attempt + 1))


# ----------------------------------------------------------------------------- purchase request text

APPROVAL_CHECK_NAMES = {
    "P1": "Purchase request adds nothing to its input",
    "P2": "Purchase request paragraphs within length",
    "P3": "Purchase request cites only documents given",
}
_P1_TEXT = ("Every fact in purpose and justification is supported by the input: no fact, risk or "
            "assessment is added that the input does not give, figures are quoted exactly with no word "
            "that judges them (for example calling a stated duration quick or slow), and a word used in "
            "place of the input's does not change the meaning. Saying why an action from the input "
            "matters passes when the findings behind it support it, and so does writing a check to be "
            "done before ordering as a condition the request rests on.")


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", (text or "").strip()) if s.strip()])


def approval_checks(payload: dict[str, Any], approval: dict[str, Any], citation_problems: list[str],
                    judge: str, retries: int = 2) -> dict[str, Any]:
    """Checks on the approval agent's text: P1 by the judge against the agent's input, P2 and P3
    in code. Same verdict rule as the analysis checks."""
    purpose, justification = str(approval.get("purpose") or ""), str(approval.get("justification") or "")
    p, j = _sentences(purpose), _sentences(justification)
    length = [m for m in (f"purpose has {p} sentences (1-2)" if not 1 <= p <= 2 else "",
                          f"justification has {j} sentences (2-4)" if not 2 <= j <= 4 else "") if m]
    results = {"P2": {"pass": not length, "detail": "; ".join(length) or "within length"},
               "P3": {"pass": not citation_problems, "detail": "; ".join(citation_problems) or "all cited documents given"}}
    prompt = ("INPUT GIVEN TO THE WRITER\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n"
              "TEXT WRITTEN\n" + json.dumps({"purpose": purpose, "justification": justification}, ensure_ascii=False, indent=2)
              + "\n\nCHECKS\n- P1: " + _P1_TEXT + "\n\n"
              'Return JSON only: {"P1": {"pass": true|false, "evidence": "<short quote or reason>"}}')
    judged = run_judge(judge, prompt, ["P1"], retries=retries)
    results = {"P1": {"pass": judged["P1"]["pass"], "detail": judged["P1"]["evidence"]}, **results}
    return {**verdict(results), "checks": results, "judge": judge}


def approval_markdown(result: dict[str, Any]) -> str:
    return ("\n## Purchase Request Text\n\n"
            f"**Result:** {verdict_sentence(result)}\n\n" + checks_table(result, APPROVAL_CHECK_NAMES))


# ----------------------------------------------------------------------------- production entry point

def evaluate_analysis(supplier: str, items: list[dict[str, Any]], analysis: Any,
                      history_docs: list[dict[str, Any]], judge: str,
                      snapshot_date: str | None = None, retries: int = 2) -> dict[str, Any]:
    """
    All applicable checks for one analysis. The verdict is decided in code: the analysis is
    ready only if every check passed, and a check the judge could not answer counts as failed.
    """
    code = code_checks(items, analysis, history_docs, snapshot_date)
    checks = applicable_judge_checks(items, history_docs)
    judged = run_judge(judge, judge_prompt(supplier, items, history_docs,
                                           analysis, checks, labelled=False), checks, retries=retries)
    results = {**{k: {"pass": v["pass"], "detail": v["detail"]} for k, v in code.items()},
               **{k: {"pass": v["pass"], "detail": v["evidence"]} for k, v in judged.items()}}
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
    return "| Check | Severity | Result | Detail |\n| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n"


def evaluation_markdown(supplier: str, result: dict[str, Any]) -> str:
    return (
        "# Analysis Quality Check\n\n"
        f"**Supplier:** {supplier}  \n"
        f"**Result:** {verdict_sentence(result)}  \n"
        f"**Checked by:** code, and `{result['judge']}` for the judgement checks\n\n"
        + checks_table(result, CHECK_NAMES)
    )
