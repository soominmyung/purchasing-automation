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
import os
import re
import time
from typing import Any

from langsmith import traceable

REQUIRED_FIELDS = ("purchasing_report_markdown", "critical_questions", "replenishment_timeline")
NO_HISTORY_SENTENCE = "No historical supplier or item data available"

CHECK_NAMES = {
    "C1": "Output format",
    "C2": "Items match the input",
    "C3": "Figures match the input",
    "J1": "No invented or distorted history",
    "J1b": "No false 'no history' statement",
    "J2a": "Supplier problem raised as a question",
    "J2b": "Item problem raised as a question",
    "J2c": "No question presumes a problem the data does not show",
    "J3": "Absence of history stated",
    "J5": "Critical items handled in the summary",
}

_JUDGE_TEXT = {
    "J1": "Every statement the analysis makes about supplier or item history (incidents, trends, "
          "reliability) is supported by the history documents given, none of them is distorted (for "
          "example, 'high demand' reported as 'medium demand'), and none is attributed to a supplier or "
          "item its document does not name (for example, a supplier-wide quality note presented as a "
          "problem of one particular item).",
    "J1b": f"The analysis never says that no history is available (for example with the sentence "
           f"'{NO_HISTORY_SENTENCE}') for an item that some given document applies to. A document "
           "naming this supplier and no item applies to every item; a document naming items applies "
           "to those items.",
    "J2c": "No critical question presumes a supplier or item problem that the inputs do not show. "
           "Neutral questions that assume no problem, such as asking whether anything is about to "
           "change, pass. Questions about problems the inputs do show also pass.",
    "J3": "The analysis states that no history is available and invents none.",
    "J5": "For each Critical item ({critical}), the report summary names the item together with its "
          "expected stockout figure as given, and recommends at least one mitigation such as "
          "expedited shipping or an alternative source.",
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

def code_checks(items: list[dict[str, Any]], analysis: Any) -> dict[str, dict[str, Any]]:
    """C1-C3 against the structured output. Each entry: {"pass": bool, "detail": str}."""
    obj = _as_obj(analysis)
    out: dict[str, dict[str, Any]] = {}

    missing = [f for f in REQUIRED_FIELDS if not obj or f not in obj]
    bad_types = [f for f in ("critical_questions", "replenishment_timeline")
                 if obj and f in obj and not isinstance(obj[f], list)]
    out["C1"] = {"pass": not missing and not bad_types,
                 "detail": "; ".join(filter(None, [
                     f"missing {', '.join(missing)}" if missing else "",
                     f"not a list: {', '.join(bad_types)}" if bad_types else ""])) or "all required fields present"}

    timeline = obj.get("replenishment_timeline") if obj and isinstance(obj.get("replenishment_timeline"), list) else []
    expected = {str(it.get("item_code")): it for it in items}
    seen: dict[str, int] = {}
    for entry in timeline:
        code = str(entry.get("item_code")) if isinstance(entry, dict) else None
        seen[code] = seen.get(code, 0) + 1
    extra = sorted(c for c in seen if c not in expected)
    absent = sorted(c for c in expected if c not in seen)
    repeated = sorted(c for c, n in seen.items() if n > 1 and c in expected)
    out["C2"] = {"pass": not (extra or absent or repeated),
                 "detail": "; ".join(filter(None, [
                     f"not in input: {', '.join(extra)}" if extra else "",
                     f"left out: {', '.join(absent)}" if absent else "",
                     f"listed twice: {', '.join(repeated)}" if repeated else ""])) or "one entry per input item"}

    if not timeline:
        out["C3"] = {"pass": False, "detail": "no replenishment_timeline to compare against the input"}
        return out
    wrong = []
    for entry in timeline:
        if not isinstance(entry, dict) or str(entry.get("item_code")) not in expected:
            continue
        src = expected[str(entry.get("item_code"))]
        for field in ("available_stock", "wks_to_oos"):
            given, said = _num(src.get(field)), _num(entry.get(field))
            if given != said and not (given is not None and said is not None and abs(given - said) < 1e-6):
                wrong.append(f"{entry.get('item_code')} {field}: input {src.get(field)}, output {entry.get(field)}")
    out["C3"] = {"pass": not wrong, "detail": "; ".join(wrong) or "all figures match"}
    return out


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
    checks = ["J1", "J2c"]
    if docs:
        checks.append("J1b")
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
    if not docs:
        checks.append("J3")
    if critical_items(items):
        checks.append("J5")
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

def _usage(input_tokens: int | None, output_tokens: int | None) -> dict[str, int]:
    i, o = int(input_tokens or 0), int(output_tokens or 0)
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def _call_openai(model: str, system: str, user: str) -> tuple[str, dict[str, int]]:
    from openai import OpenAI
    from config import settings
    client = OpenAI(api_key=settings.openai_api_key or _key("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    u = resp.usage
    return resp.choices[0].message.content, _usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))


def _call_anthropic(model: str, system: str, user: str) -> tuple[str, dict[str, int]]:
    import anthropic
    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"))
    resp = client.messages.create(model=model, max_tokens=2000, system=system,
                                  messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, _usage(resp.usage.input_tokens, resp.usage.output_tokens)


def _call_gemini(model: str, system: str, user: str) -> tuple[str, dict[str, int]]:
    from google import genai
    client = genai.Client(api_key=_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
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


# ----------------------------------------------------------------------------- production entry point

def evaluate_analysis(supplier: str, items: list[dict[str, Any]], analysis: Any,
                      history_docs: list[dict[str, Any]], judge: str) -> dict[str, Any]:
    """
    All applicable checks for one analysis. The verdict is decided in code: the analysis is
    ready only if every check passed, and a check the judge could not answer counts as failed.
    """
    code = code_checks(items, analysis)
    checks = applicable_judge_checks(items, history_docs)
    judged = run_judge(judge, judge_prompt(supplier, items, history_docs,
                                           analysis, checks, labelled=False), checks)
    results = {**{k: {"pass": v["pass"], "detail": v["detail"]} for k, v in code.items()},
               **{k: {"pass": v["pass"], "detail": v["evidence"]} for k, v in judged.items()}}
    failed = [k for k, v in results.items() if v["pass"] is not True]
    return {"verdict": "ready" if not failed else "needs_review", "failed": failed,
            "checks": results, "judge": judge}


def evaluation_markdown(supplier: str, result: dict[str, Any]) -> str:
    total, failed = len(result["checks"]), result["failed"]
    if not failed:
        verdict = f"Ready for procurement review: all {total} checks passed."
    else:
        verdict = f"Needs human review: {len(failed)} of {total} checks did not pass."
    rows = []
    for cid, v in result["checks"].items():
        mark = "Pass" if v["pass"] is True else ("Fail" if v["pass"] is False else "Not checked")
        detail = str(v["detail"] or "").replace("|", "/").replace("\n", " ")
        rows.append(f"| {CHECK_NAMES.get(cid, cid)} ({cid}) | {mark} | {detail} |")
    return (
        "# Analysis Quality Check\n\n"
        f"**Supplier:** {supplier}  \n"
        f"**Result:** {verdict}  \n"
        f"**Checked by:** code, and `{result['judge']}` for the judgement checks\n\n"
        "| Check | Result | Detail |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n"
    )
