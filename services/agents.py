"""
Agents and the pipeline graph (prompt v5): Analysis, Approval (the purchase request's
narrative), Email Draft with its validator, and the evaluation node's judge. The analysis
report, the rest of the purchase request and the email's item lines are assembled in code
(services/documents.py).
"""
import json
import re
from typing import Any, TypedDict, Annotated, List, Union
import operator

import copy
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from config import settings
from services import documents
from services.item_grouping import data_issue_lines, group_risk_level
from services.vector_store import (
    search_history,
    search_request_examples,
    search_email_examples,
)
from services.prompts import (
    ANALYSIS_AGENT_SYSTEM,
    ANALYSIS_REVISION_REQUEST,
    APPROVAL_AGENT_SYSTEM,
    EMAIL_DRAFT_AGENT_SYSTEM,
    EMAIL_REVISION_REQUEST_TEMPLATE,
    EMAIL_REVISION_SYSTEM_SUFFIX,
    EMAIL_VALIDATOR_SYSTEM,
    EMAIL_VALIDATOR_TEMPLATE,
)
from services.quality_checks import (
    BLOCKING, CHECK_NAMES, SEVERITY, approval_checks, approval_markdown, code_checks, evaluate_analysis,
    evaluation_markdown, has_history,
)
from services.run_record import instrument


# Model per agent, recorded with every run (services/run_record.py).
AGENT_MODELS = {
    "analysis": "gpt-4o",
    "approval": "gpt-4o",
    # gpt-4o-mini and gpt-4o both carried the analyst's wording into the email, naming the
    # supplier's past problem; gpt-5.4-mini followed the rules (prompts/CHANGES.md, v5)
    "email_draft": "gpt-5.4-mini",
    "email_rewrite": "gpt-5.4-mini",
    "email_validator": "gpt-4o-mini",
    "judge": settings.evaluator_model,
}


def _llm(model: str = "gpt-4o") -> ChatOpenAI:
    # gpt-5 models take only their default temperature
    kwargs = {} if model.startswith("gpt-5") else {"temperature": 0}
    return ChatOpenAI(model=model, api_key=settings.openai_api_key,
                      timeout=settings.llm_timeout_seconds, max_retries=settings.llm_retries, **kwargs)


# ----- History retrieval for the Analysis Agent -----
# One retrieval, in code, before the model is called: the documents about this supplier or
# these items from the window before the snapshot date, a few per scope (see search_history).
# Each document keeps the supplier and items named in its header, so the model is told what
# it applies to rather than guessing it. The model never writes a search query.
def retrieve_history(supplier: str, items: list[dict], as_of: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = [(str(i.get("item_code") or ""), str(i.get("item_name") or "")) for i in items]
    return search_history(supplier, pairs, as_of=as_of, months=settings.history_window_months,
                          per_scope=settings.history_docs_per_scope)


def history_for_model(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the model sees of each document: a number (cited by proposed changes, resolved back
    to the file name by code), its header fields and its text."""
    return [{"id": n, "supplier": d.get("supplier") or "", "items": d.get("items") or [],
             "item_codes": d.get("item_codes") or [], "date": d.get("date") or "", "text": d.get("text") or ""}
            for n, d in enumerate(docs, 1)]


def legacy_history_docs(supplier: str, items: list[dict], supplier_text: str | None,
                        item_text: str | None) -> list[dict[str, Any]]:
    """
    History given as the earlier two free-text fields (training data generated before the
    split was removed). The supplier text becomes a supplier-wide document; the item text
    names no item in its header, so it is attached to every item in the input.
    """
    docs = []
    if has_history(supplier_text):
        docs.append({"supplier": supplier, "items": [], "item_codes": [], "date": "",
                     "source": "history supplied with the request", "text": supplier_text})
    if has_history(item_text):
        docs.append({"supplier": "", "items": [str(i.get("item_name")) for i in items],
                     "item_codes": [str(i.get("item_code")) for i in items], "date": "",
                     "source": "history supplied with the request", "text": item_text})
    return docs


def _extract_json_from_text(text: str) -> dict | list:
    """Extract JSON block from markdown/text."""
    # ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return json.loads(m.group(1).strip())
    # {...} or [...]
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text.strip())


# ----- Referenced documents (recorded by code, not by the model) -----
def document_references(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One row per history document the analysis was given."""
    return [{"source": d.get("source") or "(file name not recorded)",
             "supplier": d.get("supplier") or "",
             "items": ", ".join(d.get("items") or d.get("item_codes") or []),
             "date": d.get("date") or ""} for d in docs]


def run_analysis_agent(
    input_json: dict[str, Any],
    history_override: list[dict[str, Any]] | None = None,
    supplier_history_override: str | None = None,
    item_history_override: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Analysis Agent: input = { snapshot_date, supplier, items[] }.
    History is retrieved in code (or supplied for evaluation and training: history_override
    as documents, or the two earlier free-text fields) and passed with the items in one call.
    Returns (analysis, history_docs): the documents exactly as the model was given them,
    for the evaluation node.
    """
    supplier = str(input_json.get("supplier") or "")
    items = input_json.get("items", [])
    window = None
    if history_override is not None:
        docs = history_override
    elif supplier_history_override or item_history_override:
        docs = legacy_history_docs(supplier, items, supplier_history_override, item_history_override)
    else:
        docs, window = retrieve_history(supplier, items, str(input_json.get("snapshot_date") or ""))

    user_text = json.dumps({**input_json, "history": history_for_model(docs)}, ensure_ascii=False)
    final_response = _llm(AGENT_MODELS["analysis"]).invoke([
        SystemMessage(content=ANALYSIS_AGENT_SYSTEM),
        HumanMessage(content=user_text),
    ])

    final_text = final_response.content if hasattr(final_response, "content") else str(final_response)
    try:
        analysis = _extract_json_from_text(final_text)
    except Exception:
        # not JSON: kept for the run record; the format check (C1) fails it
        analysis = {"unparsed_output": final_text}
    if isinstance(analysis, dict):
        analysis["references"] = {"documents": document_references(docs), "window": window}
    return analysis, docs


def run_approval_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """Approval Agent: the purchase request's Purpose and Justification, for the approvers.
    payload comes from documents.approval_input."""
    examples = search_request_examples(k=2)
    examples_text = "\n\n".join(d.page_content for d in examples) if examples else ""
    user = json.dumps(payload, ensure_ascii=False)
    if examples_text:
        user = "Reference (tone only):\n" + examples_text + "\n\nInput:\n" + user
    out = _llm(AGENT_MODELS["approval"]).invoke([
        SystemMessage(content=APPROVAL_AGENT_SYSTEM),
        HumanMessage(content=user),
    ])
    text = out.content if hasattr(out, "content") else str(out)
    try:
        result = _extract_json_from_text(text)
        return result if isinstance(result, dict) else {"unparsed_output": text}
    except Exception:
        return {"unparsed_output": text}


def run_email_draft_agent(payload: dict[str, Any]) -> str:
    """Email Draft Agent: plain-text supplier email. payload comes from documents.email_input."""
    examples = search_email_examples(k=2)
    examples_text = "\n\n".join(d.page_content for d in examples) if examples else ""
    llm = _llm(AGENT_MODELS["email_draft"])
    user = json.dumps(payload, ensure_ascii=False)
    if examples_text:
        user = "Reference (tone only):\n" + examples_text + "\n\nInput:\n" + user
    out = llm.invoke([
        SystemMessage(content=EMAIL_DRAFT_AGENT_SYSTEM),
        HumanMessage(content=user),
    ])
    return out.content if hasattr(out, "content") else str(out)


def run_evaluation_agent(
    supplier: str,
    items: list[dict],
    analysis_output: dict[str, Any],
    history_docs: list[dict[str, Any]] | None = None,
    snapshot_date: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Check the analysis against the rubric in services/quality_checks.py: deterministic
    checks in code, judgement checks by a model other than the one that wrote it.
    Returns (report markdown, structured result).
    """
    result = evaluate_analysis(supplier, items, analysis_output, history_docs or [],
                               judge=settings.evaluator_model, snapshot_date=snapshot_date,
                               retries=settings.judge_retries)
    return evaluation_markdown(supplier, result), result
# ----- LangGraph Implementation -----

class PurchasingState(TypedDict):
    """LangGraph state definition."""
    # Core fields retained for backward compatibility
    snapshot_date: str
    supplier: str
    risk_level: str
    items: list[dict]
    # Problems in the input data, written by code and placed first in the report
    data_issue_lines: list[str]

    # Inter-agent result passing
    analysis_output: dict[str, Any]
    report_md: str
    approval: dict[str, Any]          # the approval agent's Purpose and Justification
    approval_input: dict[str, Any]    # what the approval agent was given, for its check
    approval_check: dict[str, Any]    # checks on its text (P1-P3)
    pr_md: str
    email_text: str                   # the email agent's text, as checked by the validator
    email_md: str                     # the email draft with its item lines and note
    # Problems found while assembling the documents (bad citations, missing placeholder)
    document_issues: Annotated[list[str], operator.add]
    # The run id, shown at the foot of the report and in the purchase request
    run_id: str
    evaluation_md: str
    evaluation_result: dict[str, Any]
    
    # Loop control
    iteration_count: int
    correction_feedback: str
    is_valid_email: bool
    # Every email check, for the run record only (the rewrite receives correction_feedback)
    validation_log: Annotated[list[dict], operator.add]
    
    # History supplied instead of retrieved, for evaluation and training (optional):
    # history_override as documents, or the two earlier free-text fields
    history_override: list[dict]
    supplier_history_override: str
    item_history_override: str

    # History documents the Analysis Agent was given, passed to the evaluation node
    history_docs: list[dict]
    # The analysis rewrite loop: every attempt (output, checks, the feedback sent), the feedback
    # waiting for the next rewrite, and how many rewrites have been made
    analysis_attempts: Annotated[list[dict], operator.add]
    analysis_feedback: str
    analysis_rewrites_done: int

def apply_system_notes(analysis: Any, items: list[dict]) -> Any:
    """
    (Prompt v4 and earlier: v5 has no per-item notes.)
    Put the pipeline's fixed sentences (data gaps, invalid figures, Critical stockout
    gaps) at the start of each item's notes. They are computed in code, so they must
    reach the report whether or not the model repeated them.
    """
    if not isinstance(analysis, dict) or not isinstance(analysis.get("replenishment_timeline"), list):
        return analysis
    fixed_by_code = {str(it.get("item_code")): it.get("system_notes") or [] for it in items}
    for entry in analysis["replenishment_timeline"]:
        if not isinstance(entry, dict):
            continue
        existing = str(entry.get("notes") or "")
        missing = [n for n in fixed_by_code.get(str(entry.get("item_code")), []) if n not in existing]
        if missing:
            entry["notes"] = " ".join(missing + ([existing] if existing else []))
    return analysis


def analysis_node(state: PurchasingState):
    input_json = {
        "snapshot_date": state["snapshot_date"],
        "supplier": state["supplier"],
        "items": state["items"],
    }
    out, history_docs = run_analysis_agent(
        input_json,
        history_override=state.get("history_override"),
        supplier_history_override=state.get("supplier_history_override"),
        item_history_override=state.get("item_history_override"),
    )
    return {"analysis_output": out, "history_docs": history_docs, "analysis_rewrites_done": 0}


def _blocking_failures(checks: dict[str, dict[str, Any]]) -> list[str]:
    """Blocking checks that failed, as lines for the rewrite request. A check the judge could
    not answer is not one: there is nothing to correct."""
    return [f"{k} ({CHECK_NAMES.get(k, k)}): {v.get('detail') or ''}" for k, v in checks.items()
            if SEVERITY.get(k) == BLOCKING and v.get("pass") is False]


def run_analysis_rewrite(input_json: dict[str, Any], docs: list[dict[str, Any]], previous: dict[str, Any],
                         feedback: str) -> dict[str, Any]:
    """The analysis agent corrects its own output: the same input, its previous answer, and the
    checks it failed. The references (which documents it was given) carry over."""
    user_text = json.dumps({**input_json, "history": history_for_model(docs)}, ensure_ascii=False)
    shown = {k: v for k, v in previous.items() if k != "references"}
    answer = _llm(AGENT_MODELS["analysis"]).invoke([
        SystemMessage(content=ANALYSIS_AGENT_SYSTEM),
        HumanMessage(content=user_text),
        AIMessage(content=json.dumps(shown, ensure_ascii=False)),
        HumanMessage(content=feedback),
    ])
    text = answer.content if hasattr(answer, "content") else str(answer)
    try:
        out = _extract_json_from_text(text)
    except Exception:
        out = None
    if not isinstance(out, dict):
        return previous  # an answer that is not JSON changes nothing; the checks still fail
    out["references"] = previous.get("references")
    return out

def evaluation_node(state: PurchasingState):
    """Checks the analysis; a blocking failure sends it back for a rewrite (up to
    analysis_rewrites). Code checks come first: if one fails and a rewrite is still possible, the
    judge is not called, as the analysis will change. Otherwise the judge's checks run too, and a
    blocking one that fails also sends the analysis back."""
    analysis, items = state["analysis_output"], state["items"]
    docs, snapshot = state.get("history_docs") or [], state["snapshot_date"]
    started = time.perf_counter()
    rewrite_left = state.get("analysis_rewrites_done", 0) < settings.analysis_rewrites
    code = code_checks(items, analysis, docs, snapshot)
    code_failed = _blocking_failures(code)
    if code_failed and rewrite_left:
        checks, failed, judged = code, code_failed, False
        result = None
    else:
        report, result = run_evaluation_agent(state["supplier"], items, analysis, history_docs=docs,
                                              snapshot_date=snapshot)
        checks, failed, judged = result["checks"], _blocking_failures(result["checks"]), True
    feedback = (ANALYSIS_REVISION_REQUEST.replace("{failures}", "\n".join(f"- {line}" for line in failed))
                if failed and rewrite_left else "")
    attempt = {"attempt": state.get("analysis_rewrites_done", 0) + 1, "output": copy.deepcopy(analysis),
               "checks": checks, "judged": judged, "feedback": feedback or None,
               "seconds": round(time.perf_counter() - started, 2)}
    update = {"analysis_attempts": [attempt], "analysis_feedback": feedback}
    if result is not None:
        update.update({"evaluation_md": report, "evaluation_result": result})
    return update


def rewrite_node(state: PurchasingState):
    """The analysis agent corrects what the checks found."""
    out = run_analysis_rewrite({"snapshot_date": state["snapshot_date"], "supplier": state["supplier"],
                                "items": state["items"]}, state.get("history_docs") or [],
                               state["analysis_output"], state["analysis_feedback"])
    return {"analysis_output": out, "analysis_rewrites_done": state.get("analysis_rewrites_done", 0) + 1}


def after_evaluation(state: PurchasingState):
    return "rewrite" if state.get("analysis_feedback") else "documents"


def report_node(state: PurchasingState):
    """The analysis report, assembled in code."""
    analysis = state["analysis_output"]
    lines = state.get("data_issue_lines")
    if lines is None:
        lines = data_issue_lines(state["items"])
    md = documents.render_analysis_report(
        state["supplier"], state["snapshot_date"], state["items"], analysis,
        state.get("history_docs") or [], (analysis.get("references") or {}).get("window"),
        lines, state.get("run_id") or "")
    return {"report_md": md}

def approval_node(state: PurchasingState):
    """The purchase request: the approval agent writes Purpose and Justification, code the rest.
    No agent call when no item is due for a request (every item Low)."""
    analysis, items, docs = state["analysis_output"], state["items"], state.get("history_docs") or []
    approval, payload = None, {}
    if documents.requested_items(items):
        payload = documents.approval_input(state["supplier"], items, analysis, docs)
        approval = run_approval_agent(payload)
    md, problems = documents.render_purchase_request(
        state["supplier"], state["snapshot_date"], items, analysis, docs, approval, state.get("run_id") or "")
    return {"approval": approval or {}, "approval_input": payload, "pr_md": md,
            "document_issues": [f"purchase request: {p}" for p in problems]}

def approval_check_node(state: PurchasingState):
    """Checks on the purchase request's text, added to the quality-check document. Runs beside
    the email steps, since nothing waits for it."""
    if not state.get("approval"):
        return {}
    problems = [p.removeprefix("purchase request: ") for p in state.get("document_issues") or []
                if p.startswith("purchase request: ")]
    result = approval_checks(state.get("approval_input") or {}, state["approval"], problems,
                             judge=settings.evaluator_model, retries=settings.judge_retries)
    return {"approval_check": result}

def finish_node(state: PurchasingState):
    """Runs once the evaluation and both document branches are done. The purchase request's red
    draft note gets the blocking checks not passed and the warnings, which the buyer settles
    before submitting (the judge's results arrive only now, as the documents do not wait for
    them). The analysis report gets none: it is built from the analysis after its rewrites,
    and the full results are in the quality-check document, which also gets the purchase
    request's checks."""
    update = {"pr_md": documents.fill_check_notes(
        state.get("pr_md") or "", [state.get("evaluation_result") or {}, state.get("approval_check") or {}])}
    if state.get("approval_check"):
        update["evaluation_md"] = (state.get("evaluation_md") or "") + approval_markdown(state["approval_check"])
    return update

def email_draft_node(state: PurchasingState):
    """The email agent's text. After a failed check, rewritten by a stronger model with the
    validator's feedback."""
    payload = documents.email_input(state["snapshot_date"], state["supplier"], state["items"],
                                    state["analysis_output"])
    if state.get("correction_feedback") and state["iteration_count"] > 0:
        llm = _llm(model=AGENT_MODELS["email_rewrite"])  # stronger model for the rewrite
        feedback_prompt = EMAIL_REVISION_REQUEST_TEMPLATE.replace("{feedback}", state["correction_feedback"])
        out = llm.invoke([
            SystemMessage(content=EMAIL_DRAFT_AGENT_SYSTEM + EMAIL_REVISION_SYSTEM_SUFFIX),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False) + feedback_prompt),
        ])
        email_text = out.content if hasattr(out, "content") else str(out)
    else:
        email_text = run_email_draft_agent(payload)
    return {"email_text": email_text, "iteration_count": state.get("iteration_count", 0) + 1}

def validator_node(state: PurchasingState):
    """DLP validator: checks if email draft is safe for external communication. It reads the
    email agent's own text; the item lines, written in code, are not part of it."""
    email = documents.email_for_check(state["email_text"])
    
    # 1. Heuristic check (keyword-based)
    leak_keywords = ["stock level", "weeks to oos", "risk level", "internal analysis", "replenishment timeline", "wks_to_oos"]
    leaks = [k for k in leak_keywords if k in email.lower()]
    
    if leaks:
        feedback = f"Found internal terminology: {', '.join(leaks)}"
        return {
            "is_valid_email": False,
            "correction_feedback": feedback,
            "validation_log": [{"attempt": state.get("iteration_count", 0), "check": "keywords",
                                "passed": False, "feedback": feedback}],
        }
    
    # 2. LLM-based precision check
    llm = _llm(model=AGENT_MODELS["email_validator"])
    check_prompt = EMAIL_VALIDATOR_TEMPLATE.replace("{email}", email)

    out = llm.invoke([
        SystemMessage(content=EMAIL_VALIDATOR_SYSTEM),
        HumanMessage(content=check_prompt)
    ])
    result = out.content.strip()
    
    if result.startswith("PASS"):
        return {"is_valid_email": True, "correction_feedback": "",
                "validation_log": [{"attempt": state.get("iteration_count", 0), "check": "llm",
                                    "passed": True, "feedback": ""}]}
    else:
        feedback = result.replace("FAIL:", "").strip()
        return {"is_valid_email": False, "correction_feedback": feedback,
                "validation_log": [{"attempt": state.get("iteration_count", 0), "check": "llm",
                                    "passed": False, "feedback": feedback}]}

def should_continue(state: PurchasingState):
    if state.get("is_valid_email") or state.get("iteration_count", 0) >= 3:
        return "email_doc"
    return "email_draft"

def email_doc_node(state: PurchasingState):
    """The email draft: everything but the requests is written in code (greeting, purpose, item
    lines, closing, signature, draft note); the validator sees only the agent's requests."""
    md, problems = documents.render_email(state.get("email_text") or "", state["items"], state["analysis_output"],
                                          state["supplier"])
    return {"email_md": md, "document_issues": [f"email: {p}" for p in problems]}

# Build the graph.
# The analysis is checked, and rewritten on a blocking failure (code or judge), before any
# document is written, so the documents come only from an analysis that passed or used up its
# rewrites. The documents are one node of the main graph, a graph of their own inside it
# (report, then the purchase request, then its check beside the email steps).

# The documents: report, then the purchase request, then its check beside the email steps.
documents_flow = StateGraph(PurchasingState)
documents_flow.add_node("report", instrument("report", report_node))
documents_flow.add_node("approval", instrument("approval", approval_node))
documents_flow.add_node("approval_check", instrument("approval_check", approval_check_node))
documents_flow.add_node("email_draft", instrument("email_draft", email_draft_node))
documents_flow.add_node("validator", instrument("validator", validator_node))
documents_flow.add_node("email_doc", instrument("email_doc", email_doc_node))
documents_flow.set_entry_point("report")
documents_flow.add_edge("report", "approval")
documents_flow.add_edge("approval", "email_draft")
documents_flow.add_edge("approval", "approval_check")
documents_flow.add_edge("email_draft", "validator")
documents_flow.add_conditional_edges(
    "validator",
    should_continue,
    {
        "email_draft": "email_draft",
        "email_doc": "email_doc",
    }
)
documents_flow.add_edge("email_doc", END)
documents_flow.add_edge("approval_check", END)
documents_graph = documents_flow.compile()

# What the documents step hands back to the main graph
DOCUMENT_KEYS = ("report_md", "approval", "approval_input", "approval_check", "pr_md",
                 "email_text", "email_md", "iteration_count", "correction_feedback", "is_valid_email")


def documents_node(state: PurchasingState):
    out = documents_graph.invoke(state)
    update = {k: out[k] for k in DOCUMENT_KEYS if k in out}
    # lists the main graph accumulates: only what the documents added
    update["validation_log"] = out.get("validation_log") or []
    update["document_issues"] = out.get("document_issues") or []
    return update


workflow = StateGraph(PurchasingState)
workflow.add_node("analysis", instrument("analysis", analysis_node))
workflow.add_node("evaluation", instrument("evaluation", evaluation_node))
workflow.add_node("rewrite", instrument("rewrite", rewrite_node))
workflow.add_node("documents", documents_node)
workflow.add_node("finish", finish_node)
workflow.set_entry_point("analysis")
workflow.add_edge("analysis", "evaluation")
# a blocking failure, code or judge, sends the analysis back before any document is written
workflow.add_conditional_edges("evaluation", after_evaluation, {"rewrite": "rewrite", "documents": "documents"})
workflow.add_edge("rewrite", "evaluation")
workflow.add_edge("documents", "finish")
workflow.add_edge("finish", END)

purchasing_graph = workflow.compile()

def run_purchasing_pipeline_graph(input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute full pipeline for one supplier group via LangGraph."""
    initial_state = {
        "snapshot_date": input_data["snapshot_date"],
        "supplier": input_data["supplier"],
        "items": input_data["items"],
        "risk_level": group_risk_level(input_data["items"]),
        "iteration_count": 0,
        "correction_feedback": "",
        "is_valid_email": False,
        "history_override": input_data.get("history_override"),
        "supplier_history_override": input_data.get("supplier_history_override"),
        "item_history_override": input_data.get("item_history_override"),
        "data_issue_lines": input_data.get("data_issue_lines", []),
        "run_id": input_data.get("run_id", ""),
    }
    
    # Execute graph
    final_state = purchasing_graph.invoke(initial_state)
    return final_state
