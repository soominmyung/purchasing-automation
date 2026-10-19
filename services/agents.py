"""
Agents: Analysis, Report Doc, PR Draft, PR Doc, Email Draft.
Implements agent workflows using OpenAI + LangChain + LangGraph.
"""
import json
import re
from typing import Any, TypedDict, Annotated, List, Union
import operator

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from config import settings
from services.item_grouping import data_issue_lines, group_risk_level, insert_data_issues_section
from services.vector_store import (
    search_history,
    search_analysis_examples,
    search_request_examples,
    search_email_examples,
)
from services.prompts import (
    ANALYSIS_AGENT_SYSTEM,
    REPORT_DOC_AGENT_SYSTEM,
    PR_DRAFT_AGENT_SYSTEM,
    PR_DOC_AGENT_SYSTEM,
    EMAIL_DRAFT_AGENT_SYSTEM,
    EMAIL_REVISION_REQUEST_TEMPLATE,
    EMAIL_REVISION_SYSTEM_SUFFIX,
    EMAIL_VALIDATOR_SYSTEM,
    EMAIL_VALIDATOR_TEMPLATE,
)
from services.quality_checks import evaluate_analysis, evaluation_markdown, has_history
from services.run_record import instrument


# Model per agent, recorded with every run (services/run_record.py).
AGENT_MODELS = {
    "analysis": "gpt-4o",
    "report": "gpt-4o",
    "pr_draft": "gpt-4o",
    "pr_doc": "gpt-4o",
    "email_draft": "gpt-4o-mini",
    "email_rewrite": "gpt-4o",
    "email_validator": "gpt-4o-mini",
    "judge": settings.evaluator_model,
}


def _llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


# ----- History retrieval for the Analysis Agent -----
# One retrieval, in code, before the model is called: every history document about this
# supplier or about any of these items, most recent first. Each document keeps the supplier
# and items named in its header, so the model is told what it applies to rather than
# guessing it. The model never writes a search query.
HISTORY_DOCS_PER_ANALYSIS = 10


def retrieve_history(supplier: str, items: list[dict]) -> list[dict[str, Any]]:
    pairs = [(str(i.get("item_code") or ""), str(i.get("item_name") or "")) for i in items]
    return search_history(supplier, pairs, k=HISTORY_DOCS_PER_ANALYSIS)


def history_for_model(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the model sees of each document: its header fields and its text."""
    return [{"supplier": d.get("supplier") or "", "items": d.get("items") or [],
             "item_codes": d.get("item_codes") or [], "date": d.get("date") or "", "text": d.get("text") or ""}
            for d in docs]


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
REFERENCES_HEADING = "## Referenced Documents"


def document_references(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One row per history document the analysis was given."""
    return [{"source": d.get("source") or "(file name not recorded)",
             "supplier": d.get("supplier") or "",
             "items": ", ".join(d.get("items") or d.get("item_codes") or []),
             "date": d.get("date") or ""} for d in docs]


def references_markdown(references: dict[str, Any] | None) -> str:
    """The report's closing section: which history documents the analysis was based on."""
    refs = (references or {}).get("documents") or []
    lines = [REFERENCES_HEADING, ""]
    if not refs:
        return "\n".join(lines + ["No history was found for this supplier or these items."]) + "\n"
    lines += ["| Document | Supplier | Items | Date |", "| --- | --- | --- | --- |"]
    for r in refs:
        if r["supplier"]:
            items = r["items"] or "all items (supplier-wide)"
        else:
            items = r["items"] or "-"
        lines.append(f"| {r['source']} | {r['supplier'] or '-'} | {items} | {r['date'] or '-'} |")
    return "\n".join(lines) + "\n"


def append_references_section(markdown: str, references: dict[str, Any] | None) -> str:
    if references is None:
        return markdown
    return (markdown or "").rstrip() + "\n\n" + references_markdown(references)


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
    if history_override is not None:
        docs = history_override
    elif supplier_history_override or item_history_override:
        docs = legacy_history_docs(supplier, items, supplier_history_override, item_history_override)
    else:
        docs = retrieve_history(supplier, items)

    user_text = json.dumps({**input_json, "history": history_for_model(docs)}, ensure_ascii=False)
    final_response = _llm(AGENT_MODELS["analysis"]).invoke([
        SystemMessage(content=ANALYSIS_AGENT_SYSTEM),
        HumanMessage(content=user_text),
    ])

    final_text = final_response.content if hasattr(final_response, "content") else str(final_response)
    try:
        analysis = _extract_json_from_text(final_text)
    except Exception:
        analysis = {
            "purchasing_report_markdown": final_text,
            "critical_questions": [],
            "replenishment_timeline": input_json.get("items", []),
        }
    if isinstance(analysis, dict):
        analysis["references"] = {"documents": document_references(docs)}
    return analysis, docs


def run_report_doc_agent(analysis_result: dict[str, Any]) -> str:
    """Report Doc Agent: analysis_result → Markdown report. Optional: retrieve analysis_examples."""
    examples = search_analysis_examples(k=2)
    examples_text = "\n\n".join(d.page_content for d in examples) if examples else ""
    llm = _llm(AGENT_MODELS["report"])
    user = json.dumps(analysis_result, ensure_ascii=False)
    if examples_text:
        user = "Reference (tone/structure only):\n" + examples_text + "\n\nInput:\n" + user
    out = llm.invoke([
        SystemMessage(content=REPORT_DOC_AGENT_SYSTEM),
        HumanMessage(content=user),
    ])
    return out.content if hasattr(out, "content") else str(out)


def run_pr_draft_agent(
    snapshot_date: str,
    supplier: str,
    risk_level: str,
    analysis_output: dict[str, Any],
) -> dict[str, Any]:
    """PR Draft Agent: analysis_output → structured JSON for PR Doc Agent."""
    examples = search_request_examples(k=2)
    examples_text = "\n\n".join(d.page_content for d in examples) if examples else ""
    llm = _llm(AGENT_MODELS["pr_draft"])
    payload = {
        "snapshot_date": snapshot_date,
        "supplier": supplier,
        "risk_level": risk_level,
        "analysis_output": analysis_output,
    }
    user = json.dumps(payload, ensure_ascii=False)
    if examples_text:
        user = "Reference (structure only):\n" + examples_text + "\n\nInput:\n" + user
    out = llm.invoke([
        SystemMessage(content=PR_DRAFT_AGENT_SYSTEM),
        HumanMessage(content=user),
    ])
    text = out.content if hasattr(out, "content") else str(out)
    try:
        return _extract_json_from_text(text)
    except json.JSONDecodeError:
        return {"document_type": "purchase_request", "supplier": supplier, "snapshot_date": snapshot_date, "purchase_requests": []}


def run_pr_doc_agent(request_output: dict[str, Any]) -> str:
    """PR Doc Agent: request_output (from PR Draft) → Markdown purchase request."""
    examples = search_request_examples(k=2)
    examples_text = "\n\n".join(d.page_content for d in examples) if examples else ""
    llm = _llm(AGENT_MODELS["pr_doc"])
    user = json.dumps(request_output, ensure_ascii=False)
    if examples_text:
        user = "Reference (format only):\n" + examples_text + "\n\nInput:\n" + user
    out = llm.invoke([
        SystemMessage(content=PR_DOC_AGENT_SYSTEM),
        HumanMessage(content=user),
    ])
    return out.content if hasattr(out, "content") else str(out)


def run_email_draft_agent(
    snapshot_date: str,
    supplier: str,
    risk_level: str,
    items: list[dict],
    analysis_output: dict[str, Any],
) -> str:
    """Email Draft Agent: items + analysis_output → plain text supplier email."""
    examples = search_email_examples(k=2)
    examples_text = "\n\n".join(d.page_content for d in examples) if examples else ""
    llm = _llm(AGENT_MODELS["email_draft"])
    payload = {
        "snapshot_date": snapshot_date,
        "supplier": supplier,
        "risk_level": risk_level,
        "items": items,
        "analysis_output": analysis_output,
    }
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
) -> tuple[str, dict[str, Any]]:
    """
    Check the analysis against the rubric in services/quality_checks.py: deterministic
    checks in code, judgement checks by a model other than the one that wrote it.
    Returns (report markdown, structured result).
    """
    result = evaluate_analysis(supplier, items, analysis_output, history_docs or [],
                               judge=settings.evaluator_model)
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
    pr_draft: dict[str, Any]
    pr_md: str
    email_text: str
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

def apply_system_notes(analysis: Any, items: list[dict]) -> Any:
    """
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
    out = apply_system_notes(out, state["items"])
    return {"analysis_output": out, "history_docs": history_docs}

def evaluation_node(state: PurchasingState):
    """Node that evaluates the quality of analysis output."""
    out = run_evaluation_agent(
        state["supplier"],
        state["items"],
        state["analysis_output"],
        history_docs=state.get("history_docs") or [],
    )
    report, result = out
    return {"evaluation_md": report, "evaluation_result": result}

def report_node(state: PurchasingState):
    analysis_result = {
        "snapshot_date": state["snapshot_date"],
        "supplier": state["supplier"],
        "purchasing_report_markdown": state["analysis_output"].get("purchasing_report_markdown", ""),
        "critical_questions": state["analysis_output"].get("critical_questions", []),
        "replenishment_timeline": state["analysis_output"].get("replenishment_timeline", state["items"]),
    }
    out = run_report_doc_agent(analysis_result)
    lines = state.get("data_issue_lines")
    if lines is None:
        lines = data_issue_lines(state["items"])
    out = insert_data_issues_section(out, lines)
    return {"report_md": append_references_section(out, state["analysis_output"].get("references"))}

def pr_draft_node(state: PurchasingState):
    out = run_pr_draft_agent(
        state["snapshot_date"], 
        state["supplier"], 
        state["risk_level"], 
        state["analysis_output"]
    )
    return {"pr_draft": out}

def pr_doc_node(state: PurchasingState):
    out = run_pr_doc_agent(state["pr_draft"])
    return {"pr_md": out}

def email_draft_node(state: PurchasingState):
    # If feedback exists, augment the prompt or override system message.
    # Currently run_email_draft_agent uses a fixed prompt,
    # so we only apply revision logic when correction_feedback is present.
    
    if state.get("correction_feedback") and state["iteration_count"] > 0:
        # Feedback present: invoke agent directly for self-correction (prompt augmentation)
        llm = _llm(model=AGENT_MODELS["email_rewrite"])  # stronger model for the rewrite
        feedback_prompt = EMAIL_REVISION_REQUEST_TEMPLATE.replace("{feedback}", state["correction_feedback"])
        
        payload = {
            "snapshot_date": state["snapshot_date"],
            "supplier": state["supplier"],
            "risk_level": state["risk_level"],
            "items": state["items"],
            "analysis_output": state["analysis_output"],
        }
        user = json.dumps(payload, ensure_ascii=False) + feedback_prompt
        
        out = llm.invoke([
            SystemMessage(content=EMAIL_DRAFT_AGENT_SYSTEM + EMAIL_REVISION_SYSTEM_SUFFIX),
            HumanMessage(content=user),
        ])
        email_text = out.content if hasattr(out, "content") else str(out)
    else:
        email_text = run_email_draft_agent(
            state["snapshot_date"],
            state["supplier"],
            state["risk_level"],
            state["items"],
            state["analysis_output"]
        )
    
    return {"email_text": email_text, "iteration_count": state.get("iteration_count", 0) + 1}

def validator_node(state: PurchasingState):
    """DLP validator: checks if email draft is safe for external communication."""
    email = state["email_text"]
    
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
        return END
    return "email_draft"

# Build the graph
workflow = StateGraph(PurchasingState)

workflow.add_node("analysis", instrument("analysis", analysis_node))
workflow.add_node("evaluation", instrument("evaluation", evaluation_node))
workflow.add_node("report", instrument("report", report_node))
workflow.add_node("pr_draft", instrument("pr_draft", pr_draft_node))
workflow.add_node("pr_doc", instrument("pr_doc", pr_doc_node))
workflow.add_node("email_draft", instrument("email_draft", email_draft_node))
workflow.add_node("validator", instrument("validator", validator_node))

workflow.set_entry_point("analysis")
workflow.add_edge("analysis", "evaluation")
workflow.add_edge("evaluation", "report")
workflow.add_edge("report", "pr_draft")
workflow.add_edge("pr_draft", "pr_doc")
workflow.add_edge("pr_doc", "email_draft")
workflow.add_edge("email_draft", "validator")

workflow.add_conditional_edges(
    "validator",
    should_continue,
    {
        "email_draft": "email_draft",
        END: END
    }
)

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
    }
    
    # Execute graph
    final_state = purchasing_graph.invoke(initial_state)
    return final_state
