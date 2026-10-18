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
    search_supplier_history,
    search_item_history,
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
)
from services.quality_checks import evaluate_analysis, evaluation_markdown, has_history


def _llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


# ----- History retrieval for the Analysis Agent -----
# Retrieval is a metadata filter (supplier name, item codes) ranked by document date, done
# in code before the model is called. The model never writes a search query: it would not
# change what is retrieved, and leaving the lookup to the model meant it could be skipped.
NO_SUPPLIER_HISTORY = "No supplier history found."
NO_ITEM_HISTORY = "No item history found."
HISTORY_DOCS_PER_LOOKUP = 5


def retrieve_supplier_history(supplier: str) -> list:
    """The most recent documents for this supplier only; none if the supplier is unnamed."""
    if not supplier:
        return []
    return search_supplier_history(k=HISTORY_DOCS_PER_LOOKUP, filter={"supplier_name": supplier})


def retrieve_item_history(item_codes: list[str]) -> list:
    """The most recent documents for these item codes only; none if there are no codes."""
    if not item_codes:
        return []
    return search_item_history(k=HISTORY_DOCS_PER_LOOKUP, filter={"item_code": {"$in": item_codes}})


def _history_text(docs: list, none_text: str) -> str:
    return "\n\n".join(d.page_content for d in docs) if docs else none_text


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


def _doc_references(history: str, docs: list) -> list[dict[str, str]]:
    """One entry per source document retrieved (chunks of one file are merged)."""
    refs, seen = [], set()
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source") or "(file name not recorded)"
        applies_to = meta.get("supplier_name") if history == "supplier" else meta.get("item_code")
        date = meta.get("event_date") or (meta.get("ingested_at") or "")[:10]
        key = (source, applies_to, date)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"history": history, "source": source, "applies_to": applies_to or "", "date": date})
    return refs


def _supplied_reference(history: str, text: str) -> list[dict[str, str]]:
    """History passed in directly (evaluation, training) rather than retrieved from the store."""
    if has_history(text):
        return [{"history": history, "source": "history supplied with the request", "applies_to": "", "date": ""}]
    return []


def references_markdown(references: dict[str, Any] | None) -> str:
    """The report's closing section: which history documents the analysis was based on."""
    refs = (references or {}).get("documents") or []
    lines = [REFERENCES_HEADING, ""]
    if refs:
        lines += ["| History | Document | Applies to | Date |", "| --- | --- | --- | --- |"]
        lines += [f"| {r['history']} | {r['source']} | {r['applies_to'] or '-'} | {r['date'] or '-'} |" for r in refs]
        lines.append("")
    for history in ("supplier", "item"):
        if not any(r["history"] == history for r in refs):
            lines.append(f"- No {history} history was found.")
    return "\n".join(lines).rstrip() + "\n"


def append_references_section(markdown: str, references: dict[str, Any] | None) -> str:
    if references is None:
        return markdown
    return (markdown or "").rstrip() + "\n\n" + references_markdown(references)


def run_analysis_agent(
    input_json: dict[str, Any],
    supplier_history_override: str = None,
    item_history_override: str = None
) -> dict[str, Any]:
    """
    Analysis Agent: input = { snapshot_date, supplier, items[] }.
    The supplier and item histories are retrieved in code (or supplied as overrides for
    evaluation and training) and passed to the model with the items, in one call.
    Returns: (analysis_dict, used_supplier_history, used_item_history) — the history
    texts are exactly what the model was given, for the evaluation node.
    """
    documents: list[dict[str, str]] = []  # for the report's "Referenced Documents" section

    if supplier_history_override:
        used_supplier_history = supplier_history_override
        documents += _supplied_reference("supplier", supplier_history_override)
    else:
        docs = retrieve_supplier_history(str(input_json.get("supplier") or ""))
        used_supplier_history = _history_text(docs, NO_SUPPLIER_HISTORY)
        documents += _doc_references("supplier", docs)

    if item_history_override:
        used_item_history = item_history_override
        documents += _supplied_reference("item", item_history_override)
    else:
        item_codes = [str(i.get("item_code")) for i in input_json.get("items", []) if i.get("item_code")]
        docs = retrieve_item_history(item_codes)
        used_item_history = _history_text(docs, NO_ITEM_HISTORY)
        documents += _doc_references("item", docs)

    user_text = json.dumps(
        {**input_json, "supplier_history": used_supplier_history, "item_history": used_item_history},
        ensure_ascii=False,
    )
    final_response = _llm().invoke([
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
        analysis["references"] = {"documents": documents}
    return analysis, used_supplier_history, used_item_history


def run_report_doc_agent(analysis_result: dict[str, Any]) -> str:
    """Report Doc Agent: analysis_result → Markdown report. Optional: retrieve analysis_examples."""
    examples = search_analysis_examples(k=2)
    examples_text = "\n\n".join(d.page_content for d in examples) if examples else ""
    llm = _llm()
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
    llm = _llm()
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
    llm = _llm()
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
    llm = _llm(model="gpt-4o-mini")
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
    supplier_history: str = None,
    item_history: str = None,
) -> tuple[str, dict[str, Any]]:
    """
    Check the analysis against the rubric in services/quality_checks.py: deterministic
    checks in code, judgement checks by a model other than the one that wrote it.
    Returns (report markdown, structured result).
    """
    result = evaluate_analysis(supplier, items, analysis_output, supplier_history, item_history,
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
    
    # Synthetic history override for training (Optional)
    supplier_history_override: str
    item_history_override: str

    # History actually used by the Analysis Agent (tool output or override),
    # passed to the Evaluation Agent to verify grounding.
    retrieved_supplier_history: str
    retrieved_item_history: str

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
    out, used_supplier_history, used_item_history = run_analysis_agent(
        input_json,
        supplier_history_override=state.get("supplier_history_override"),
        item_history_override=state.get("item_history_override")
    )
    out = apply_system_notes(out, state["items"])
    return {
        "analysis_output": out,
        "retrieved_supplier_history": used_supplier_history,
        "retrieved_item_history": used_item_history,
    }

def evaluation_node(state: PurchasingState):
    """Node that evaluates the quality of analysis output."""
    out = run_evaluation_agent(
        state["supplier"],
        state["items"],
        state["analysis_output"],
        supplier_history=state.get("retrieved_supplier_history") or state.get("supplier_history_override"),
        item_history=state.get("retrieved_item_history") or state.get("item_history_override"),
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
        llm = _llm(model="gpt-4o")  # Use stronger model for revision
        feedback_prompt = f"\n\n[REVISION REQUEST]\nYour previous draft was rejected for the following reason: {state['correction_feedback']}\nPlease rewrite the email while strictly avoiding those issues."
        
        payload = {
            "snapshot_date": state["snapshot_date"],
            "supplier": state["supplier"],
            "risk_level": state["risk_level"],
            "items": state["items"],
            "analysis_output": state["analysis_output"],
        }
        user = json.dumps(payload, ensure_ascii=False) + feedback_prompt
        
        out = llm.invoke([
            SystemMessage(content=EMAIL_DRAFT_AGENT_SYSTEM + "\nSTRICT: Ensure no internal analysis terminology or stock levels are leaked."),
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
        return {
            "is_valid_email": False, 
            "correction_feedback": f"Found internal terminology: {', '.join(leaks)}"
        }
    
    # 2. LLM-based precision check
    llm = _llm(model="gpt-4o-mini")
    check_prompt = f"""Analyze the following email draft to a supplier. 
Does it contain ANY internal-only information such as:
- Internal stock quantities
- "Weeks to Out of Stock" (WksToOOS)
- Internal risk assessments (High/Medium/Low risk)
- Mentions of internal analysis logic or tools

Email Draft:
---
{email}
---
If it contains any internal leaks, respond with 'FAIL: <reason>'. 
If it is safe for external communication, respond with 'PASS'."""

    out = llm.invoke([
        SystemMessage(content="You are a strict data loss prevention (DLP) auditor."),
        HumanMessage(content=check_prompt)
    ])
    result = out.content.strip()
    
    if result.startswith("PASS"):
        return {"is_valid_email": True, "correction_feedback": ""}
    else:
        return {"is_valid_email": False, "correction_feedback": result.replace("FAIL:", "").strip()}

def should_continue(state: PurchasingState):
    if state.get("is_valid_email") or state.get("iteration_count", 0) >= 3:
        return END
    return "email_draft"

# Build the graph
workflow = StateGraph(PurchasingState)

workflow.add_node("analysis", analysis_node)
workflow.add_node("evaluation", evaluation_node)
workflow.add_node("report", report_node)
workflow.add_node("pr_draft", pr_draft_node)
workflow.add_node("pr_doc", pr_doc_node)
workflow.add_node("email_draft", email_draft_node)
workflow.add_node("validator", validator_node)

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
        "supplier_history_override": input_data.get("supplier_history_override"),
        "item_history_override": input_data.get("item_history_override")
    }
    
    # Execute graph
    final_state = purchasing_graph.invoke(initial_state)
    return final_state
