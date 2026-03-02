"""
Agents: Analysis, Report Doc, PR Draft, PR Doc, Email Draft.
Implements agent workflows using OpenAI + LangChain + LangGraph.
"""
import json
import re
from typing import Any, TypedDict, Annotated, List, Union
import operator

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from config import settings
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
    EVALUATION_AGENT_SYSTEM,
)


def _llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


# ----- Tools for Analysis Agent -----
@tool
def supplier_history(query: str) -> str:
    """Look up past information about ONE supplier. Input: short JSON or text including supplier name (e.g. 'SupplierA'). Output: concise documents about delivery delays, price changes, quality incidents, negotiation patterns. If no relevant document, return empty list."""
    docs = search_supplier_history(query, k=5)
    return "\n\n".join(d.page_content for d in docs) if docs else "No supplier history found."


@tool
def item_history(query: str) -> str:
    """Look up past information about ONE OR MORE items. Input: text including one or more item codes (e.g. 'Item history for 100000 ItemA and 100004 ItemE'). Output: concise documents about stock-outs, demand spikes, quality incidents, lead times. If no relevant document, return empty list."""
    docs = search_item_history(query, k=5)
    return "\n\n".join(d.page_content for d in docs) if docs else "No item history found."


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


def run_analysis_agent(
    input_json: dict[str, Any],
    supplier_history_override: str = None,
    item_history_override: str = None
) -> dict[str, Any]:
    """
    Analysis Agent: input = { snapshot_date, supplier, items[] }.
    Tools: supplier_history, item_history (unless overridden).
    Output: { purchasing_report_markdown, critical_questions[], replenishment_timeline[] }.
    """
    llm = _llm().bind_tools([supplier_history, item_history])
    user_text = json.dumps(input_json, ensure_ascii=False)
    messages = [
        SystemMessage(content=ANALYSIS_AGENT_SYSTEM),
        HumanMessage(content=user_text),
    ]
    
    # Synthetic history injection for training or testing
    if supplier_history_override or item_history_override:
        # Manually construct tool results
        synthetic_results = []
        fake_tool_calls_list = []
        
        if supplier_history_override:
            synthetic_results.append(ToolMessage(content=supplier_history_override, tool_call_id="synth_s_1"))
            fake_tool_calls_list.append({"name": "supplier_history", "args": {"query": "synthetic"}, "id": "synth_s_1"})
        
        if item_history_override:
            synthetic_results.append(ToolMessage(content=item_history_override, tool_call_id="synth_i_1"))
            fake_tool_calls_list.append({"name": "item_history", "args": {"query": "synthetic"}, "id": "synth_i_1"})
            
        # Create message as if AI called the tools
        ai_msg = AIMessage(content="", tool_calls=fake_tool_calls_list)
        
        # Construct [System, Human, AI_with_ToolCalls, ToolResults] sequence for LLM
        extended_messages = messages + [ai_msg] + synthetic_results
        final_response = llm.invoke(extended_messages)
    else:
        # Standard tool-call loop (dynamic history lookup)
        first_resp = llm.invoke(messages)
        
        t_calls = getattr(first_resp, "tool_calls", []) or []
        if t_calls:
            t_results = []
            for tc in t_calls:
                t_name = (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)) or "supplier_history"
                t_args = (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})) or {}
                t_id = (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")) or ""
                
                if t_name == "supplier_history":
                    t_out = supplier_history.invoke(t_args.get("query", str(input_json.get("supplier", ""))))
                elif t_name == "item_history":
                    t_query = t_args.get("query", " ".join(f"item_code: {i.get('item_code')}" for i in input_json.get("items", [])))
                    t_out = item_history.invoke(t_query)
                else:
                    t_out = ""
                
                t_results.append(ToolMessage(content=str(t_out), tool_call_id=t_id))
            
            # Re-invoke with tool results
            final_response = llm.invoke(messages + [first_resp] + t_results)
        else:
            final_response = first_resp

    final_text = final_response.content if hasattr(final_response, "content") else str(final_response)
    try:
        return _extract_json_from_text(final_text)
    except Exception:
        return {
            "purchasing_report_markdown": final_text,
            "critical_questions": [],
            "replenishment_timeline": input_json.get("items", []),
        }


def run_report_doc_agent(analysis_result: dict[str, Any]) -> str:
    """Report Doc Agent: analysis_result → Markdown report. Optional: retrieve analysis_examples."""
    examples = search_analysis_examples("analysis report structure and tone", k=2)
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
    examples = search_request_examples("purchase request structure", k=2)
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
    examples = search_request_examples("purchase requisition format", k=2)
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
    examples = search_email_examples("supplier email tone and structure", k=2)
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
) -> str:
    """Evaluation Agent: Critique the analysis and provide a score report."""
    llm = _llm(model="gpt-4o") # Use strong model for evaluation
    payload = {
        "supplier": supplier,
        "items": items,
        "analysis_output": analysis_output,
        "provided_supplier_history": supplier_history or "No supplier history was provided.",
        "provided_item_history": item_history or "No item history was provided.",
    }
    user = json.dumps(payload, ensure_ascii=False)
    out = llm.invoke([
        SystemMessage(content=EVALUATION_AGENT_SYSTEM),
        HumanMessage(content=user),
    ])
    return out.content if hasattr(out, "content") else str(out)
# ----- LangGraph Implementation -----

class PurchasingState(TypedDict):
    """LangGraph state definition."""
    # Core fields retained for backward compatibility
    snapshot_date: str
    supplier: str
    risk_level: str
    items: list[dict]
    
    # Inter-agent result passing
    analysis_output: dict[str, Any]
    report_md: str
    pr_draft: dict[str, Any]
    pr_md: str
    email_text: str
    evaluation_md: str
    
    # Loop control
    iteration_count: int
    correction_feedback: str
    is_valid_email: bool
    
    # Synthetic history override for training (Optional)
    supplier_history_override: str
    item_history_override: str

def analysis_node(state: PurchasingState):
    input_json = {
        "snapshot_date": state["snapshot_date"],
        "supplier": state["supplier"],
        "items": state["items"],
    }
    out = run_analysis_agent(
        input_json, 
        supplier_history_override=state.get("supplier_history_override"),
        item_history_override=state.get("item_history_override")
    )
    return {"analysis_output": out}

def evaluation_node(state: PurchasingState):
    """Node that evaluates the quality of analysis output."""
    out = run_evaluation_agent(
        state["supplier"],
        state["items"],
        state["analysis_output"],
        supplier_history=state.get("supplier_history_override"),
        item_history=state.get("item_history_override"),
    )
    return {"evaluation_md": out}

def report_node(state: PurchasingState):
    analysis_result = {
        "snapshot_date": state["snapshot_date"],
        "supplier": state["supplier"],
        "purchasing_report_markdown": state["analysis_output"].get("purchasing_report_markdown", ""),
        "critical_questions": state["analysis_output"].get("critical_questions", []),
        "replenishment_timeline": state["analysis_output"].get("replenishment_timeline", state["items"]),
    }
    out = run_report_doc_agent(analysis_result)
    return {"report_md": out}

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
        "risk_level": input_data["items"][0].get("risk_level", "N/A") if input_data["items"] else "N/A",
        "iteration_count": 0,
        "correction_feedback": "",
        "is_valid_email": False,
        "supplier_history_override": input_data.get("supplier_history_override"),
        "item_history_override": input_data.get("item_history_override")
    }
    
    # Execute graph
    final_state = purchasing_graph.invoke(initial_state)
    return final_state
