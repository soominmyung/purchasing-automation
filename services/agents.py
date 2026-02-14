"""
에이전트: Analysis, Report Doc, PR Draft, PR Doc, Email Draft.
OpenAI + LangChain으로 에이전트 워크플로우를 구현.
"""
import json
import re
from typing import Any, TypedDict, Annotated, List, Union
import operator

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
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
    """마크다운/텍스트에서 JSON 블록 추출."""
    # ```json ... ``` 또는 ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return json.loads(m.group(1).strip())
    # {...} 또는 [...]
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text.strip())


def run_analysis_agent(input_json: dict[str, Any]) -> dict[str, Any]:
    """
    Analysis Agent: input = { snapshot_date, supplier, items[] }.
    Tools: supplier_history, item_history.
    Output: { purchasing_report_markdown, critical_questions[], replenishment_timeline[] }.
    """
    llm = _llm().bind_tools([supplier_history, item_history])
    user_text = json.dumps(input_json, ensure_ascii=False)
    messages = [
        SystemMessage(content=ANALYSIS_AGENT_SYSTEM),
        HumanMessage(content=user_text),
    ]
    # 1회 호출로 도구 사용 유도 후, 도구 결과를 넣고 다시 호출하는 루프 (간단히 2회까지)
    from langchain_core.messages import ToolMessage

    response = llm.invoke(messages)
    tool_calls = getattr(response, "tool_calls", []) or []
    if tool_calls:
        tool_results = []
        for tc in tool_calls:
            name = (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)) or "supplier_history"
            args = (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})) or {}
            tid = (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")) or ""
            if name == "supplier_history":
                out = supplier_history.invoke(args.get("query", str(input_json.get("supplier", ""))))
            elif name == "item_history":
                q = args.get("query", " ".join(f"item_code: {i.get('item_code')}" for i in input_json.get("items", [])))
                out = item_history.invoke(q)
            else:
                out = ""
            tool_results.append(ToolMessage(content=str(out), tool_call_id=tid))
        messages = messages + [response] + tool_results
        response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    try:
        return _extract_json_from_text(text)
    except json.JSONDecodeError:
        return {
            "purchasing_report_markdown": text,
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
    analysis_output: dict[str, Any]
) -> str:
    """Evaluation Agent: Critique the analysis and provide a score report."""
    llm = _llm(model="gpt-4o") # Use strong model for evaluation
    payload = {
        "supplier": supplier,
        "items": items,
        "analysis_output": analysis_output
    }
    user = json.dumps(payload, ensure_ascii=False)
    out = llm.invoke([
        SystemMessage(content=EVALUATION_AGENT_SYSTEM),
        HumanMessage(content=user),
    ])
    return out.content if hasattr(out, "content") else str(out)
# ----- LangGraph Implementation -----

class PurchasingState(TypedDict):
    """LangGraph 상태 정의."""
    # 하위 호환성을 위해 기존 변수들 유지
    snapshot_date: str
    supplier: str
    risk_level: str
    items: list[dict]
    
    # 에이전트 간 결과 전달
    analysis_output: dict[str, Any]
    report_md: str
    pr_draft: dict[str, Any]
    pr_md: str
    email_text: str
    evaluation_md: str
    
    # 루프 제어
    iteration_count: int
    correction_feedback: str
    is_valid_email: bool

def analysis_node(state: PurchasingState):
    input_json = {
        "snapshot_date": state["snapshot_date"],
        "supplier": state["supplier"],
        "items": state["items"],
    }
    out = run_analysis_agent(input_json)
    return {"analysis_output": out}

def evaluation_node(state: PurchasingState):
    """분석 결과의 품질을 평가하는 노드."""
    out = run_evaluation_agent(
        state["supplier"],
        state["items"],
        state["analysis_output"]
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
    # 만약 피드백이 있다면 프롬프트를 보강하거나 시스템 메시지 수정 가능
    # 여기서는 간단히 기존 함수를 호출하되, 피드백이 있으면 인자로 전달하는 방식으로 구현 가능
    # 현재 run_email_draft_agent는 고정된 프롬프트를 쓰므로, 피드백이 있을 때만 로직 보강
    
    if state.get("correction_feedback") and state["iteration_count"] > 0:
        # 피드백이 있는 경우: 자가 수정을 지원하기 위해 에이전트 직접 호출 (프롬프트 보강)
        llm = _llm(model="gpt-4o") # 수정을 위해선 더 강력한 모델 사용
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
    """이메일 초안이 외부 공개 가능한 수준인지 검사 (Validator)."""
    email = state["email_text"]
    
    # 1. 휴리스틱 검사 (단어 기반)
    leak_keywords = ["stock level", "weeks to oos", "risk level", "internal analysis", "replenishment timeline", "wks_to_oos"]
    leaks = [k for k in leak_keywords if k in email.lower()]
    
    if leaks:
        return {
            "is_valid_email": False, 
            "correction_feedback": f"Found internal terminology: {', '.join(leaks)}"
        }
    
    # 2. LLM 기반 정밀 검사
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

# 그래프 구축
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
    """LangGraph를 사용하여 전체 파이프라인(1개 공급사 그룹) 실행."""
    initial_state = {
        "snapshot_date": input_data["snapshot_date"],
        "supplier": input_data["supplier"],
        "items": input_data["items"],
        "risk_level": input_data["items"][0].get("risk_level", "N/A") if input_data["items"] else "N/A",
        "iteration_count": 0,
        "correction_feedback": "",
        "is_valid_email": False
    }
    
    # 그래프 실행
    final_state = purchasing_graph.invoke(initial_state)
    return final_state
