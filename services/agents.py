"""
에이전트: Analysis, Report Doc, PR Draft, PR Doc, Email Draft.
n8n LangChain Agent 노드 동작을 OpenAI + LangChain으로 구현.
"""
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
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
