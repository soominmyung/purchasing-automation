"""
Purchasing Automation - FastAPI 앱.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings

# --- LangSmith (Observability) Setup ---
# MUST BE DONE BEFORE ANY OTHER IMPORTS
import os
import re

if settings.langchain_tracing_v2:
    # 알파뉴머릭과 언더바, 대시만 남기고 제거 (숨겨진 문자, 따옴표 완벽 차단)
    clean_key = re.sub(r'[^a-zA-Z0-9_-]', '', (settings.langchain_api_key or ""))
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_API_KEY"] = clean_key
    os.environ["LANGCHAIN_PROJECT"] = "purchasing-ai-v1"
    
    # --- LangSmith Debugging Section ---
    key_len = len(clean_key)
    masked_key = f"{clean_key[:4]}...{clean_key[-4:]}" if key_len > 8 else "INVALID_KEY"
    
    print(f"--- [LangSmith Debug Start] ---")
    print(f"Project: {os.environ.get('LANGCHAIN_PROJECT')}")
    print(f"Tracing: {os.environ.get('LANGCHAIN_TRACING_V2')}")
    print(f"Key Masked: {masked_key}")
    print(f"Key Length: {key_len}")
    print(f"--- [LangSmith Debug End] ---")

import langsmith as ls  # 이제 안전하게 임포트

from services.vector_store import get_vector_stores
from routers import pipeline, ingest, output


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 벡터스토어 초기화."""
    get_vector_stores()
    yield
    # shutdown 시 정리 (선택)


app = FastAPI(
    title="Purchasing Automation API",
    description="Asynchronous multi-agent pipeline for automated purchasing analysis and documentation.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- 보안 및 사용량 제한 (Rate Limiting) ---
# services/security.py 로 이동됨.

# --- CORS 설정 ---
# 포트폴리오의 편의성을 위해 모든 Origin 허용 (보안은 X-API-Key 토큰으로 수행)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # allow_origins=["*"] 인 경우 False여야 함
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router, prefix="/api", tags=["pipeline"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(output.router, prefix="/api", tags=["output"])


@app.get("/")
def root():
    return {"service": "Purchasing Automation", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(settings.openai_api_key),
        "langsmith_tracing_env": os.environ.get("LANGCHAIN_TRACING_V2"),
        "langsmith_project_env": os.environ.get("LANGCHAIN_PROJECT"),
        "langsmith_key_present": bool(os.environ.get("LANGCHAIN_API_KEY")),
        "settings_langsmith_tracing": settings.langchain_tracing_v2,
        "settings_langsmith_project": settings.langchain_project
    }
