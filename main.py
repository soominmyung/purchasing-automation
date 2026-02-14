"""
Purchasing Automation - FastAPI 앱.
n8n 'Purchasing Automation' 워크플로를 Python 기반으로 재구현.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
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
    description="n8n Purchasing Automation 워크플로 기반 구매 자동화 API (Python/FastAPI)",
    version="1.0.0",
    lifespan=lifespan,
)

# --- 보안 및 사용량 제한 (Rate Limiting) ---
# services/security.py 로 이동됨.

_cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
if settings.extra_cors_origins:
    _cors_origins = _cors_origins + [o.strip() for o in settings.extra_cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
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
    return {"status": "ok", "openai_configured": bool(settings.openai_api_key)}
