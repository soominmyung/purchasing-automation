"""
Purchasing Automation: the FastAPI app.
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
    # keep only the characters an API key can contain (stray spaces or quotes from the environment)
    clean_key = re.sub(r'[^a-zA-Z0-9_-]', '', (settings.langchain_api_key or ""))
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_API_KEY"] = clean_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langsmith_hide_content:
        os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
        os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"

import langsmith as ls

from services.vector_store import get_vector_stores
from routers import pipeline, ingest, output


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the history store at start-up."""
    get_vector_stores()
    yield


app = FastAPI(
    title="Purchasing Automation API",
    description="Asynchronous multi-agent pipeline for automated purchasing analysis and documentation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Access control and rate limiting: services/security.py.

# CORS: any origin; access is controlled by the API key header, not by origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False with allow_origins=["*"]
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
