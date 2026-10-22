"""Configuration: environment variable based."""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Security: loaded from .env or environment variables (no defaults)
    openai_api_key: Optional[str] = None
    
    # API access token (X-API-Key header verification)
    api_access_token: Optional[str] = None

    # Model that checks each analysis ("provider:model"). From a different provider than the
    # model that writes the analysis (GPT-4o), so the analysis is never judged by its own
    # author or a model of the same family. Needs GEMINI_API_KEY.
    evaluator_model: str = "gemini:gemini-3.7-flash"
    # History given to the analysis agent (agreed with the purchasing team): documents from the
    # last 12 months before the snapshot date, at most 5 per supplier and 5 per item.
    history_window_months: int = 12
    history_docs_per_scope: int = 5
    # Which prompts/<version>/ folder the agents use; recorded with every run.
    prompt_version: str = "v4"
    # Run records (services/run_record.py): one JSON file per run on the server.
    run_records_enabled: bool = True
    run_records_dir: str = "runs"
    # True only behind a proxy that sets X-Forwarded-For (the Cloud Run demo). On the
    # company server the connection address is the requesting PC.
    trust_forwarded_for: bool = False


    # Vector store backend: "memory" | "chroma"
    vector_store_backend: str = "memory"
    
    # Allow external domains (e.g., Framer)
    extra_cors_origins: str = ""
    
    # Security: per-IP daily request limit
    rate_limit_per_day: int = 5
    
    # Ephemeral server support: save generated files to output/temp, delete after max_age
    use_temp_for_output: bool = False
    temp_output_max_age_minutes: int = 5

    # LangSmith: Observability (tracing)
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "purchasing-automation"
    # True wherever real company data runs: traces then carry structure, timings, token
    # counts and model names, but no prompt or response text. Error messages are still
    # sent, so exceptions must not quote data (see docs/observability.md).
    langsmith_hide_content: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
