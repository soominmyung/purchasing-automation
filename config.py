"""Configuration: environment variable based."""
from datetime import date
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
    # author or a model of the same family. Provisional until judge validation: Claude Haiku
    # since 2026-10-23, after gemini-3.7-flash took 30-109 s per call and then returned 504,
    # 503 and 429. Needs CLAUDE_API_KEY (or ANTHROPIC_API_KEY).
    # chosen by judge validation, 2026-10-24 (docs/judge_validation.md, docs/judge_selection.md): Haiku
    # thinking at high effort, its answer shaped like the scoring sheet; one call per output
    evaluator_model: str = "anthropic:claude-haiku-5-5?effort=high&format=schema"
    # An analysis that fails a blocking code check is sent back to the analysis agent with the
    # failures, up to this many times, before any document is written (services/agents.py).
    analysis_rewrites: int = 2
    # Time limits on every model call, so one call that never answers cannot hold up a run.
    # A judge call that runs out is retried once and then recorded as "Not checked"; the
    # documents do not wait for the judge (services/agents.py).
    llm_timeout_seconds: int = 60
    llm_retries: int = 1
    judge_timeout_seconds: int = 90
    judge_retries: int = 1
    # History given to the analysis agent (agreed with the purchasing team): documents from the
    # last 12 months before the snapshot date, at most 5 per supplier and 5 per item.
    history_window_months: int = 12
    history_docs_per_scope: int = 5
    # The analysis is as of the snapshot date, so the stock data must be current (stock moves
    # every day; outbound runs at weekends too; the snapshot is taken nightly at 23:00). A run
    # accepts a snapshot up to 3 days old, and the report shows its age beside the date.
    require_fresh_snapshot: bool = True
    max_snapshot_age_days: int = 3
    # The run date the snapshot's age is measured against. Fixed for now to the day after the
    # sample CSV's snapshot (050425.csv, 2025-04-05), so runs on that fixed past data pass the
    # check. In production this is left unset (None) and the run date is the actual date the
    # run is made.
    run_date: Optional[str] = "2025-04-06"
    # A self-hosted model served by vLLM (OpenAI-compatible API), used for a model named
    # "local:<served model name>": the SFT base-model screening on the RTX 5090 and, later, the
    # fine-tuned analysis model on the GCP L4 VM (docs/sft_plan.md).
    local_llm_url: str = "http://localhost:8000/v1"
    local_llm_api_key: Optional[str] = None
    local_llm_max_tokens: int = 6000
    # Which prompts/<version>/ folder the agents use; recorded with every run.
    prompt_version: str = "v5"
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


def run_date() -> date:
    """The date of the run: settings.run_date while runs use fixed past data; in production
    run_date is unset and this is the actual date the run is made."""
    return date.fromisoformat(settings.run_date) if settings.run_date else date.today()
