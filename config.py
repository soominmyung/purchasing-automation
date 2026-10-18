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

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
