"""설정: 환경변수 기반."""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """앱 설정."""

    # 보안: .env 또는 환경변수에서 로드 (기본값 없음)
    openai_api_key: Optional[str] = None
    
    # API 접근 토큰 (X-API-Key 헤더 확인용)
    api_access_token: Optional[str] = None

    # 벡터스토어: "memory" | "chroma"
    vector_store_backend: str = "memory"
    
    # Framer 등 외부 도메인 허용
    extra_cors_origins: str = ""
    
    # 보안: IP당 일일 요청 제한
    rate_limit_per_day: int = 5
    
    # 휘발성 서버 환경 대응: 생성 파일을 output/temp 에 저장하고 max_age 분 후 삭제
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
