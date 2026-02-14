from datetime import date
from typing import Dict, Tuple

from fastapi import Header, HTTPException, Request

from config import settings

# --- 사용량 제한 (Rate Limiting) 저장소 ---
# { (ip, date): count }
_usage_cache: Dict[Tuple[str, date], int] = {}

async def verify_api_access(
    request: Request,
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    1. API 접근 토큰 확인 (방식 A)
    2. IP당 일일 횟수 제한 확인 (방식 C)
    """
    # 1. 토큰 확인
    if settings.api_access_token and x_api_key != settings.api_access_token:
        raise HTTPException(status_code=403, detail="Invalid API Access Token")

    # 2. 사용량 제한 확인
    client_ip = request.client.host if request.client else "unknown"
    today = date.today()
    key = (client_ip, today)
    
    current_usage = _usage_cache.get(key, 0)
    if current_usage >= settings.rate_limit_per_day:
        raise HTTPException(
            status_code=429, 
            detail=f"Daily request limit reached ({settings.rate_limit_per_day}). Please try again tomorrow."
        )
    
    # 횟수 증가
    _usage_cache[key] = current_usage + 1
    return True
