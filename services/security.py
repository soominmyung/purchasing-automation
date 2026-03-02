from datetime import date
from typing import Dict, Tuple

from fastapi import Header, HTTPException, Request, Depends

from config import settings

# --- Rate Limiting storage ---
# { (ip, date): count }
_usage_cache: Dict[Tuple[str, date], int] = {}

async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    Simple API access token verification (Method A).
    Used for low-cost operations (e.g., Ingest) that don't incur OpenAI costs.
    """
    if settings.api_access_token and x_api_key != settings.api_access_token:
        raise HTTPException(status_code=403, detail="Invalid API Access Token")
    return True

async def verify_api_access(
    request: Request,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    1. Verify API access token (via verify_api_key dependency)
    2. Check per-IP daily rate limit (Method C) - used for costly operations (LLM calls).
    """
    # 2. Check rate limit
    client_ip = request.client.host if request.client else "unknown"
    today = date.today()
    key = (client_ip, today)
    
    current_usage = _usage_cache.get(key, 0)
    if current_usage >= settings.rate_limit_per_day:
        raise HTTPException(
            status_code=429, 
            detail=f"Daily request limit reached ({settings.rate_limit_per_day}). Please try again tomorrow."
        )
    
    # Increment usage count
    _usage_cache[key] = current_usage + 1
    return True
