"""
Access control for the API.

- verify_api_key: the X-API-Key header must match api_access_token. With no token set, every
  request passes (local development); set one wherever the API can be reached by others.
- verify_api_access: the key, then a daily limit per client address on the calls that run the
  models (rate_limit_per_day). The counts are held in memory, so they start again when the
  server restarts, and each server instance counts on its own.
"""
from datetime import date
from typing import Dict, Tuple

from fastapi import Header, HTTPException, Request, Depends

from config import settings
from services.run_record import client_ip

# { (client address, date): runs }
_usage_cache: Dict[Tuple[str, date], int] = {}


async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """The access token only: for calls that cost no model use (history intake, downloads)."""
    if settings.api_access_token and x_api_key != settings.api_access_token:
        raise HTTPException(status_code=403, detail="Invalid API Access Token")
    return True


async def verify_api_access(
    request: Request,
    api_key_valid: bool = Depends(verify_api_key)
):
    """The access token, then the daily limit per client address: for calls that run the models.
    The address is the one the run record keeps (client_ip), so behind a proxy (the Cloud Run
    demo, trust_forwarded_for) each user is counted, not the proxy."""
    key = (client_ip(request), date.today())
    current_usage = _usage_cache.get(key, 0)
    if current_usage >= settings.rate_limit_per_day:
        raise HTTPException(
            status_code=429,
            detail=f"Daily request limit reached ({settings.rate_limit_per_day}). Please try again tomorrow."
        )
    _usage_cache[key] = current_usage + 1
    return True
