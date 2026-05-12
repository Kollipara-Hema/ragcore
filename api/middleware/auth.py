"""API key authentication middleware.

Activated when settings.auth_enabled is True. Checks the X-API-Key
header on every request against settings.api_key. Exempts health and
metrics endpoints. When auth_enabled is False, this middleware should
not be registered at all — see api/main.py.
"""
from __future__ import annotations

import secrets
from typing import Iterable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Paths exempt from auth — must remain reachable for monitoring and ops
EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
    "/metrics",
})


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require X-API-Key header on non-exempt routes."""

    def __init__(self, app, api_key: str, exempt_paths: Iterable[str] = EXEMPT_PATHS):
        super().__init__(app)
        if not api_key:
            raise ValueError(
                "APIKeyAuthMiddleware requires a non-empty api_key. "
                "Check settings.api_key when auth_enabled=True."
            )
        self._api_key = api_key
        self._exempt_paths = frozenset(exempt_paths)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        # Constant-time comparison to avoid timing oracle on the key
        if not secrets.compare_digest(provided, self._api_key):
            return JSONResponse(
                {"error": "Invalid or missing API key"},
                status_code=401,
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)
