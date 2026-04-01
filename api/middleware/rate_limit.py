"""Simple token-bucket rate limiter middleware for FastAPI."""
from __future__ import annotations
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in {"/health", "/metrics"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = self._buckets[client_ip]

        # Remove timestamps outside the window
        self._buckets[client_ip] = [t for t in bucket if now - t < self.window]

        if len(self._buckets[client_ip]) >= self.max_requests:
            return JSONResponse(
                {"error": "Rate limit exceeded. Try again later."},
                status_code=429,
            )

        self._buckets[client_ip].append(now)
        return await call_next(request)
