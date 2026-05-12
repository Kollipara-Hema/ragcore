"""
Simple sliding-window rate limiter middleware for FastAPI.

State is process-local; under N uvicorn workers an attacker can make up to
N × max_requests requests per window before being throttled by any single worker.
A Redis-backed implementation would be needed to enforce the limit across workers,
which is out of scope for this demo.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from monitoring.metrics import rate_limit_rejected

EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
    "/metrics",
})


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: int = 60,
        trust_proxy_headers: bool = False,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.trust_proxy_headers = trust_proxy_headers
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _client_ip(self, request: Request) -> str:
        if self.trust_proxy_headers:
            xff = request.headers.get("x-forwarded-for", "")
            if xff:
                return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        client_ip = self._client_ip(request)
        now = time.monotonic()

        self._buckets[client_ip] = [t for t in self._buckets[client_ip] if now - t < self.window]
        bucket = self._buckets[client_ip]

        if len(bucket) >= self.max_requests:
            retry_after = max(1, math.ceil(self.window - (now - bucket[0]))) if bucket else 1
            rate_limit_rejected.labels(path=request.url.path).inc()
            return JSONResponse(
                {"error": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)
