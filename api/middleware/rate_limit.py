"""
Simple sliding-window rate limiter middleware for FastAPI.

State is process-local; under N uvicorn workers an attacker can make up to
N × max_requests requests per window before being throttled by any single worker.
A Redis-backed implementation would be needed to enforce the limit across workers,
which is out of scope for this demo.
"""
from __future__ import annotations

import ipaddress
import math
import time
from collections import defaultdict

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from monitoring.metrics import rate_limit_rejected

logger = structlog.get_logger(__name__)

EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
    "/metrics",
})

# Sentinel bucket keys. Distinct so operational events (no XFF vs malformed XFF)
# don't get conflated when reading dashboards or logs.
NO_XFF_BUCKET = "__no_xff__"
MALFORMED_XFF_BUCKET = "__malformed_xff__"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: int = 60,
        trust_proxy_headers: bool = False,
        proxy_hop_count: int = 1,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.trust_proxy_headers = trust_proxy_headers
        self.proxy_hop_count = proxy_hop_count
        self._buckets: dict[str, list[float]] = defaultdict(list)
        # Dedupe — without it a misconfigured hop_count would warn on every
        # request. We want the signal once per distinct suspicious key.
        self._warned: set[str] = set()

    def _warn_once(self, key: str, event: str, **kwargs) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(event, **kwargs)

    def _client_ip(self, request: Request) -> str:
        if not self.trust_proxy_headers:
            return request.client.host if request.client else "unknown"

        xff = request.headers.get("x-forwarded-for", "")
        if not xff:
            self._warn_once(
                NO_XFF_BUCKET,
                "client_ip_no_xff_under_trust",
                hint="Request arrived without X-Forwarded-For while trust=true; "
                     "either platform is not setting XFF, or request bypassed "
                     "the proxy. All such requests share one bucket.",
            )
            return NO_XFF_BUCKET

        # Strip whitespace; drop empty segments produced by malformed inputs
        # like ", , 1.2.3.4" so they don't shift the hop_count math.
        segments = [s.strip() for s in xff.split(",") if s.strip()]

        if len(segments) < self.proxy_hop_count:
            self._warn_once(
                MALFORMED_XFF_BUCKET,
                "client_ip_xff_shorter_than_hop_count",
                xff=xff,
                segments_observed=len(segments),
                hop_count_configured=self.proxy_hop_count,
                hint="XFF contains fewer segments than configured hop_count. "
                     "Calibration likely wrong, or upstream proxy stripped XFF.",
            )
            return MALFORMED_XFF_BUCKET

        candidate = segments[-self.proxy_hop_count]
        try:
            addr = ipaddress.ip_address(candidate)
        except ValueError:
            self._warn_once(
                MALFORMED_XFF_BUCKET,
                "client_ip_xff_invalid_at_trusted_position",
                xff=xff,
                candidate=candidate,
                hop_count_configured=self.proxy_hop_count,
                hint="Segment at the trusted hop position is not a valid IP.",
            )
            return MALFORMED_XFF_BUCKET

        # A non-global address at the trusted-hop position usually means
        # hop_count is wrong (we resolved to an internal proxy, not the real
        # client). Calibration alarm.
        if not addr.is_global:
            self._warn_once(
                candidate,
                "client_ip_resolved_to_non_global",
                resolved_ip=candidate,
                hop_count_configured=self.proxy_hop_count,
                xff=xff,
                hint="Resolved client IP is private/loopback/link-local — "
                     "proxy_hop_count may be miscalibrated for this platform.",
            )

        return candidate

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
