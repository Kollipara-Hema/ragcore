"""Unit tests for rate limiting middleware."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.rate_limit import RateLimitMiddleware, EXEMPT_PATHS


def _make_client(
    max_requests: int,
    window_seconds: int,
    trust_proxy_headers: bool = False,
) -> TestClient:
    """Minimal FastAPI app with RateLimitMiddleware for isolated testing."""
    test_app = FastAPI()

    @test_app.get("/query")
    async def query():
        return {"ok": True}

    @test_app.get("/health/live")
    async def health_live():
        return {"status": "alive"}

    @test_app.get("/health/ready")
    async def health_ready():
        return {"status": "ready"}

    test_app.add_middleware(
        RateLimitMiddleware,
        max_requests=max_requests,
        window_seconds=window_seconds,
        trust_proxy_headers=trust_proxy_headers,
    )
    return TestClient(test_app)


def test_configurable_limits_applied():
    """Third request over the configured limit is rejected with 429."""
    client = _make_client(max_requests=2, window_seconds=60)
    assert client.get("/query").status_code == 200
    assert client.get("/query").status_code == 200
    assert client.get("/query").status_code == 429


def test_retry_after_present_and_positive_integer():
    """429 includes a Retry-After header that parses as a positive integer."""
    client = _make_client(max_requests=1, window_seconds=10)
    client.get("/query")  # consume the one allowed request
    response = client.get("/query")
    assert response.status_code == 429
    retry_after = response.headers.get("Retry-After")
    assert retry_after is not None
    assert int(retry_after) >= 1


def test_proxy_header_respected_when_enabled():
    """X-Forwarded-For first IP is used for bucketing when trust_proxy_headers=True."""
    client = _make_client(max_requests=2, window_seconds=60, trust_proxy_headers=True)
    xff = "1.2.3.4, 5.6.7.8"
    assert client.get("/query", headers={"X-Forwarded-For": xff}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": xff}).status_code == 200
    # 1.2.3.4 bucket is now full
    assert client.get("/query", headers={"X-Forwarded-For": xff}).status_code == 429
    # Request without XFF falls into testclient bucket — still allowed
    assert client.get("/query").status_code == 200


def test_proxy_header_ignored_when_disabled():
    """X-Forwarded-For is ignored when trust_proxy_headers=False (default)."""
    client = _make_client(max_requests=2, window_seconds=60, trust_proxy_headers=False)
    # Fill testclient's bucket with two requests (XFF is present but ignored)
    assert client.get("/query", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 200
    # Different XFF, but same testclient IP — still rate limited
    assert client.get("/query", headers={"X-Forwarded-For": "5.6.7.8"}).status_code == 429


def test_health_live_exempt():
    """/health/live bypasses rate limiting regardless of request count."""
    client = _make_client(max_requests=1, window_seconds=60)
    for _ in range(10):
        assert client.get("/health/live").status_code == 200


def test_health_ready_exempt():
    """/health/ready bypasses rate limiting regardless of request count."""
    client = _make_client(max_requests=1, window_seconds=60)
    for _ in range(10):
        assert client.get("/health/ready").status_code == 200


def test_counter_increments_on_429():
    """ragcore_rate_limit_rejected_total increments by 1 on a 429 response."""
    from monitoring.metrics import rate_limit_rejected

    client = _make_client(max_requests=1, window_seconds=60)
    client.get("/query")  # succeeds; bucket now full

    before = rate_limit_rejected.labels(path="/query")._value.get()
    response = client.get("/query")
    after = rate_limit_rejected.labels(path="/query")._value.get()

    assert response.status_code == 429
    assert after == before + 1


def test_env_vars_configure_limits(monkeypatch):
    """RAGCORE_RATE_LIMIT_* env vars are read by Settings."""
    monkeypatch.setenv("RAGCORE_RATE_LIMIT_MAX_REQUESTS", "2")
    monkeypatch.setenv("RAGCORE_RATE_LIMIT_WINDOW_SECONDS", "10")
    from config.settings import Settings

    fresh = Settings()
    assert fresh.ragcore_rate_limit_max_requests == 2
    assert fresh.ragcore_rate_limit_window_seconds == 10


def test_env_var_trust_proxy_headers(monkeypatch):
    """RAGCORE_TRUST_PROXY_HEADERS=true is read by Settings."""
    monkeypatch.setenv("RAGCORE_TRUST_PROXY_HEADERS", "true")
    from config.settings import Settings

    fresh = Settings()
    assert fresh.ragcore_trust_proxy_headers is True


def test_exempt_paths_frozenset_contains_health_endpoints():
    """EXEMPT_PATHS frozenset includes all four exempt paths."""
    assert "/health" in EXEMPT_PATHS
    assert "/health/live" in EXEMPT_PATHS
    assert "/health/ready" in EXEMPT_PATHS
    assert "/metrics" in EXEMPT_PATHS
