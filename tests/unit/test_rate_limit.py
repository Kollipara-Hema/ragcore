"""Unit tests for rate limiting middleware."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.rate_limit import (
    EXEMPT_PATHS,
    MALFORMED_XFF_BUCKET,
    NO_XFF_BUCKET,
    RateLimitMiddleware,
)


def _make_client(
    max_requests: int,
    window_seconds: int,
    trust_proxy_headers: bool = False,
    proxy_hop_count: int = 1,
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
        proxy_hop_count=proxy_hop_count,
    )
    return TestClient(test_app)


def _get_middleware(client: TestClient) -> RateLimitMiddleware:
    """Reach into Starlette's middleware stack to grab the limiter instance —
    needed for asserting against the internal warning-dedup set."""
    app = client.app
    # Traverse the middleware stack until we find the RateLimitMiddleware.
    current = app.middleware_stack
    while current is not None:
        if isinstance(current, RateLimitMiddleware):
            return current
        current = getattr(current, "app", None)
    raise AssertionError("RateLimitMiddleware not found in stack")


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


def test_spoofed_leftmost_does_not_change_bucket_under_trust():
    """The fix in action. Under the OLD leftmost parser, each spoofed leftmost
    is its own bucket key — an attacker rotates spoofed leftmosts to evade
    the limiter. Under the fixed rightmost-N parser, all three requests share
    bucket 1.2.3.4 (the rightmost = real client per the trusted proxy), so
    the third request hits the configured cap. This test was the leftmost
    parser's regression marker — it cannot pass against that implementation."""
    client = _make_client(
        max_requests=2, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    # All three requests share rightmost=1.2.3.4 but rotate the spoofed leftmost.
    assert client.get("/query", headers={"X-Forwarded-For": "7.7.7.7, 1.2.3.4"}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": "8.8.8.8, 1.2.3.4"}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": "9.9.9.9, 1.2.3.4"}).status_code == 429


def test_single_hop_single_segment_xff():
    """hop_count=1 with one XFF segment — basic case, real client is the only segment."""
    client = _make_client(
        max_requests=1, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    assert client.get("/query", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 429
    # Different IP, fresh bucket.
    assert client.get("/query", headers={"X-Forwarded-For": "5.6.7.8"}).status_code == 200


def test_two_hop_three_segment_xff():
    """hop_count=2 (e.g. CDN + HF gateway). With XFF `<spoof>, <real>, <cdn>`,
    the real client is segments[-2] = the middle segment."""
    client = _make_client(
        max_requests=1, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=2,
    )
    # Both requests share real-client segments[-2]=1.2.3.4; spoofed leftmost varies.
    assert client.get("/query", headers={"X-Forwarded-For": "7.7.7.7, 1.2.3.4, 9.9.9.9"}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": "8.8.8.8, 1.2.3.4, 9.9.9.9"}).status_code == 429
    # Different real client, fresh bucket.
    assert client.get("/query", headers={"X-Forwarded-For": "8.8.8.8, 5.6.7.8, 9.9.9.9"}).status_code == 200


def test_hop_count_exceeds_segments_yields_malformed_sentinel():
    """hop_count=2 but only one XFF segment — calibration mismatch.
    Falls into the malformed-XFF sentinel bucket."""
    client = _make_client(
        max_requests=1, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=2,
    )
    assert client.get("/query", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 200
    # Different XFF, same sentinel bucket → second request blocked.
    assert client.get("/query", headers={"X-Forwarded-For": "5.6.7.8"}).status_code == 429
    middleware = _get_middleware(client)
    assert MALFORMED_XFF_BUCKET in middleware._buckets


def test_no_xff_under_trust_yields_no_xff_sentinel():
    """trust=true + missing XFF → all such requests share the no-xff sentinel
    bucket. Distinct from the malformed sentinel so logs/dashboards can tell
    them apart."""
    client = _make_client(
        max_requests=1, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    assert client.get("/query").status_code == 200
    assert client.get("/query").status_code == 429
    middleware = _get_middleware(client)
    assert NO_XFF_BUCKET in middleware._buckets
    assert MALFORMED_XFF_BUCKET not in middleware._buckets


def test_malformed_xff_invalid_ip_yields_malformed_sentinel():
    """trust=true + non-IP at the trusted-hop position → malformed sentinel."""
    client = _make_client(
        max_requests=1, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    assert client.get("/query", headers={"X-Forwarded-For": "not-an-ip"}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": "also-not-an-ip"}).status_code == 429
    middleware = _get_middleware(client)
    assert MALFORMED_XFF_BUCKET in middleware._buckets


def test_ipv6_xff_accepted():
    """IPv6 addresses parse and bucket correctly."""
    client = _make_client(
        max_requests=1, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    assert client.get("/query", headers={"X-Forwarded-For": "2001:db8::1"}).status_code == 200
    assert client.get("/query", headers={"X-Forwarded-For": "2001:db8::1"}).status_code == 429
    # Different IPv6, fresh bucket.
    assert client.get("/query", headers={"X-Forwarded-For": "2001:db8::2"}).status_code == 200


def test_calibration_warning_on_private_ip_under_trust():
    """Non-global resolved IP fires the 'hop_count may be miscalibrated'
    warning. Asserts via the dedup set, since the warning is the operational
    signal the user requested."""
    client = _make_client(
        max_requests=10, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    # 10.0.0.5 is RFC1918 private → not is_global → warning fires.
    client.get("/query", headers={"X-Forwarded-For": "10.0.0.5"})
    middleware = _get_middleware(client)
    assert "10.0.0.5" in middleware._warned


def test_calibration_warning_dedupes_per_key():
    """The same suspicious key only logs once per process — without dedup,
    a misconfigured deploy would flood logs."""
    client = _make_client(
        max_requests=10, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    for _ in range(5):
        client.get("/query", headers={"X-Forwarded-For": "10.0.0.5"})
    middleware = _get_middleware(client)
    # One distinct warned key for the private IP, regardless of repeat hits.
    assert middleware._warned == {"10.0.0.5"}


def test_global_ipv4_does_not_trigger_calibration_warning():
    """Public IPs are the expected case under correct calibration; no warning."""
    client = _make_client(
        max_requests=10, window_seconds=60,
        trust_proxy_headers=True, proxy_hop_count=1,
    )
    client.get("/query", headers={"X-Forwarded-For": "8.8.8.8"})
    middleware = _get_middleware(client)
    assert middleware._warned == set()


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


def test_env_var_proxy_hop_count(monkeypatch):
    """RAGCORE_PROXY_HOP_COUNT is read by Settings; default is 1."""
    monkeypatch.setenv("RAGCORE_PROXY_HOP_COUNT", "2")
    from config.settings import Settings

    fresh = Settings()
    assert fresh.ragcore_proxy_hop_count == 2

    monkeypatch.delenv("RAGCORE_PROXY_HOP_COUNT")
    default = Settings()
    assert default.ragcore_proxy_hop_count == 1


def test_exempt_paths_frozenset_contains_health_endpoints():
    """EXEMPT_PATHS frozenset includes all four exempt paths."""
    assert "/health" in EXEMPT_PATHS
    assert "/health/live" in EXEMPT_PATHS
    assert "/health/ready" in EXEMPT_PATHS
    assert "/metrics" in EXEMPT_PATHS
