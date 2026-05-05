"""
Tests for structured logging configuration and request-ID middleware.

Capture approach: a logging.StreamHandler backed by StringIO, with the same
ProcessorFormatter installed by configure_logging(). This covers both the
stdlib-bridge path (logging.getLogger()) and the native-structlog path
(structlog.get_logger()) because both ultimately flush through the root
handler chain.
"""
from __future__ import annotations
import json
import logging
import re
from io import StringIO

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitoring.logging_config import configure_logging
from api.middleware.request_id import RequestIdMiddleware


_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
# Timestamp: YYYY-MM-DDTHH:MM:SS.<fractional>Z  (structlog 24.x uses Z suffix)
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$")


# ── Minimal app used only for middleware tests ─────────────────────────────────
_mw_app = FastAPI()
_mw_app.add_middleware(RequestIdMiddleware)


@_mw_app.get("/ping")
async def _ping():
    logging.getLogger("test.ping").info("ping_called")
    return {"ok": True}


# ── Fixture ────────────────────────────────────────────────────────────────────
@pytest.fixture()
def captured_logs():
    """
    Configure structlog for the test and redirect log output to a StringIO
    buffer. Yields the buffer; caller reads and parses it after the action.
    """
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    configure_logging()
    # configure_logging() installed StreamHandler(sys.stdout) + ProcessorFormatter.
    # Swap stdout for a StringIO buffer while keeping the same formatter.
    root = logging.getLogger()
    formatter = root.handlers[0].formatter
    buf = StringIO()
    capture = logging.StreamHandler(buf)
    capture.setFormatter(formatter)
    root.handlers.clear()
    root.addHandler(capture)
    yield buf
    root.handlers.clear()
    structlog.contextvars.clear_contextvars()


def _records(buf: StringIO) -> list[dict]:
    return [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_configure_logging_produces_valid_json(captured_logs):
    logging.getLogger("ragcore.test").info("hello_world")

    records = _records(captured_logs)
    assert len(records) == 1
    r = records[0]

    assert r["event"] == "hello_world"
    assert r["level"] == "info"
    assert r["logger"] == "ragcore.test"
    assert r["service"] == "ragcore"
    assert "environment" in r
    assert _TS_RE.match(r["timestamp"]), f"unexpected timestamp format: {r['timestamp']!r}"


def test_middleware_echoes_and_propagates_provided_request_id(captured_logs):
    with TestClient(_mw_app, raise_server_exceptions=True) as client:
        response = client.get("/ping", headers={"X-Request-Id": "req-test-abc"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test-abc"

    ping = [r for r in _records(captured_logs) if r.get("event") == "ping_called"]
    assert len(ping) >= 1
    assert all(r.get("request_id") == "req-test-abc" for r in ping)


def test_middleware_generates_uuid4_when_no_request_id_header(captured_logs):
    with TestClient(_mw_app, raise_server_exceptions=True) as client:
        response = client.get("/ping")

    assert response.status_code == 200
    echoed = response.headers.get("x-request-id", "")
    assert _UUID4_RE.match(echoed), f"response X-Request-Id is not a UUID4: {echoed!r}"

    ping = [r for r in _records(captured_logs) if r.get("event") == "ping_called"]
    assert len(ping) >= 1
    assert all(_UUID4_RE.match(r.get("request_id", "")) for r in ping)
