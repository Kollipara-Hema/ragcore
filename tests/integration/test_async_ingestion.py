"""
Integration test: async ingestion happy path.

Requires a running Celery broker (Redis on localhost:6379) and a Celery
worker consuming from ingestion.pipeline:celery_app.  Skips cleanly when
the broker is not reachable.

In CI the broker is a Redis service container and the worker is started as
a background process before pytest runs (see .github/workflows/ci.yml).

Out of scope (not tested here):
    - Ingested document queryable via /query
    - Worker restart resilience / failure path
    - Intermediate state shapes (STARTED)
    - Temp file cleanup
"""
from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ── Skip guard ────────────────────────────────────────────────────────────────

def _broker_reachable() -> bool:
    """Return True if the Celery broker (Redis DB 1) responds to PING."""
    try:
        import redis as redis_lib
        r = redis_lib.from_url("redis://localhost:6379/1", socket_connect_timeout=2)
        r.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _broker_reachable(),
    reason="Celery broker not reachable; skipping async ingestion test.",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ── Test document ─────────────────────────────────────────────────────────────

_SAMPLE_TEXT = """\
The mitochondria is the powerhouse of the cell.
Photosynthesis converts light energy into chemical energy.
The speed of light in a vacuum is approximately 299,792,458 metres per second.
Water freezes at zero degrees Celsius under standard atmospheric pressure.
The human body contains approximately 37 trillion cells.
DNA is a double helix composed of four nucleotide bases.
The Earth orbits the Sun at an average distance of 149.6 million kilometres.
Gravity causes objects with mass to attract one another.
The periodic table organises elements by atomic number.
An algorithm is a finite sequence of well-defined instructions.
"""


# ── Test ──────────────────────────────────────────────────────────────────────

def test_async_ingest_happy_path(client):
    """
    POST /ingest/file async_processing=true → PENDING response with job_id.
    GET /ingest/status/{job_id} polls until SUCCESS (120-second budget).
    SUCCESS result contains doc_id, chunk_count > 0, status="indexed".

    The 120-second budget accommodates a cold BGE model download on first
    CI run; subsequent runs hit the HuggingFace cache.

    The PENDING ambiguity (Celery returns PENDING for both genuinely-queued
    and unknown task IDs) is why the CI job runs a worker-readiness check
    before this test — it rules out the "worker absent" false-PENDING case.
    """
    # ── 1. Submit file ─────────────────────────────────────────────────────
    response = client.post(
        "/ingest/file",
        files={"file": ("science.txt", io.BytesIO(_SAMPLE_TEXT.encode()), "text/plain")},
        data={"async_processing": "true"},
    )
    assert response.status_code == 200, f"Unexpected status: {response.text}"

    body = response.json()
    job_id = body["job_id"]
    assert job_id, "Response must contain a non-empty job_id"
    assert body["status"] == "pending", f"Expected pending, got {body['status']}"

    # ── 2. Poll until SUCCESS or FAILURE (120-second budget) ───────────────
    _TIMEOUT_S = 120
    _POLL_INTERVAL_S = 2
    deadline = time.monotonic() + _TIMEOUT_S
    state = None
    status_body: dict = {}

    while time.monotonic() < deadline:
        status_resp = client.get(f"/ingest/status/{job_id}")
        assert status_resp.status_code == 200, (
            f"Status endpoint error: {status_resp.text}"
        )
        status_body = status_resp.json()
        state = status_body["state"]

        if state in ("SUCCESS", "FAILURE"):
            break

        time.sleep(_POLL_INTERVAL_S)
    else:
        pytest.fail(
            f"Async ingestion did not complete within {_TIMEOUT_S}s. "
            f"Last observed state: {status_body.get('state', 'unknown')}"
        )

    # ── 3. Assert outcome ──────────────────────────────────────────────────
    result = status_body.get("result")
    assert state == "SUCCESS", (
        f"Expected SUCCESS, got {state}. Result: {result}"
    )
    assert isinstance(result, dict), (
        f"SUCCESS result should be a dict, got: {type(result)}"
    )
    assert "doc_id" in result, f"Result missing doc_id: {result}"
    assert "chunk_count" in result, f"Result missing chunk_count: {result}"
    assert result.get("status") == "indexed", (
        f"Expected status=indexed, got {result.get('status')}"
    )
    assert isinstance(result["chunk_count"], int) and result["chunk_count"] > 0, (
        f"chunk_count should be a positive int, got {result['chunk_count']}"
    )
