"""
Body-size middleware tests.

Load-bearing invariant: when a request body exceeds the cap, the downstream
handler / ingestion pipeline is NEVER entered. The 413 must come from the
outermost middleware on a single Content-Length compare, before any setup
cost is paid by anything inside it. A silent middleware reorder that moves
BodySize inward defeats the guard's purpose; that regression is what
test_oversized_body_does_not_reach_pipeline catches.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from config.settings import settings


_CAP = settings.ragcore_ingest_max_body_bytes


@pytest.fixture
def client():
    return TestClient(app)


def test_oversized_body_does_not_reach_pipeline(client):
    """
    The invariant: body over the cap → 413, ingestion pipeline never called.

    Spies on both ingest_file and ingest_text. If middleware order regresses
    (BodySize moves inward), or if the path scoping breaks, the handler
    runs and AsyncMock records the call — assertion fires.
    """
    spy_file = AsyncMock(side_effect=AssertionError("ingest_file reached"))
    spy_text = AsyncMock(side_effect=AssertionError("ingest_text reached"))

    with patch("api.main.ingestion_pipeline") as mock_pipeline:
        mock_pipeline.ingest_file = spy_file
        mock_pipeline.ingest_text = spy_text

        # Send a body genuinely larger than the cap. httpx computes
        # Content-Length from the payload, so Stage 1 (header compare)
        # fires before any byte reaches Starlette's body buffer.
        oversized = b"x" * (_CAP + 1)
        response = client.post(
            "/ingest/text",
            content=oversized,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413, response.text
    spy_file.assert_not_called()
    spy_text.assert_not_called()


def test_small_request_passes_middleware(client):
    """Body well under cap → middleware passes through; not a 413."""
    response = client.post(
        "/ingest/text",
        json={"text_content": "a small valid payload"},
    )
    # Outcome depends on real pipeline state, but must NOT be 413.
    assert response.status_code != 413


def test_text_content_over_pydantic_max_returns_422(client):
    """
    text_content > 1 MB pydantic max_length but body < 10 MB middleware cap
    → middleware passes, pydantic rejects with 422. Defense-in-depth: if
    middleware path scoping ever drifts off /ingest/text, this still blocks.
    """
    # 1.5 MB ASCII — over the 1 MB pydantic cap, under the 10 MB middleware cap.
    over_pydantic = "a" * 1_500_000
    response = client.post(
        "/ingest/text",
        json={"text_content": over_pydantic},
    )
    assert response.status_code == 422, response.text


def test_query_endpoint_not_scoped(client):
    """
    BodySize is scoped to /ingest/* — /query is unaffected. A normal-sized
    query body must not get a 413 even though it shares the app.
    """
    response = client.post("/query", json={"query": "test query"})
    assert response.status_code != 413
