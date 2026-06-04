"""
Session-scoped ingest tests.

Load-bearing regression (test_fiqa_count_unchanged_after_session_ingest):
the default FiQA FAISS store's chunk count must NOT change after a session
ingest. If a future change accidentally re-wires the session ingest path
back to get_vector_store() (or the now-unused module-level
ingestion_pipeline), the default store grows and this test fails.

The other tests pin the session contract: header in/out, limit codes,
limit enforcement actually entered before the pipeline runs.
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from config.settings import settings
from vectorstore.session_store import SessionStore
from vectorstore.vector_store import get_vector_store


@pytest.fixture
def client_and_store(tmp_path, monkeypatch):
    """Enter the TestClient context so lifespan fires (constructing the
    real session_store from settings.ragcore_session_root), THEN override
    the module-level session_store with one rooted in tmp_path. The
    handlers look up `session_store` via the module namespace, so the
    monkeypatch is visible to them."""
    with TestClient(app) as c:
        store = SessionStore(root=tmp_path)
        monkeypatch.setattr(api_main, "session_store", store)
        yield c, store


def test_fiqa_count_unchanged_after_session_ingest(client_and_store):
    """THE regression: the default FiQA FAISS store must not be touched by
    a session ingest. If session ingest is re-wired to get_vector_store()
    or to the module-level ingestion_pipeline (which is bound to FiQA),
    the default store's chunk count grows and this test fails."""
    client, _ = client_and_store
    fiqa_count_before = get_vector_store().count()

    r = client.post("/ingest/text", json={"text_content": "some test text"})
    assert r.status_code == 200, r.text

    fiqa_count_after = get_vector_store().count()
    assert fiqa_count_after == fiqa_count_before, (
        f"FiQA was polluted: count went from {fiqa_count_before} to "
        f"{fiqa_count_after}. Session ingest must NOT touch the default store."
    )


def test_no_session_id_mints_token(client_and_store):
    """A request without X-Session-Id mints a fresh session and returns
    the token in the response header."""
    client, _ = client_and_store
    r = client.post("/ingest/text", json={"text_content": "hello"})
    assert r.status_code == 200, r.text
    assert "X-Session-Id" in r.headers
    assert len(r.headers["X-Session-Id"]) >= 32  # secrets.token_urlsafe(32) → 43 chars


def test_existing_session_id_writes_to_same_store(client_and_store):
    """Echoing the X-Session-Id from a first response on a second request
    routes the write to the same session-scoped ChromaVectorStore."""
    client, store = client_and_store
    text = "Apple Inc. designs and manufactures consumer electronics. " * 5
    r1 = client.post("/ingest/text", json={"text_content": text})
    token = r1.headers["X-Session-Id"]
    record = store.get(token)
    count_after_first = record.store.count()

    r2 = client.post(
        "/ingest/text",
        json={"text_content": text},
        headers={"X-Session-Id": token},
    )
    assert r2.status_code == 200, r2.text
    assert r2.headers["X-Session-Id"] == token  # echoed, not re-minted

    record = store.get(token)
    count_after_second = record.store.count()
    assert count_after_second > count_after_first, (
        "Second ingest did not write to the same session store"
    )


def test_file_over_max_size_rejected_pipeline_not_entered(client_and_store):
    """Per-file size guard rejects with 413 and the pipeline never runs.
    Spy on IngestionPipeline.ingest_file — any call fires AssertionError."""
    client, _ = client_and_store
    spy = AsyncMock(side_effect=AssertionError("ingest_file reached despite oversize"))

    # Payload just over the per-file cap but under the 10 MB body cap
    # (so the outer BodySize middleware doesn't catch it first).
    oversize = b"a" * (settings.ragcore_session_max_file_bytes + 1)

    with patch("ingestion.pipeline.IngestionPipeline.ingest_file", spy):
        r = client.post(
            "/ingest/file",
            files={"file": ("x.txt", io.BytesIO(oversize), "text/plain")},
        )

    assert r.status_code == 413, r.text
    spy.assert_not_called()


def test_max_files_per_session_rejected_with_readable_message(client_and_store):
    """N successful ingests, then (N+1)th → 409 with a human-readable
    detail naming the limit and the recovery action. The detail is what
    Streamlit will surface to the end user; a bare status would be a
    dead-end UX."""
    client, _ = client_and_store
    max_files = settings.ragcore_session_max_files

    r1 = client.post("/ingest/text", json={"text_content": "first"})
    assert r1.status_code == 200
    token = r1.headers["X-Session-Id"]

    for i in range(2, max_files + 1):
        ri = client.post(
            "/ingest/text",
            json={"text_content": f"file {i}"},
            headers={"X-Session-Id": token},
        )
        assert ri.status_code == 200, f"file {i} unexpectedly failed: {ri.text}"

    rfinal = client.post(
        "/ingest/text",
        json={"text_content": "overflow"},
        headers={"X-Session-Id": token},
    )
    assert rfinal.status_code == 409, rfinal.text
    detail = rfinal.json()["detail"]
    assert "limit" in detail.lower()
    assert str(max_files) in detail
    assert "new session" in detail.lower()


def test_max_concurrent_sessions_rejected(client_and_store):
    """Mint max_concurrent sessions; the next mint hits the cap and
    returns 503. Mints are triggered by omitting X-Session-Id."""
    client, _ = client_and_store
    max_concurrent = settings.ragcore_session_max_concurrent

    for i in range(max_concurrent):
        ri = client.post("/ingest/text", json={"text_content": f"sess-{i}"})
        assert ri.status_code == 200, ri.text

    rfinal = client.post("/ingest/text", json={"text_content": "overflow"})
    assert rfinal.status_code == 503, rfinal.text
