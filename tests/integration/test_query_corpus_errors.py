"""
HTTP-level integration tests for unknown-corpus error translation.

Covers two paths that differ by HTTP-semantic necessity:
- POST /query catches KeyError from the executor and returns 400.
- POST /query/stream pre-checks corpus membership before constructing
  the StreamingResponse, because once streaming begins the HTTP status
  is committed as 200 and a mid-stream KeyError can only be surfaced as
  an in-band [ERROR] event, not as the 400 the caller deserves.

The executor-layer KeyError contract is covered by
test_corpus_routing.py::test_unknown_corpus_raises_keyerror_cleanly;
this file covers the API translation layer specifically.

TestClient is constructed WITHOUT the `with` context manager so the
lifespan startup does not run. Each test registers its own corpora.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from vectorstore.chroma_store import ChromaVectorStore
from vectorstore.vector_store import register_corpus, reset_corpus_registry


@pytest.fixture
def client_with_only_default(tmp_path, monkeypatch):
    """Register only 'default' so any other corpus is unknown."""
    reset_corpus_registry()
    default_store = ChromaVectorStore(
        persist_dir=str(tmp_path / "default"), collection_name="default",
    )
    register_corpus("default", default_store)

    from api.main import app
    return TestClient(app)


class _FakeOrchestratorRaisingKeyError:
    """Minimal stand-in for RAGOrchestrator. .query() raises the same
    KeyError shape get_corpus() produces for an unknown corpus, so the
    /query handler's translation can be tested without standing up the
    real orchestrator (which loads ML models at construction time)."""

    async def query(self, request):
        raise KeyError(
            f"Unknown corpus: {request.corpus!r}. Registered: ['default']"
        )


def test_query_unknown_corpus_returns_400(client_with_only_default, monkeypatch):
    monkeypatch.setattr("api.main.orchestrator", _FakeOrchestratorRaisingKeyError())

    resp = client_with_only_default.post(
        "/query", json={"query": "anything at all", "corpus": "bogus"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "bogus" in detail
    assert "default" in detail  # error names the registered set


def test_query_known_corpus_does_not_translate_keyerror(
    client_with_only_default, monkeypatch,
):
    """A KeyError raised for a *known* corpus (e.g. from a missing dict
    entry inside orchestrator) must NOT be translated to 400 — the 400
    contract is specifically for unknown-corpus, and the handler relies
    on get_corpus's curated message naming the registered set. Verify
    that a 'default' corpus request still reaches the orchestrator."""

    called = {"value": False}

    class _DefaultOnlyOrch:
        async def query(self, request):
            called["value"] = True
            # Real orchestrator would return a QueryResponse-shaped dict.
            return {
                "answer": "ok", "citations": [], "query_type": "lookup",
                "strategy_used": "keyword", "model_used": "stub",
                "total_tokens": 0, "latency_ms": 0.0, "cached": False,
            }

    monkeypatch.setattr("api.main.orchestrator", _DefaultOnlyOrch())
    resp = client_with_only_default.post(
        "/query", json={"query": "anything at all"},  # corpus omitted -> default
    )
    assert resp.status_code == 200
    assert called["value"] is True


def test_query_stream_unknown_corpus_returns_400(client_with_only_default):
    """Pre-check fires before StreamingResponse is constructed, so the
    400 lands at the HTTP boundary, not as an in-stream [ERROR] event."""
    resp = client_with_only_default.post(
        "/query/stream", json={"query": "anything at all", "corpus": "bogus"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "bogus" in detail
    assert "default" in detail
    # Confirm the response is NOT a text/event-stream — the pre-check
    # rejects the request before SSE setup.
    assert "text/event-stream" not in resp.headers.get("content-type", "")


def test_query_stream_default_corpus_passes_precheck(client_with_only_default, monkeypatch):
    """corpus='default' must pass the pre-check even when not present in
    list_corpora() output — the existing default-corpus shim handles it."""

    # Make the pre-check fall back to the get_vector_store shim path by
    # simulating a registry that does NOT explicitly list 'default'.
    monkeypatch.setattr("api.main.list_corpora", lambda: [])

    class _NoOpStream:
        async def stream_query(self, request):
            yield "hello"

    monkeypatch.setattr("api.main.orchestrator", _NoOpStream())

    resp = client_with_only_default.post(
        "/query/stream", json={"query": "anything at all"},  # corpus omitted -> default
    )
    # Stream starts -> 200, not 400.
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
