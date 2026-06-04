"""
Session isolation tests for /query, /query/stream, /retrieve.

The load-bearing invariant: a query carrying a valid X-Session-Id reads
ONLY from that session's store. A query without the header reads ONLY from
the public registry. The `corpus` body field NEVER selects a session store
under any input, and an unknown/expired session token returns 404 — never
a silent fallback to the public default corpus.

Test strategy: replace api.main.session_store and api.main.orchestrator
with fakes so we can assert exactly which store the orchestrator received.
The FakeOrchestrator records its store_override argument and emits a
retrieval candidate whose excerpt names the store — tests then assert
which sentinel appears in the response. This proves *routing*, not
end-to-end retrieval correctness (covered by other tests).

TestClient is constructed WITHOUT the `with` context manager so lifespan
startup is skipped. Tests register their own corpora and session records.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from vectorstore.chroma_store import ChromaVectorStore
from vectorstore.vector_store import (
    BaseVectorStore,
    get_corpus,
    register_corpus,
    reset_corpus_registry,
)
from vectorstore.session_store import SessionRecord, SessionStore


SESSION_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{40,}")


@dataclass
class _SentinelStore:
    """Stand-in for a vector store. Carries a sentinel string so tests can
    assert which store the orchestrator was routed to. Not a real
    BaseVectorStore — the FakeOrchestrator never calls its methods.
    """
    sentinel: str


class _CapturingOrchestrator:
    """Records every orchestrator call so tests can assert exactly which
    store_override the handler passed in, and emits a response whose
    retrieval_candidates carry the routed store's sentinel."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def _sentinel_for(self, body, store_override) -> str:
        if store_override is not None:
            return store_override.sentinel
        # Mirror the real executor's public-path KeyError so the handler's
        # 400 translation fires for unknown corpora. The real path goes
        # through retrieval_executor.execute -> get_corpus(corpus).
        if body.corpus != "default":
            get_corpus(body.corpus)  # raises KeyError on unknown
        return f"public:{body.corpus}"

    def _response(self, body, store_override, *, retrieval_only: bool) -> dict:
        sentinel = self._sentinel_for(body, store_override)
        candidates = [{
            "rank": 0,
            "doc_id": "doc-stub",
            "chunk_id": "chunk-stub",
            "source": sentinel,
            "score": 1.0,
            "pre_rerank_score": 1.0,
            "used_in_answer": True,
            "excerpt": f"FROM_STORE:{sentinel}",
        }]
        if retrieval_only:
            return {
                "query_type": "factual",
                "strategy_used": "hybrid",
                "latency_ms": 1.0,
                "stage_timings": {"router_ms": 0.0, "retrieve_ms": 0.0, "rerank_ms": 0.0, "total_ms": 0.0},
                "retrieval_candidates": candidates,
                "top_rerank_score": 1.0,
            }
        return {
            "answer": f"answer-from:{sentinel}",
            "citations": [],
            "query_type": "factual",
            "strategy_used": "hybrid",
            "model_used": "stub",
            "total_tokens": 0,
            "latency_ms": 1.0,
            "cached": False,
            "retrieval_candidates": candidates,
        }

    async def query(self, body, *, store_override=None):
        self.calls.append({"method": "query", "corpus": body.corpus, "store_override": store_override})
        return self._response(body, store_override, retrieval_only=False)

    async def retrieve_only(self, body, *, store_override=None):
        self.calls.append({"method": "retrieve_only", "corpus": body.corpus, "store_override": store_override})
        return self._response(body, store_override, retrieval_only=True)

    async def stream_query(self, body, *, store_override=None):
        self.calls.append({"method": "stream_query", "corpus": body.corpus, "store_override": store_override})
        sentinel = self._sentinel_for(body, store_override)
        yield f"FROM_STORE:{sentinel}"


def _seed_session(session_store: SessionStore, token: str, sentinel: str) -> None:
    """Inject a SessionRecord with a sentinel store directly into the store
    dict, bypassing the real ChromaVectorStore construction."""
    import time as _t
    record = SessionRecord(
        token=token,
        store=_SentinelStore(sentinel),  # type: ignore[arg-type]
        persist_dir=Path("/tmp/test-session-not-used"),
        created_at=_t.time(),
        last_access=_t.time(),
    )
    with session_store._lock:  # noqa: SLF001
        session_store._records[token] = record  # noqa: SLF001


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Wire a fresh app state: empty corpus registry with only a 'default',
    a fresh SessionStore, and the capturing orchestrator. Lifespan does
    NOT run (constructed without `with`)."""
    reset_corpus_registry()
    register_corpus(
        "default",
        ChromaVectorStore(persist_dir=str(tmp_path / "default"), collection_name="default"),
    )
    register_corpus(
        "apple_press_releases",
        ChromaVectorStore(persist_dir=str(tmp_path / "apple_pr"), collection_name="apple_press_releases"),
    )

    fresh_sessions = SessionStore(root=tmp_path / "sessions")
    capturing = _CapturingOrchestrator()

    from api import main as api_main
    monkeypatch.setattr(api_main, "session_store", fresh_sessions)
    monkeypatch.setattr(api_main, "orchestrator", capturing)

    test_client = TestClient(api_main.app)
    test_client._sessions = fresh_sessions  # type: ignore[attr-defined]
    test_client._orchestrator = capturing  # type: ignore[attr-defined]
    return test_client


def _extract_response_sentinel(resp_json: dict) -> str:
    """Pull the sentinel the FakeOrchestrator stamped into the retrieval
    candidate excerpt — tells us which store was actually used."""
    return resp_json["retrieval_candidates"][0]["excerpt"]


# ─────────────────────────────────────────────────────────────────────────────
# A. Isolation: two sessions cannot reach each other
# ─────────────────────────────────────────────────────────────────────────────

def test_A1_session_a_does_not_see_session_b(client):
    _seed_session(client._sessions, "tokAAA" + "a" * 40, "sentinel-A")
    _seed_session(client._sessions, "tokBBB" + "b" * 40, "sentinel-B")
    token_a = "tokAAA" + "a" * 40

    resp = client.post(
        "/query",
        headers={"X-Session-Id": token_a},
        json={"query": "anything at all"},
    )
    assert resp.status_code == 200
    excerpt = _extract_response_sentinel(resp.json())
    assert "sentinel-A" in excerpt
    assert "sentinel-B" not in excerpt
    assert "public:" not in excerpt


def test_A2_header_swap_routes_to_header_owner(client):
    _seed_session(client._sessions, "tokAAA" + "a" * 40, "sentinel-A")
    _seed_session(client._sessions, "tokBBB" + "b" * 40, "sentinel-B")

    resp = client.post(
        "/query",
        headers={"X-Session-Id": "tokBBB" + "b" * 40},
        json={"query": "anything at all"},
    )
    assert resp.status_code == 200
    excerpt = _extract_response_sentinel(resp.json())
    assert "sentinel-B" in excerpt
    assert "sentinel-A" not in excerpt


def test_A3_corpus_field_with_session_token_value_is_ignored(client):
    """PRECEDENCE: X-Session-Id wins unconditionally. corpus=token_B must
    have ZERO effect. Assert POSITIVELY: A's sentinel is present, B's is
    absent. An empty result here would mean routing collapsed somewhere —
    that's a failure, not a pass."""
    _seed_session(client._sessions, "tokAAA" + "a" * 40, "sentinel-A")
    token_b = "tokBBB" + "b" * 40
    _seed_session(client._sessions, token_b, "sentinel-B")

    resp = client.post(
        "/query",
        headers={"X-Session-Id": "tokAAA" + "a" * 40},
        json={"query": "anything at all", "corpus": token_b},
    )
    assert resp.status_code == 200, resp.text
    excerpt = _extract_response_sentinel(resp.json())
    assert "sentinel-A" in excerpt, "session A's store must be routed to"
    assert "sentinel-B" not in excerpt, "session B's store MUST NOT be reached"
    assert "public:" not in excerpt, "public path must not be taken on session header"


def test_A4_corpus_field_with_valid_public_corpus_is_ignored(client):
    """Tighter precedence test: corpus names a REAL public corpus
    ('apple_press_releases'). With a session header, that corpus is still
    ignored — session A's store is used. A's sentinel present, the Apple
    public-path marker absent."""
    _seed_session(client._sessions, "tokAAA" + "a" * 40, "sentinel-A")

    resp = client.post(
        "/query",
        headers={"X-Session-Id": "tokAAA" + "a" * 40},
        json={"query": "anything at all", "corpus": "apple_press_releases"},
    )
    assert resp.status_code == 200, resp.text
    excerpt = _extract_response_sentinel(resp.json())
    assert "sentinel-A" in excerpt
    assert "public:apple_press_releases" not in excerpt
    assert "public:" not in excerpt


def test_A5_session_token_in_corpus_field_without_header_returns_400(client):
    """Without X-Session-Id, a session-shaped string in `corpus` falls
    through to the public registry and 400s as 'Unknown corpus'.

    The submitted value is echoed back ("Unknown corpus: '<value>'") — that
    is not a cross-session leak because the client sent the value to begin
    with. The isolation invariant under test is that OTHER sessions' tokens
    never appear in the 400 body (enumeration leak); the Registered list
    must contain only public corpora."""
    submitted_token = "tokAAA" + "a" * 40
    other_token = "tokBBB" + "b" * 40
    _seed_session(client._sessions, submitted_token, "sentinel-A")
    _seed_session(client._sessions, other_token, "sentinel-B")

    resp = client.post(
        "/query",
        json={"query": "anything at all", "corpus": submitted_token},
    )
    assert resp.status_code == 400
    body = resp.text
    # Other sessions MUST NOT appear (enumeration-leak guard).
    assert other_token not in body
    # The Registered list (everything after "Registered:") must not contain
    # any session-shaped token. Slice past the echoed submitted value first.
    detail = resp.json()["detail"]
    registered_segment = detail.split("Registered:", 1)[-1]
    assert not SESSION_TOKEN_RE.search(registered_segment), (
        f"session-shaped token leaked into Registered list: {registered_segment}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# B. No X-Session-Id — today's behavior unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_B1_no_header_routes_to_default(client):
    resp = client.post("/query", json={"query": "anything at all"})
    assert resp.status_code == 200
    excerpt = _extract_response_sentinel(resp.json())
    assert excerpt == "FROM_STORE:public:default"


def test_B2_no_header_with_apple_corpus(client):
    resp = client.post(
        "/query",
        json={"query": "anything at all", "corpus": "apple_press_releases"},
    )
    assert resp.status_code == 200
    excerpt = _extract_response_sentinel(resp.json())
    assert excerpt == "FROM_STORE:public:apple_press_releases"


def test_B3_unknown_public_corpus_400(client):
    resp = client.post(
        "/query",
        json={"query": "anything at all", "corpus": "nonexistent"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "nonexistent" in detail
    # Body must list only public corpora — no session tokens (none seeded
    # here, but the assertion guards regression of the enumeration leak).
    assert not SESSION_TOKEN_RE.search(detail)


# ─────────────────────────────────────────────────────────────────────────────
# C. Unknown/expired session — no silent fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_C1_unknown_session_returns_404(client):
    bogus = "never-minted-" + "x" * 40
    resp = client.post(
        "/query",
        headers={"X-Session-Id": bogus},
        json={"query": "anything at all"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found or expired"
    # 404 body MUST NOT contain the submitted token.
    assert bogus not in resp.text


def test_C2_unknown_session_PLUS_valid_corpus_still_404(client):
    """Header-presence commits to the session path. Even when corpus names
    a valid public corpus, an unknown session token returns 404 — never a
    silent public-corpus fallback. Repeat for 'default' and 'apple_*'."""
    bogus = "never-minted-" + "y" * 40

    for valid_corpus in ("default", "apple_press_releases"):
        resp = client.post(
            "/query",
            headers={"X-Session-Id": bogus},
            json={"query": "anything at all", "corpus": valid_corpus},
        )
        assert resp.status_code == 404, (
            f"corpus={valid_corpus!r} must NOT cause silent fallback; "
            f"got {resp.status_code}: {resp.text}"
        )
        assert resp.json()["detail"] == "Session not found or expired"


def test_C3_session_after_simulated_restart_is_404(client):
    """A token that was syntactically valid but no longer in the SessionStore
    (e.g. after a process restart that lost the in-memory dict) must 404,
    not fall back to a public corpus."""
    stale = "stale-token-" + "z" * 40
    # Nothing seeded → token is unknown.
    resp = client.post(
        "/query",
        headers={"X-Session-Id": stale},
        json={"query": "anything at all", "corpus": "default"},
    )
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# D. Enumeration leak — sessions never appear in 400 bodies
# ─────────────────────────────────────────────────────────────────────────────

def test_D1_unknown_corpus_400_never_lists_session_tokens(client):
    tokens = [f"sess-{i}-" + ("a" if i == 0 else "b" if i == 1 else "c") * 40 for i in range(3)]
    for i, t in enumerate(tokens):
        _seed_session(client._sessions, t, f"sentinel-{i}")

    resp = client.post(
        "/query",
        json={"query": "anything at all", "corpus": "zzz-nope"},
    )
    assert resp.status_code == 400
    body = resp.text
    for t in tokens:
        assert t not in body, f"session token {t!r} leaked into 400 body"
    assert not SESSION_TOKEN_RE.search(body)


def test_D1_same_on_retrieve(client):
    tokens = [f"sessR-{i}-" + ("a" if i == 0 else "b") * 40 for i in range(2)]
    for i, t in enumerate(tokens):
        _seed_session(client._sessions, t, f"sentinel-{i}")

    resp = client.post(
        "/retrieve",
        json={"query": "anything at all", "corpus": "zzz-nope"},
    )
    assert resp.status_code == 400
    for t in tokens:
        assert t not in resp.text


# ─────────────────────────────────────────────────────────────────────────────
# F. /query/stream — status committed before the stream opens
# ─────────────────────────────────────────────────────────────────────────────

def test_F1_stream_unknown_session_returns_404_not_sse(client):
    """404 lands at the HTTP boundary, NOT as a 200 SSE with an in-band
    [ERROR] event. content-type must be JSON, not text/event-stream."""
    bogus = "stream-bogus-" + "q" * 40
    resp = client.post(
        "/query/stream",
        headers={"X-Session-Id": bogus},
        json={"query": "anything at all"},
    )
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
    assert "text/event-stream" not in resp.headers["content-type"]
    body = resp.text
    assert "[ERROR]" not in body, "must not surface as in-band SSE error"
    assert resp.json()["detail"] == "Session not found or expired"


def test_F2_stream_precedence_session_wins_over_corpus(client):
    _seed_session(client._sessions, "tokSTRM" + "s" * 40, "sentinel-STRM")

    resp = client.post(
        "/query/stream",
        headers={"X-Session-Id": "tokSTRM" + "s" * 40},
        json={"query": "anything at all", "corpus": "apple_press_releases"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "FROM_STORE:sentinel-STRM" in body
    assert "public:apple_press_releases" not in body


def test_F3_stream_unknown_corpus_no_header_400(client):
    """No header, unknown public corpus → 400 BEFORE the stream opens.
    JSON body, not 200 SSE."""
    resp = client.post(
        "/query/stream",
        json={"query": "anything at all", "corpus": "zzz-nope"},
    )
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")
    assert "text/event-stream" not in resp.headers["content-type"]


# ─────────────────────────────────────────────────────────────────────────────
# G. /retrieve parity
# ─────────────────────────────────────────────────────────────────────────────

def test_G1_retrieve_session_precedence(client):
    _seed_session(client._sessions, "tokRET" + "r" * 40, "sentinel-RET")

    resp = client.post(
        "/retrieve",
        headers={"X-Session-Id": "tokRET" + "r" * 40},
        json={"query": "anything at all", "corpus": "apple_press_releases"},
    )
    assert resp.status_code == 200
    excerpt = resp.json()["retrieval_candidates"][0]["excerpt"]
    assert "sentinel-RET" in excerpt
    assert "public:apple_press_releases" not in excerpt


def test_G2_retrieve_unknown_session_404(client):
    bogus = "retrieve-bogus-" + "p" * 40
    resp = client.post(
        "/retrieve",
        headers={"X-Session-Id": bogus},
        json={"query": "anything at all", "corpus": "default"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found or expired"


# ─────────────────────────────────────────────────────────────────────────────
# E. Real-path round trip — the routed store IS the ingest store
# ─────────────────────────────────────────────────────────────────────────────
#
# E.1 deliberately does NOT use the _seed_session harness. The seeded tests
# above prove "the handler routes to the store I injected." E.1 proves the
# stronger property: the store the handler routes to is the SAME backing
# store that the real ingest pipeline wrote into. That's the property the
# harness cannot verify by construction.

@pytest.fixture
def real_path_client(tmp_path, monkeypatch):
    """Wire real ingest + real ChromaVectorStore + real RetrievalExecutor.
    No seeded sentinels. Reranking is disabled (cross-encoder model load is
    skipped); generation is stubbed out via constructor injection so no
    LLM API key is needed. The embedder DOES load on first ingest — that
    cost is paid once per pytest session."""
    from config.settings import settings as _settings
    monkeypatch.setattr(_settings, "enable_reranking", False)

    reset_corpus_registry()

    from orchestrator import RAGOrchestrator
    real_orch = RAGOrchestrator(generation=object())  # /retrieve never calls .generate()

    fresh_sessions = SessionStore(root=tmp_path / "sessions")

    from api import main as api_main
    monkeypatch.setattr(api_main, "session_store", fresh_sessions)
    monkeypatch.setattr(api_main, "orchestrator", real_orch)

    test_client = TestClient(api_main.app)
    test_client._sessions = fresh_sessions  # type: ignore[attr-defined]
    return test_client


def test_E1_real_ingest_then_retrieve_round_trip(real_path_client):
    """The store the query path routes to MUST be the same store the
    ingest path wrote into. Uses the real /ingest/text → /retrieve flow
    end-to-end: a session is minted by ingest, the sentinel content lands
    in a real ChromaVectorStore at the session's persist_dir, and the
    /retrieve call with the same X-Session-Id pulls the sentinel back.

    A regression that resolves the session record to a DIFFERENT
    ChromaVectorStore (e.g. wrong persist_dir, stale instance from a
    different test, accidental fallback to default) fails this test —
    the sentinel is in exactly one store and only the real round-trip
    finds it."""
    sentinel = "ROUNDTRIPSENTINELxyzzy42"

    ingest_resp = real_path_client.post(
        "/ingest/text",
        json={
            "text_content": (
                f"This document discusses {sentinel} extensively. "
                f"The {sentinel} appears in multiple sentences to ensure "
                f"BM25 indexing picks it up reliably."
            ),
            "metadata": {},
        },
    )
    assert ingest_resp.status_code == 200, ingest_resp.text
    token = ingest_resp.headers["X-Session-Id"]
    assert token, "ingest must mint and return a server-generated session token"

    retrieve_resp = real_path_client.post(
        "/retrieve",
        headers={"X-Session-Id": token},
        # Force semantic (vector) search: BM25 with a single-document corpus
        # has IDF=0 for terms appearing in 100% of docs, so the keyword path
        # cannot find the sentinel even though it was indexed. Vector search
        # exercises the same store via a different code path and works on a
        # single-doc corpus. The property under test (the routed store ==
        # the ingest store) is orthogonal to the strategy choice.
        json={"query": sentinel, "strategy_override": "semantic"},
    )
    assert retrieve_resp.status_code == 200, retrieve_resp.text
    candidates = retrieve_resp.json()["retrieval_candidates"]
    assert candidates, (
        "no chunks retrieved — the store ingest wrote to is NOT the store "
        "the query path routed to"
    )
    assert any(sentinel in c["excerpt"] for c in candidates), (
        f"sentinel {sentinel!r} missing from retrieved chunks: {candidates}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# H. Orchestrator never sees a session token via corpus
# ─────────────────────────────────────────────────────────────────────────────

def test_H1_orchestrator_never_receives_session_token_as_corpus(client):
    """Structural assertion: the orchestrator's request body sees an
    unmodified `corpus` string, BUT when session_override is set, the
    corpus is irrelevant. We assert that on the session path, the
    orchestrator received a non-None store_override — proving the session
    store is reached by reference, not by passing the token as a corpus
    string anywhere downstream."""
    token_a = "audit-tok-" + "a" * 40
    _seed_session(client._sessions, token_a, "sentinel-AUD")

    resp = client.post(
        "/query",
        headers={"X-Session-Id": token_a},
        json={"query": "anything at all", "corpus": "whatever-string"},
    )
    assert resp.status_code == 200

    call = client._orchestrator.calls[-1]
    assert call["method"] == "query"
    assert call["store_override"] is not None
    assert call["store_override"].sentinel == "sentinel-AUD"
    # corpus body passes through unchanged (defended by the orchestrator
    # via store_override taking precedence); the corpus string itself
    # NEVER reaches session_store.get() — the registry it hits is
    # _corpus_registry only, and that's hit only when store_override is
    # None. With store_override set, no lookup happens at all.
    assert call["corpus"] == "whatever-string"
