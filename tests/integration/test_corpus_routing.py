"""Integration test for /query corpus routing.

Registers two Chroma corpora ("alpha" and "beta") with disjoint vocabularies
plus a "default" corpus with distinct content. Drives
RetrievalExecutor.execute() with the KEYWORD strategy (no embedder usage)
and verifies:

- corpus="alpha" returns only alpha-content
- corpus="beta" returns only beta-content
- corpus omitted (defaults to "default") returns only default-content
- corpus="unknown" raises KeyError cleanly (no fallback to a wrong corpus)

Chroma's collection-name rule requires 3-63 chars, alphanumeric / underscore /
hyphen — hence "alpha" / "beta" rather than single letters.

The embedder is mocked via monkeypatch to skip the 1.3 GB BGE model load —
KEYWORD strategy doesn't need embeddings, but RetrievalExecutor.__init__
still constructs one through get_embedder().
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from retrieval.router.query_router import RoutingDecision
from utils.models import Chunk, QueryType, RetrievalStrategy
from vectorstore.chroma_store import ChromaVectorStore
from vectorstore.vector_store import register_corpus, reset_corpus_registry


# Three disjoint vocabularies so cross-corpus leakage would be unambiguous.
_A_CONTENTS = (
    "the cat sat on the mat",
    "the dog ran across the field",
    "the fox jumps over the lazy hound",
)
_B_CONTENTS = (
    "quantum mechanics describes particle wave duality",
    "general relativity bends spacetime around mass",
    "electromagnetism unites electric and magnetic fields",
)
_DEFAULT_CONTENTS = (
    "photosynthesis converts sunlight into chemical energy",
    "mitochondria are the powerhouse of the cell",
    "DNA encodes genetic instructions for protein synthesis",
)


def _upsert(store: ChromaVectorStore, contents: tuple[str, ...]) -> None:
    doc_id = uuid4()
    chunks = [
        Chunk(content=text, doc_id=doc_id, embedding=[0.1 + 0.1 * i] * 8)
        for i, text in enumerate(contents)
    ]
    asyncio.run(store.upsert(chunks))


def _keyword_decision(query: str) -> RoutingDecision:
    return RoutingDecision(
        query_type=QueryType.LOOKUP,
        primary_strategy=RetrievalStrategy.KEYWORD,
        fallback_strategy=RetrievalStrategy.SEMANTIC,
        expanded_queries=[query],
        metadata_filter=None,
    )


@pytest.fixture
def executor_with_three_corpora(tmp_path, monkeypatch):
    """Register a, b, and default corpora; mock embedder; return executor."""
    # Mock get_embedder BEFORE RetrievalExecutor.__init__ runs.
    fake_embedder = AsyncMock()
    fake_embedder.embed_query = AsyncMock(return_value=[0.1] * 8)
    monkeypatch.setattr(
        "retrieval.strategies.retrieval_executor.get_embedder",
        lambda: fake_embedder,
    )

    # Each corpus gets its own persist_dir (BM25 sidecar isolation, per
    # ChromaVectorStore's class docstring).
    store_alpha = ChromaVectorStore(persist_dir=str(tmp_path / "alpha"), collection_name="alpha")
    store_beta = ChromaVectorStore(persist_dir=str(tmp_path / "beta"), collection_name="beta")
    store_default = ChromaVectorStore(persist_dir=str(tmp_path / "default"), collection_name="default")

    _upsert(store_alpha, _A_CONTENTS)
    _upsert(store_beta, _B_CONTENTS)
    _upsert(store_default, _DEFAULT_CONTENTS)

    reset_corpus_registry()  # belt-and-suspenders; conftest also clears post-test
    register_corpus("alpha", store_alpha)
    register_corpus("beta", store_beta)
    register_corpus("default", store_default)

    from retrieval.strategies.retrieval_executor import RetrievalExecutor
    return RetrievalExecutor()


def test_corpus_alpha_routes_to_store_alpha(executor_with_three_corpora):
    result = asyncio.run(
        executor_with_three_corpora.execute(_keyword_decision("cat"), top_k=5, corpus="alpha")
    )
    assert len(result.chunks) > 0
    contents = [rc.chunk.content for rc in result.chunks]
    assert any("cat" in c for c in contents)
    # Must NOT contain beta- or default-vocabulary terms
    joined = " ".join(contents)
    assert "quantum" not in joined
    assert "photosynthesis" not in joined


def test_corpus_beta_routes_to_store_beta(executor_with_three_corpora):
    result = asyncio.run(
        executor_with_three_corpora.execute(_keyword_decision("quantum"), top_k=5, corpus="beta")
    )
    assert len(result.chunks) > 0
    contents = [rc.chunk.content for rc in result.chunks]
    assert any("quantum" in c for c in contents)
    joined = " ".join(contents)
    assert "cat" not in joined
    assert "photosynthesis" not in joined


def test_corpus_omitted_routes_to_default(executor_with_three_corpora):
    """No corpus kwarg -> default parameter value 'default' -> default corpus."""
    result = asyncio.run(
        executor_with_three_corpora.execute(_keyword_decision("mitochondria"), top_k=5)
    )
    assert len(result.chunks) > 0
    contents = [rc.chunk.content for rc in result.chunks]
    assert any("mitochondria" in c for c in contents)
    joined = " ".join(contents)
    assert "cat" not in joined
    assert "quantum" not in joined


def test_unknown_corpus_raises_keyerror_cleanly(executor_with_three_corpora):
    """Unknown corpus must surface as KeyError, not get swallowed by the
    fallback path and produce a misleading empty-results response."""
    with pytest.raises(KeyError) as exc_info:
        asyncio.run(
            executor_with_three_corpora.execute(
                _keyword_decision("anything"), top_k=5, corpus="nonexistent",
            )
        )
    msg = str(exc_info.value)
    assert "nonexistent" in msg
    # Error names registered corpora so the caller can spot a typo.
    assert "alpha" in msg and "beta" in msg and "default" in msg
