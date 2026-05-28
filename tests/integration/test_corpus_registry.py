"""Integration tests for the corpus registry.

The end-to-end "shim returns the same instance across two get_vector_store()
calls and ingestion is visible to retrieval" guarantee is exercised by
tests/integration/test_vectorstore_singleton.py — that file's tests now run
against the registry's "default" key. This file focuses on the registry API
surface itself: register/get/list/reset semantics, error path, and the
relationship between reset_vector_store() (default-only) and
reset_corpus_registry() (everything).
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

import pytest

from utils.models import Chunk, RetrievedChunk
from vectorstore.vector_store import (
    BaseVectorStore,
    get_corpus,
    get_vector_store,
    list_corpora,
    register_corpus,
    reset_corpus_registry,
    reset_vector_store,
)


class _DummyStore(BaseVectorStore):
    """Minimal BaseVectorStore for registry-API tests. No I/O."""

    async def upsert(self, chunks: List[Chunk]) -> int:
        return 0

    async def vector_search(self, query_embedding, top_k=20, metadata_filter=None) -> List[RetrievedChunk]:
        return []

    async def keyword_search(self, query, top_k=20, metadata_filter=None) -> List[RetrievedChunk]:
        return []

    async def hybrid_search(self, query, query_embedding, top_k=20, alpha=0.7, metadata_filter=None) -> List[RetrievedChunk]:
        return []

    async def delete_document(self, doc_id: UUID) -> int:
        return 0

    def ping(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with an empty registry."""
    reset_corpus_registry()
    yield
    reset_corpus_registry()


def test_register_and_get_corpus_returns_same_instance():
    store = _DummyStore()
    register_corpus("alpha", store)
    assert get_corpus("alpha") is store


def test_get_unknown_corpus_raises_keyerror_naming_registered_corpora():
    register_corpus("alpha", _DummyStore())
    register_corpus("beta", _DummyStore())
    with pytest.raises(KeyError) as exc_info:
        get_corpus("gamma")
    # Error must name the unknown corpus AND the registered set so callers
    # can correct the typo or notice a missing registration.
    msg = str(exc_info.value)
    assert "gamma" in msg
    assert "alpha" in msg and "beta" in msg


def test_list_corpora_returns_registered_names():
    register_corpus("a", _DummyStore())
    register_corpus("b", _DummyStore())
    assert sorted(list_corpora()) == ["a", "b"]


def test_register_corpus_overwrites_existing_entry():
    first = _DummyStore()
    second = _DummyStore()
    register_corpus("x", first)
    register_corpus("x", second)
    assert get_corpus("x") is second


def test_reset_corpus_registry_clears_all():
    register_corpus("a", _DummyStore())
    register_corpus("b", _DummyStore())
    register_corpus("default", _DummyStore())
    reset_corpus_registry()
    assert list_corpora() == []


def test_reset_vector_store_only_clears_default():
    """reset_vector_store() is the backward-compat name for the old singleton's
    reset. It must remove only the "default" key — leaving other registered
    corpora alone — so tests for non-default corpora aren't accidentally wiped
    by code that calls reset_vector_store() out of habit."""
    register_corpus("default", _DummyStore())
    register_corpus("other", _DummyStore())
    reset_vector_store()
    assert "default" not in list_corpora()
    assert "other" in list_corpora()


def test_get_vector_store_returns_explicitly_registered_default():
    """If someone has pre-registered "default", the shim returns it instead of
    lazy-constructing. Required for tests and for the Day 3 lifespan that
    registers FAISS-as-default explicitly."""
    store = _DummyStore()
    register_corpus("default", store)
    assert get_vector_store() is store
