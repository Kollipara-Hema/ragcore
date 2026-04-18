"""
Integration tests for the FAISSVectorStore singleton.

These tests reproduce the exact production bug:
  - get_vector_store() used to construct a new FAISSVectorStore() on every call
  - Ingestion and retrieval would each hold separate in-memory instances
  - Writes via ingestion were invisible to retrieval's stale instance

Each test calls reset_vector_store() via the autouse fixture so the singleton
is clean at the start of every test.
"""
from __future__ import annotations
import asyncio
import tempfile
import os
import pytest
from uuid import uuid4


# ---------------------------------------------------------------------------
# Fixture: reset the singleton before every test for clean state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_singleton(tmp_path):
    """
    Point the singleton at temp files and reset it before (and after) each test
    so tests don't bleed into each other via shared disk files.
    """
    from vectorstore.vector_store import reset_vector_store
    import vectorstore.vector_store as vs_module

    # Reset before test
    reset_vector_store()

    # Override default file paths so tests don't touch faiss_index.idx in the repo root
    original_init = vs_module.FAISSVectorStore.__init__

    idx_file = str(tmp_path / "test.idx")
    meta_file = str(tmp_path / "test.pkl")

    def patched_init(self, index_file=None, metadata_file=None):
        original_init(self, index_file=idx_file, metadata_file=meta_file)

    vs_module.FAISSVectorStore.__init__ = patched_init

    yield

    # Reset after test
    reset_vector_store()
    vs_module.FAISSVectorStore.__init__ = original_init


# ---------------------------------------------------------------------------
# Helper: build a minimal Chunk with a deterministic embedding
# ---------------------------------------------------------------------------

def _make_chunk(content: str, dim: int = 128) -> object:
    from utils.models import Chunk
    doc_id = uuid4()
    chunk_id = uuid4()
    embedding = [0.1] * dim   # deterministic; same for all chunks in these tests
    return Chunk(
        content=content,
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        embedding=embedding,
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# Test 1: singleton identity
# ---------------------------------------------------------------------------

class TestGetVectorStoreSingleton:

    def test_get_vector_store_returns_same_instance(self):
        """Two calls to get_vector_store() must return the exact same object."""
        from vectorstore.vector_store import get_vector_store

        store_a = get_vector_store()
        store_b = get_vector_store()

        assert store_a is store_b, (
            "get_vector_store() returned two different instances. "
            "Ingestion and retrieval would operate on separate in-memory stores."
        )

    def test_reset_clears_singleton(self):
        """reset_vector_store() must cause the next call to construct a fresh instance."""
        from vectorstore.vector_store import get_vector_store, reset_vector_store

        store_a = get_vector_store()
        reset_vector_store()
        store_b = get_vector_store()

        assert store_a is not store_b, (
            "After reset_vector_store(), get_vector_store() should return a new instance."
        )


# ---------------------------------------------------------------------------
# Test 2: writes are visible to reads through the singleton
# ---------------------------------------------------------------------------

class TestIngestVisibleToRetrieval:

    def test_upserted_chunks_are_searchable_via_second_get_vector_store_call(self):
        """
        Reproduces the production bug:
          1. Simulate the ingestion path — get the store once, upsert a chunk.
          2. Simulate the retrieval path — call get_vector_store() again (a second,
             independent call as retrieval_executor does in __init__).
          3. Assert the search result contains the ingested content.

        With the old code (no singleton) this test would return 0 results.
        With the fix it must return >= 1 result.
        """
        from vectorstore.vector_store import get_vector_store
        from utils.models import RetrievalStrategy

        dim = 128
        chunk = _make_chunk("The transformer architecture uses self-attention", dim=dim)

        # ── Ingestion path ────────────────────────────────────────────────────
        ingestion_store = get_vector_store()          # as IngestionPipeline.__init__ does
        count = asyncio.run(ingestion_store.upsert([chunk]))
        assert count == 1, "upsert should report 1 chunk written"

        # ── Retrieval path ────────────────────────────────────────────────────
        retrieval_store = get_vector_store()          # as RetrievalExecutor.__init__ does
        assert retrieval_store is ingestion_store, "pre-condition: same instance"

        query_embedding = [0.1] * dim                 # identical vector → score = 1.0
        results = asyncio.run(
            retrieval_store.vector_search(query_embedding, top_k=5)
        )

        assert len(results) >= 1, (
            "vector_search returned 0 results after upsert. "
            "This is the singleton bug: ingestion and retrieval hold separate instances."
        )
        assert results[0].chunk.content == "The transformer architecture uses self-attention"

    def test_keyword_search_also_reflects_upserted_content(self):
        """BM25 index must also stay in sync — keyword_search should find upserted content."""
        from vectorstore.vector_store import get_vector_store

        dim = 128
        # BM25 IDF requires >= 2 docs to produce non-zero scores for shared terms.
        # Use 3 chunks: one target and two fillers with distinct vocabulary.
        target = _make_chunk("attention mechanism transformer architecture", dim=dim)
        filler1 = _make_chunk("gradient descent optimization algorithm", dim=dim)
        filler2 = _make_chunk("convolutional neural network image recognition", dim=dim)

        ingestion_store = get_vector_store()
        asyncio.run(ingestion_store.upsert([target, filler1, filler2]))

        retrieval_store = get_vector_store()
        results = asyncio.run(retrieval_store.keyword_search("attention transformer", top_k=3))

        assert len(results) >= 1, "keyword_search returned 0 results after upsert"
        top_content = results[0].chunk.content
        assert "attention" in top_content or "transformer" in top_content
