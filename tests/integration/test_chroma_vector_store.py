"""
Integration tests for ChromaVectorStore.
All tests use tmp_path — no network, no shared disk state.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from utils.models import Chunk, RetrievalStrategy
from vectorstore.chroma_store import ChromaVectorStore


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_store_with_chunks(tmp_path) -> ChromaVectorStore:
    """
    ChromaVectorStore at tmp_path with three pre-indexed chunks.
    Three chunks ensure BM25 IDF > 0 (avoids log(1.5/1.5)=0 with only 2 docs).
    """
    store = ChromaVectorStore(persist_dir=str(tmp_path))
    doc_id = uuid4()
    chunks = [
        Chunk(content="the cat sat on the mat", doc_id=doc_id, embedding=[0.1] * 8),
        Chunk(content="machine learning and neural networks", doc_id=doc_id, embedding=[0.9] * 8),
        Chunk(content="deep learning transforms computer vision", doc_id=doc_id, embedding=[0.5] * 8),
    ]
    asyncio.run(store.upsert(chunks))
    return store


# ── Tests mirroring TestVectorStoreSearch ─────────────────────────────────────

class TestChromaVectorStore:

    def test_keyword_search_returns_results(self, tmp_path):
        store = _make_store_with_chunks(tmp_path)
        results = asyncio.run(store.keyword_search("cat mat", top_k=5))
        assert isinstance(results, list)
        assert len(results) > 0

    def test_keyword_search_ranks_relevant_chunk_first(self, tmp_path):
        store = _make_store_with_chunks(tmp_path)
        results = asyncio.run(store.keyword_search("cat mat", top_k=5))
        assert len(results) > 0
        assert results[0].chunk.content == "the cat sat on the mat"
        assert all(r.strategy_used == RetrievalStrategy.KEYWORD for r in results)

    def test_vector_search_returns_results(self, tmp_path):
        store = _make_store_with_chunks(tmp_path)
        results = asyncio.run(store.vector_search([0.1] * 8, top_k=5))
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_hybrid_search_returns_results(self, tmp_path):
        store = _make_store_with_chunks(tmp_path)
        results = asyncio.run(store.hybrid_search("cat mat", [0.1] * 8, top_k=5, alpha=0.7))
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(r.strategy_used == RetrievalStrategy.HYBRID for r in results)

    def test_keyword_search_empty_when_no_data(self, tmp_path):
        store = ChromaVectorStore(persist_dir=str(tmp_path))
        results = asyncio.run(store.keyword_search("anything"))
        assert results == []

    # ── Chroma-specific tests ─────────────────────────────────────────────────

    def test_persistence_survives_reload(self, tmp_path):
        _make_store_with_chunks(tmp_path)

        store2 = ChromaVectorStore(persist_dir=str(tmp_path))

        kw = asyncio.run(store2.keyword_search("cat mat", top_k=5))
        assert len(kw) > 0

        vec = asyncio.run(store2.vector_search([0.1] * 8, top_k=5))
        assert len(vec) > 0

    def test_bm25_corpus_ids_stay_aligned_after_upsert(self, tmp_path):
        store = _make_store_with_chunks(tmp_path)
        assert len(store._corpus_ids) == len(store._bm25_index._corpus)
        assert len(store._corpus_ids) == 3
        for cid in store._corpus_ids:
            assert isinstance(cid, str)

    def test_delete_document_removes_from_bm25(self, tmp_path):
        store = ChromaVectorStore(persist_dir=str(tmp_path))

        doc_id_a = uuid4()
        doc_id_b = uuid4()
        chunks = [
            Chunk(content="elephant safari africa wildlife", doc_id=doc_id_a, embedding=[0.1] * 8),
            Chunk(content="quantum physics particle wave duality", doc_id=doc_id_b, embedding=[0.9] * 8),
            Chunk(content="photosynthesis chlorophyll sunlight energy", doc_id=doc_id_b, embedding=[0.5] * 8),
            # Fourth chunk so 3 docs remain after deleting doc_id_a — BM25 IDF needs >=3 docs for score > 0.
            Chunk(content="thermodynamics entropy heat transfer laws", doc_id=doc_id_b, embedding=[0.4] * 8),
        ]
        asyncio.run(store.upsert(chunks))

        asyncio.run(store.delete_document(doc_id_a))

        deleted_results = asyncio.run(store.keyword_search("elephant", top_k=5))
        assert all("elephant" not in r.chunk.content for r in deleted_results)

        retained_results = asyncio.run(store.keyword_search("quantum physics", top_k=5))
        assert len(retained_results) > 0
        assert any("quantum" in r.chunk.content for r in retained_results)

    def test_corpus_ids_rebuilt_correctly_after_delete(self, tmp_path):
        store = ChromaVectorStore(persist_dir=str(tmp_path))

        doc_id_a = uuid4()
        doc_id_b = uuid4()
        chunks = [
            Chunk(content="alpha text here", doc_id=doc_id_a, embedding=[0.1] * 8),
            Chunk(content="beta content there", doc_id=doc_id_b, embedding=[0.9] * 8),
            Chunk(content="gamma words elsewhere", doc_id=doc_id_b, embedding=[0.5] * 8),
        ]
        asyncio.run(store.upsert(chunks))
        asyncio.run(store.delete_document(doc_id_a))

        # Length invariant must hold after delete
        assert len(store._corpus_ids) == len(store._bm25_index._corpus)
        # doc_id_a's chunk was removed; two remain
        assert len(store._corpus_ids) == 2

    def test_bm25_state_pkl_survives_reload(self, tmp_path):
        _make_store_with_chunks(tmp_path)

        store2 = ChromaVectorStore(persist_dir=str(tmp_path))

        assert len(store2._corpus_ids) == 3
        assert len(store2._bm25_index._corpus) == 3

    def test_delete_works_after_uuid_roundtrip(self, tmp_path):
        """Guards OQ-4: doc_id stored as str in Chroma must be matchable by UUID on delete."""
        store = ChromaVectorStore(persist_dir=str(tmp_path))

        doc_id = uuid4()
        other_doc_id = uuid4()
        chunks = [
            Chunk(content="sphinx of black quartz judge my vow", doc_id=doc_id, embedding=[0.1] * 8),
            Chunk(content="pack my box with five dozen liquor jugs", doc_id=doc_id, embedding=[0.2] * 8),
            Chunk(content="the quick brown fox jumps over lazy dog", doc_id=other_doc_id, embedding=[0.3] * 8),
        ]
        asyncio.run(store.upsert(chunks))

        removed = asyncio.run(store.delete_document(doc_id))
        assert removed == 2

        results = asyncio.run(store.keyword_search("sphinx", top_k=5))
        assert all("sphinx" not in r.chunk.content for r in results)

        assert store._collection.count() == 1
