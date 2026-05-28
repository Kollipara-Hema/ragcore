"""
Integration tests for FAISSVectorStore.count().

Mirrors the pair of count() tests in test_chroma_vector_store.py.
All tests use tmp_path — no shared disk state.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

from utils.models import Chunk
from vectorstore.vector_store import FAISSVectorStore


def _make_store(tmp_path) -> FAISSVectorStore:
    return FAISSVectorStore(
        index_file=str(tmp_path / "faiss_index.idx"),
        metadata_file=str(tmp_path / "faiss_metadata.pkl"),
    )


class TestFAISSCount:

    def test_count_empty_store_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        # No index file on disk, no upsert yet — index attribute is None.
        assert store.count() == 0

    def test_count_after_upsert_matches_chunk_total(self, tmp_path):
        store = _make_store(tmp_path)
        doc_id = uuid4()
        chunks = [
            Chunk(content="alpha", doc_id=doc_id, embedding=[1.0, 0.0, 0.0, 0.0]),
            Chunk(content="beta",  doc_id=doc_id, embedding=[0.0, 1.0, 0.0, 0.0]),
            Chunk(content="gamma", doc_id=doc_id, embedding=[0.0, 0.0, 1.0, 0.0]),
        ]
        asyncio.run(store.upsert(chunks))
        assert store.count() == 3
