"""
Integration tests for FAISSVectorStore.delete_document (Path 1 fix).

Mirrors the four delete assertions in test_chroma_vector_store.py,
plus a legacy-pickle migration test.
All tests use tmp_path — no shared disk state.
"""
from __future__ import annotations

import asyncio
import pickle
from uuid import uuid4

import numpy as np
import pytest

from utils.models import Chunk
from vectorstore.vector_store import FAISSVectorStore


def _make_store(tmp_path) -> FAISSVectorStore:
    return FAISSVectorStore(
        index_file=str(tmp_path / "faiss_index.idx"),
        metadata_file=str(tmp_path / "faiss_metadata.pkl"),
    )


class TestFAISSDeleteDocument:

    def test_delete_document_decrements_index_ntotal(self, tmp_path):
        store = _make_store(tmp_path)
        doc_id_a = uuid4()
        doc_id_b = uuid4()
        chunks = [
            Chunk(content="elephant safari africa", doc_id=doc_id_a, embedding=[1.0, 0.0, 0.0, 0.0]),
            Chunk(content="elephant savanna plains", doc_id=doc_id_a, embedding=[0.9, 0.1, 0.0, 0.0]),
            Chunk(content="quantum physics waves", doc_id=doc_id_b, embedding=[0.0, 1.0, 0.0, 0.0]),
            Chunk(content="quantum entanglement spin", doc_id=doc_id_b, embedding=[0.0, 0.9, 0.1, 0.0]),
        ]
        asyncio.run(store.upsert(chunks))
        assert store.index.ntotal == 4

        removed = asyncio.run(store.delete_document(doc_id_a))

        assert removed == 2
        assert store.index.ntotal == 2, (
            f"Expected ntotal=2 after deleting 2 chunks, got {store.index.ntotal}"
        )
        assert len(store.metadata) == 2

    def test_vector_search_empty_for_deleted_doc(self, tmp_path):
        store = _make_store(tmp_path)
        doc_id_a = uuid4()
        doc_id_b = uuid4()
        chunks = [
            Chunk(content="elephant safari africa", doc_id=doc_id_a, embedding=[1.0, 0.0, 0.0, 0.0]),
            Chunk(content="elephant savanna plains", doc_id=doc_id_a, embedding=[0.9, 0.1, 0.0, 0.0]),
            Chunk(content="quantum physics waves", doc_id=doc_id_b, embedding=[0.0, 1.0, 0.0, 0.0]),
            Chunk(content="quantum entanglement spin", doc_id=doc_id_b, embedding=[0.0, 0.9, 0.1, 0.0]),
        ]
        asyncio.run(store.upsert(chunks))
        asyncio.run(store.delete_document(doc_id_a))

        results = asyncio.run(store.vector_search([1.0, 0.0, 0.0, 0.0], top_k=5))
        assert all(str(r.chunk.doc_id) != str(doc_id_a) for r in results), (
            "Vector search returned a chunk from the deleted doc_id_a"
        )

    def test_keyword_search_empty_for_deleted_doc(self, tmp_path):
        store = _make_store(tmp_path)
        doc_id_a = uuid4()
        doc_id_b = uuid4()
        # Four chunks so three remain after deleting doc_id_a — BM25 IDF needs >=3 docs for score > 0.
        chunks = [
            Chunk(content="elephant safari africa wildlife", doc_id=doc_id_a, embedding=[1.0, 0.0, 0.0, 0.0]),
            Chunk(content="quantum physics particle wave duality", doc_id=doc_id_b, embedding=[0.0, 1.0, 0.0, 0.0]),
            Chunk(content="photosynthesis chlorophyll sunlight energy", doc_id=doc_id_b, embedding=[0.0, 0.0, 1.0, 0.0]),
            Chunk(content="thermodynamics entropy heat transfer laws", doc_id=doc_id_b, embedding=[0.0, 0.0, 0.0, 1.0]),
        ]
        asyncio.run(store.upsert(chunks))
        asyncio.run(store.delete_document(doc_id_a))

        results = asyncio.run(store.keyword_search("elephant", top_k=5))
        assert all("elephant" not in r.chunk.content for r in results)

    def test_surviving_chunks_still_searchable(self, tmp_path):
        store = _make_store(tmp_path)
        doc_id_a = uuid4()
        doc_id_b = uuid4()
        chunks = [
            Chunk(content="elephant safari africa wildlife", doc_id=doc_id_a, embedding=[1.0, 0.0, 0.0, 0.0]),
            Chunk(content="quantum physics particle wave duality", doc_id=doc_id_b, embedding=[0.0, 1.0, 0.0, 0.0]),
            Chunk(content="photosynthesis chlorophyll sunlight energy", doc_id=doc_id_b, embedding=[0.0, 0.0, 1.0, 0.0]),
            Chunk(content="thermodynamics entropy heat transfer laws", doc_id=doc_id_b, embedding=[0.0, 0.0, 0.0, 1.0]),
        ]
        asyncio.run(store.upsert(chunks))
        asyncio.run(store.delete_document(doc_id_a))

        vec_results = asyncio.run(store.vector_search([0.0, 1.0, 0.0, 0.0], top_k=5))
        assert len(vec_results) > 0
        assert any("quantum" in r.chunk.content for r in vec_results)

        kw_results = asyncio.run(store.keyword_search("quantum physics", top_k=5))
        assert len(kw_results) > 0
        assert any("quantum" in r.chunk.content for r in kw_results)


class TestFAISSLegacyPickleMigration:

    def test_load_migrates_legacy_pickle(self, tmp_path):
        index_file = str(tmp_path / "faiss_index.idx")
        metadata_file = str(tmp_path / "faiss_metadata.pkl")

        # Build a real store so we have a valid FAISS index on disk.
        store = FAISSVectorStore(index_file=index_file, metadata_file=metadata_file)
        doc_id = uuid4()
        asyncio.run(store.upsert([
            Chunk(content="alpha text", doc_id=doc_id, embedding=[1.0, 0.0, 0.0, 0.0]),
            Chunk(content="beta text", doc_id=doc_id, embedding=[0.0, 1.0, 0.0, 0.0]),
        ]))

        # Strip 'embedding' from the pickle to simulate a pre-fix legacy store.
        with open(metadata_file, "rb") as f:
            meta = pickle.load(f)
        for m in meta:
            m.pop("embedding", None)
        with open(metadata_file, "wb") as f:
            pickle.dump(meta, f)

        # Reload — should trigger one-time migration.
        store2 = FAISSVectorStore(index_file=index_file, metadata_file=metadata_file)

        assert all("embedding" in m for m in store2.metadata), (
            "Migration should have backfilled 'embedding' on all metadata entries"
        )
        # Values must match what the FAISS index stores at each row.
        for i, m in enumerate(store2.metadata):
            expected = store2.index.reconstruct(i)
            actual = np.array(m["embedding"], dtype=np.float32)
            np.testing.assert_allclose(actual, expected, rtol=1e-5)
