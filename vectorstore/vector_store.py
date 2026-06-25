"""
Vector store abstractions and FAISS implementation.
"""
from __future__ import annotations
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np

from config.settings import settings, VectorStoreProvider
from utils.models import Chunk, RetrievedChunk, RetrievalStrategy
from vectorstore.bm25_index import BM25Index
from vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Module-level corpus registry — replaces the previous singleton. Each corpus
# name maps to its own BaseVectorStore instance, allowing multiple backends
# (FAISS, Chroma) and multiple Chroma collections to coexist in one process.
# The "default" key preserves the singleton contract for callers that haven't
# been corpus-aware'd yet — they go through get_vector_store() which lazy-
# constructs and registers "default" on first call.
_corpus_registry: dict[str, "BaseVectorStore"] = {}


class BaseVectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks: List[Chunk]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def vector_search(self, query_embedding: List[float], top_k: int = 20, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    async def keyword_search(self, query: str, top_k: int = 20, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    async def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 20, alpha: float = 0.7, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    async def delete_document(self, doc_id: UUID) -> int:
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> None:
        """Raise RuntimeError with a one-line reason if the store is unreachable."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed chunks. 0 if the store is empty."""
        raise NotImplementedError


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, index_file: str = None, metadata_file: str = None):
        data_dir = Path(settings.faiss_data_dir)
        os.makedirs(data_dir, exist_ok=True)
        self.index_file = index_file or str(data_dir / "faiss_index.idx")
        self.metadata_file = metadata_file or str(data_dir / "faiss_metadata.pkl")
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._bm25_index: BM25Index = BM25Index()
        self.load()

    def load(self) -> None:
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = None

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            self._migrate_legacy_pickle()
            self._bm25_index.build([m.get("content", "") for m in self.metadata])

    def _migrate_legacy_pickle(self) -> None:
        """One-time migration: backfill 'embedding' into pre-fix metadata entries."""
        if not self.metadata or "embedding" in self.metadata[0]:
            return
        if self.index is None:
            logger.warning("Cannot migrate legacy pickle: FAISS index not loaded")
            return
        for i, m in enumerate(self.metadata):
            m["embedding"] = self.index.reconstruct(i).tolist()
        self.save()
        logger.info("Migrated %d metadata entries to include embeddings (one-time)", len(self.metadata))

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def _to_retrieved(self, idx: int, score: float, strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC) -> RetrievedChunk:
        meta = self.metadata[idx]
        chunk = Chunk(
            content=meta.get("content", ""),
            doc_id=meta.get("doc_id"),
            chunk_id=meta.get("chunk_id"),
            chunk_index=meta.get("chunk_index", 0),
            metadata=meta,
        )
        return RetrievedChunk(chunk=chunk, score=score, strategy_used=strategy)

    async def upsert(self, chunks: List[Chunk]) -> int:
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        meta = [
            {
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "embedding": list(chunk.embedding),
                **chunk.metadata,
            }
            for chunk in chunks
            if chunk.embedding is not None
        ]

        if not embeddings:
            return 0

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])

        self.index.add(vectors)
        self.metadata.extend(meta)
        self._bm25_index.build([m.get("content", "") for m in self.metadata])
        self.save()

        return len(vectors)

    async def vector_search(self, query_embedding: List[float], top_k: int = 20, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        result = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if metadata_filter and not all(self.metadata[idx].get(k) == v for k, v in metadata_filter.items()):
                continue
            result.append(self._to_retrieved(idx, float(score), RetrievalStrategy.SEMANTIC))

        return result

    async def keyword_search(self, query: str, top_k: int = 20, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        """BM25 sparse retrieval over the indexed corpus."""
        if not self.metadata:
            return []

        assert len(self._bm25_index._corpus) == len(self.metadata), (
            f"BM25 corpus length {len(self._bm25_index._corpus)} != metadata length {len(self.metadata)}"
        )

        raw_candidates = self._bm25_index.query(query, len(self.metadata))

        candidates = []
        for idx, score in raw_candidates:
            if metadata_filter and not all(self.metadata[idx].get(k) == v for k, v in metadata_filter.items()):
                continue
            candidates.append((idx, score))

        return [self._to_retrieved(idx, score, RetrievalStrategy.KEYWORD) for idx, score in candidates[:top_k]]

    async def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 20, alpha: float = 0.7, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        """
        Fuse vector search (alpha) and BM25 keyword search (1-alpha).
        BM25 scores are normalized to [0, 1] before fusion.
        """
        vector_results = await self.vector_search(query_embedding, top_k * 2, metadata_filter)
        keyword_results = await self.keyword_search(query, top_k * 2, metadata_filter)

        vector_scores = {str(rc.chunk.chunk_id): rc.score for rc in vector_results}
        keyword_scores_raw = {str(rc.chunk.chunk_id): rc.score for rc in keyword_results}

        # Normalize BM25 scores to [0, 1]
        max_kw = max(keyword_scores_raw.values(), default=1.0)
        keyword_scores = {k: v / max_kw for k, v in keyword_scores_raw.items()}

        # Build lookup for chunks from both result sets
        chunk_map = {str(rc.chunk.chunk_id): rc for rc in vector_results + keyword_results}
        all_ids = set(vector_scores) | set(keyword_scores)

        fused = {
            cid: alpha * vector_scores.get(cid, 0.0) + (1 - alpha) * keyword_scores.get(cid, 0.0)
            for cid in all_ids
        }

        sorted_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:top_k]
        results = []
        for rank, cid in enumerate(sorted_ids):
            rc = chunk_map[cid]
            rc.score = fused[cid]
            rc.rank = rank
            rc.strategy_used = RetrievalStrategy.HYBRID
            results.append(rc)

        return results

    async def delete_document(self, doc_id: UUID) -> int:
        original_count = len(self.metadata)
        self.metadata = [m for m in self.metadata if str(m.get("doc_id")) != str(doc_id)]
        if self.index is not None:
            dim = self.index.d
            if self.metadata:
                survivors = np.array([m["embedding"] for m in self.metadata], dtype=np.float32)
                faiss.normalize_L2(survivors)
                new_index = faiss.IndexFlatIP(dim)
                new_index.add(survivors)
                self.index = new_index
            else:
                self.index = faiss.IndexFlatIP(dim)
        self._bm25_index.build([m.get("content", "") for m in self.metadata])
        self.save()
        return original_count - len(self.metadata)

    def ping(self) -> None:
        if not Path(settings.faiss_data_dir).is_dir():
            raise RuntimeError(f"FAISS data directory missing: {settings.faiss_data_dir}")

    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0


def register_corpus(name: str, store: BaseVectorStore) -> None:
    """Register a vector store under a corpus name. Overwrites any prior entry."""
    _corpus_registry[name] = store


def get_corpus(name: str) -> BaseVectorStore:
    """Look up a registered store by corpus name. Raises KeyError on unknown.

    Use get_vector_store() instead if you want the lazy-constructed "default"
    corpus that preserves pre-registry behavior.
    """
    if name not in _corpus_registry:
        raise KeyError(
            f"Unknown corpus: {name!r}. Registered: {sorted(_corpus_registry)}"
        )
    return _corpus_registry[name]


def list_corpora() -> list[str]:
    """Return the names of all registered corpora."""
    return list(_corpus_registry)


def reset_corpus_registry() -> None:
    """Clear all registered corpora. For tests only."""
    _corpus_registry.clear()


def get_vector_store(provider: VectorStoreProvider = None) -> BaseVectorStore:
    """Return the "default" corpus. Lazy-constructs it on first call from
    settings.vector_store_provider so two calls return the same instance —
    the singleton contract the pre-registry implementation provided.
    """
    if "default" in _corpus_registry:
        return _corpus_registry["default"]
    provider = provider or settings.vector_store_provider
    if provider == VectorStoreProvider.CHROMA:
        store: BaseVectorStore = ChromaVectorStore()
    else:
        store = FAISSVectorStore()
    _corpus_registry["default"] = store
    return store


def reset_vector_store() -> None:
    """Remove the "default" corpus from the registry. For tests and future
    'clear store' operations only. Other corpora are unaffected — use
    reset_corpus_registry() to wipe everything."""
    _corpus_registry.pop("default", None)
