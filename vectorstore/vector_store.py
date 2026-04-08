"""
Vector store abstractions and FAISS implementation.
"""
from __future__ import annotations
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from uuid import UUID

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import settings, VectorStoreProvider
from utils.models import Chunk, RetrievedChunk, RetrievalStrategy

logger = logging.getLogger(__name__)


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


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, index_file: str = "faiss_index.idx", metadata_file: str = "faiss_metadata.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []
        self.load()

    def load(self) -> None:
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = None

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            self._rebuild_bm25()

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from current metadata corpus."""
        if not self.metadata:
            self._bm25 = None
            self._bm25_corpus = []
            return
        self._bm25_corpus = [m.get("content", "") for m in self.metadata]
        tokenized = [doc.lower().split() for doc in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)

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
        self._rebuild_bm25()
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
        if self._bm25 is None or not self.metadata:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        candidates = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            if metadata_filter and not all(self.metadata[idx].get(k) == v for k, v in metadata_filter.items()):
                continue
            candidates.append((idx, float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)
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
            self.index.reset()
        self._rebuild_bm25()
        self.save()
        return original_count - len(self.metadata)


def get_vector_store(provider: VectorStoreProvider = None) -> BaseVectorStore:
    provider = provider or settings.vector_store_provider
    if provider == VectorStoreProvider.WEAVIATE:
        logger.warning("Weaviate provider selected but not implemented; falling back to FAISS")
    return FAISSVectorStore()
