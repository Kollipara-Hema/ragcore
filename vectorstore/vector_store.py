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
    async def delete_document(self, doc_id: UUID) -> int:
        raise NotImplementedError


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, index_file: str = "faiss_index.idx", metadata_file: str = "faiss_metadata.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = None

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def _to_retrieved(self, idx: int, score: float) -> RetrievedChunk:
        meta = self.metadata[idx]
        chunk = Chunk(
            content=meta.get("content", ""),
            doc_id=meta.get("doc_id"),
            chunk_id=meta.get("chunk_id"),
            chunk_index=meta.get("chunk_index", 0),
            metadata=meta,
        )
        return RetrievedChunk(chunk=chunk, score=score, strategy_used=RetrievalStrategy.SEMANTIC)

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
            result.append(self._to_retrieved(idx, float(score)))

        return result

    async def keyword_search(self, query: str, top_k: int = 20, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        # Not implemented; returning empty to avoid missing interface.
        return []

    async def delete_document(self, doc_id: UUID) -> int:
        original_count = len(self.metadata)
        self.metadata = [m for m in self.metadata if str(m.get("doc_id")) != str(doc_id)]
        if self.index is not None:
            self.index.reset()
        self.save()
        return original_count - len(self.metadata)


def get_vector_store(provider: VectorStoreProvider = None) -> BaseVectorStore:
    provider = provider or settings.vector_store_provider
    if provider == VectorStoreProvider.WEAVIATE:
        logger.warning("Weaviate provider selected but not implemented; falling back to FAISS")
    return FAISSVectorStore()
