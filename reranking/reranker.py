"""
Reranking layer — improves retrieval precision by scoring (query, chunk) pairs.

Two backends:
  1. Cross-encoder (local): ms-marco-MiniLM — fast, free, good quality
  2. Cohere Rerank API: highest quality, low latency, paid

Reranking runs AFTER initial vector/hybrid retrieval (two-stage retrieval):
  Stage 1: Retrieve top-K candidates with fast ANN (large recall set)
  Stage 2: Rerank with cross-encoder, return top-N to LLM (high precision)
"""
from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from config.settings import settings
from utils.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...


class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder reranker using sentence-transformers.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22MB, runs on CPU)
    
    Tradeoff: Slightly slower than Cohere API but free and private.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_rerank, query, chunks, top_k)

    def _sync_rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        model = self._get_model()
        pairs = [(query, rc.chunk.content) for rc in chunks]
        scores = model.predict(pairs)

        for rc, score in zip(chunks, scores):
            rc.score = float(score)

        chunks.sort(key=lambda rc: rc.score, reverse=True)

        for rank, rc in enumerate(chunks[:top_k]):
            rc.rank = rank

        return chunks[:top_k]


class CohereReranker(BaseReranker):
    """
    Cohere's rerank-english-v3.0 API.
    Best quality; adds ~100-200ms latency per call.
    """

    def __init__(self, model: str = "rerank-english-v3.0"):
        self.model = model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            import cohere
            client = cohere.AsyncClient()
            docs = [rc.chunk.content for rc in chunks]
            response = await client.rerank(
                query=query,
                documents=docs,
                model=self.model,
                top_n=top_k,
            )
            reranked = []
            for i, result in enumerate(response.results):
                rc = chunks[result.index]
                rc.score = result.relevance_score
                rc.rank = i
                reranked.append(rc)
            return reranked
        except Exception as e:
            logger.warning("Cohere rerank failed: %s — returning original order", e)
            return chunks[:top_k]


class NoOpReranker(BaseReranker):
    """Passthrough — used when reranking is disabled."""
    async def rerank(self, query, chunks, top_k) -> list[RetrievedChunk]:
        for i, rc in enumerate(chunks[:top_k]):
            rc.rank = i
        return chunks[:top_k]


def get_reranker(enabled: bool = None) -> BaseReranker:
    if enabled is None:
        enabled = settings.enable_reranking
    if not enabled:
        return NoOpReranker()
    return CrossEncoderReranker()
