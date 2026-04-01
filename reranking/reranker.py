"""
Cross-encoder reranker.
"""
from __future__ import annotations
import logging
from typing import List

from sentence_transformers import CrossEncoder

from config.settings import settings
from utils.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker:
    """Base interface for all rerankers."""

    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, rc.chunk.content) for rc in chunks]
        scores = self.model.predict(pairs)

        for rc, score in zip(chunks, scores):
            rc.score = float(score)

        chunks.sort(key=lambda rc: rc.score, reverse=True)

        for rank, rc in enumerate(chunks[:top_k]):
            rc.rank = rank

        return chunks[:top_k]


class NoOpReranker(BaseReranker):
    """Passthrough — used when reranking is disabled."""

    async def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        for i, rc in enumerate(chunks[:top_k]):
            rc.rank = i
        return chunks[:top_k]


def get_reranker(enabled: bool = None) -> BaseReranker:
    if enabled is None:
        enabled = settings.enable_reranking
    if not enabled:
        return NoOpReranker()
    return CrossEncoderReranker()
