"""
Cross-encoder reranker.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        """Rerank chunks for the query."""
        if not chunks:
            return {"before": [], "after": []}

        # Before scores
        before = [{"rank": c["rank"], "score": c["score"], "text": c["text_preview"]} for c in chunks]

        # Prepare pairs
        pairs = [[query, chunk["text_preview"]] for chunk in chunks]

        # Rerank
        scores = self.model.predict(pairs)

        # Sort by new scores
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        after = []
        for rank, (chunk, score) in enumerate(scored_chunks[:top_k]):
            after.append({
                "rank": rank + 1,
                "score": float(score),
                "text": chunk["text_preview"]
            })

        return {
            "before": before,
            "after": after
        }
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
