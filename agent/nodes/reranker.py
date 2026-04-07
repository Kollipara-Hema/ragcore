"""
Reranker node — re-scores retrieved chunks with a cross-encoder.
Wraps get_reranker(); updates reranked_chunks in AgentState.
"""
from __future__ import annotations

import logging

from agent.state import AgentState
from reranking.reranker import get_reranker
from utils.models import Chunk, RetrievedChunk, RetrievalStrategy
from uuid import UUID

logger = logging.getLogger(__name__)
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = get_reranker()
    return _reranker


def _dict_to_retrieved_chunk(d: dict) -> RetrievedChunk:
    chunk = Chunk(
        content=d["content"],
        doc_id=UUID(d["doc_id"]),
        chunk_id=UUID(d["chunk_id"]),
        metadata=d.get("metadata", {}),
    )
    return RetrievedChunk(
        chunk=chunk,
        score=d["score"],
        strategy_used=RetrievalStrategy(d.get("strategy_used", "semantic")),
        rank=d.get("rank", 0),
    )


async def reranker_node(state: AgentState) -> dict:
    """Rerank retrieved chunks by relevance to the original query."""
    chunks_dicts = state.get("retrieved_chunks", [])
    if not chunks_dicts:
        return {"reranked_chunks": [], "rerank_skipped": True}

    try:
        retrieved = [_dict_to_retrieved_chunk(d) for d in chunks_dicts]
        reranked = await _get_reranker().rerank(
            query=state["query"],
            chunks=retrieved,
            top_k=min(5, len(retrieved)),
        )
        result = [
            {
                "content": rc.chunk.content,
                "chunk_id": str(rc.chunk.chunk_id),
                "doc_id": str(rc.chunk.doc_id),
                "score": rc.score,
                "rank": rc.rank,
                "metadata": rc.chunk.metadata,
            }
            for rc in reranked
        ]
        return {"reranked_chunks": result, "rerank_skipped": False}
    except Exception as exc:
        logger.error("reranker_node failed: %s — passing through original chunks", exc)
        return {"reranked_chunks": chunks_dicts[:5], "rerank_skipped": True}
