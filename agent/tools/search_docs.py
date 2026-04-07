"""
Tool: search_docs — semantic search over the indexed document corpus.
Can be used by the agent to look up specific information mid-reasoning.
"""
from __future__ import annotations

from typing import Optional
from retrieval.strategies.retrieval_executor import RetrievalExecutor
from retrieval.router.query_router import RoutingDecision
from utils.models import QueryType, RetrievalStrategy

_executor = None


def _get_executor():
    global _executor
    if _executor is None:
        _executor = RetrievalExecutor()
    return _executor


async def search_docs(
    query: str,
    top_k: int = 5,
    strategy: str = "semantic",
    metadata_filter: Optional[dict] = None,
) -> list[dict]:
    """
    Search indexed documents and return the top_k most relevant chunks.

    Args:
        query: The search query string.
        top_k: Number of results to return.
        strategy: One of 'semantic', 'keyword', 'hybrid'.
        metadata_filter: Optional field-value filters (e.g. {"doc_type": "pdf"}).

    Returns:
        List of chunk dicts with 'content', 'source', 'score', 'doc_id'.
    """
    try:
        strat = RetrievalStrategy(strategy)
    except ValueError:
        strat = RetrievalStrategy.SEMANTIC

    decision = RoutingDecision(
        query_type=QueryType.SEMANTIC,
        primary_strategy=strat,
        fallback_strategy=RetrievalStrategy.SEMANTIC,
        metadata_filter=metadata_filter,
        expanded_queries=[query],
    )
    result = await _get_executor().execute(decision, top_k=top_k)
    return [
        {
            "content": rc.chunk.content,
            "source": rc.chunk.metadata.get("source", "unknown"),
            "doc_id": str(rc.chunk.doc_id),
            "score": round(rc.score, 4),
        }
        for rc in result.chunks
    ]
