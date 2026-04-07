"""
Retriever node — fetches relevant chunks from the vector/keyword index.
Wraps RetrievalExecutor; updates retrieved_chunks in AgentState.
"""
from __future__ import annotations

import logging
import time

from agent.state import AgentState
from retrieval.strategies.retrieval_executor import RetrievalExecutor
from retrieval.router.query_router import RoutingDecision
from utils.models import QueryType, RetrievalStrategy

logger = logging.getLogger(__name__)
_executor = None


def _get_executor():
    global _executor
    if _executor is None:
        _executor = RetrievalExecutor()
    return _executor


async def retriever_node(state: AgentState) -> dict:
    """Retrieve document chunks using the strategy chosen by the router."""
    start = time.monotonic()
    try:
        decision = RoutingDecision(
            query_type=QueryType(state.get("query_type", "semantic")),
            primary_strategy=RetrievalStrategy(state.get("primary_strategy", "semantic")),
            fallback_strategy=RetrievalStrategy(state.get("fallback_strategy", "semantic")),
            metadata_filter=state.get("metadata_filter"),
            expanded_queries=state.get("expanded_queries", [state["query"]]),
            reasoning=state.get("routing_reasoning", ""),
        )
        result = await _get_executor().execute(decision, top_k=state.get("top_k", 20))
        chunks = [
            {
                "content": rc.chunk.content,
                "chunk_id": str(rc.chunk.chunk_id),
                "doc_id": str(rc.chunk.doc_id),
                "score": rc.score,
                "rank": rc.rank,
                "metadata": rc.chunk.metadata,
                "strategy_used": rc.strategy_used.value,
            }
            for rc in result.chunks
        ]
        latency = (time.monotonic() - start) * 1000
        return {
            "retrieved_chunks": chunks,
            "retrieval_latency_ms": latency,
            "retrieval_fallback_used": result.fallback_used,
        }
    except Exception as exc:
        logger.error("retriever_node failed: %s", exc)
        latency = (time.monotonic() - start) * 1000
        return {
            "retrieved_chunks": [],
            "retrieval_latency_ms": latency,
            "retrieval_fallback_used": False,
            "error": str(exc),
        }
