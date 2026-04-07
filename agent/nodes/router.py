"""
Router node — classifies the query and selects a retrieval strategy.
Wraps the existing QueryRouter; updates routing fields in AgentState.
"""
from __future__ import annotations

import logging

from agent.state import AgentState
from retrieval.router.query_router import QueryRouter

logger = logging.getLogger(__name__)
_router = None


def _get_router():
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


async def router_node(state: AgentState) -> dict:
    """Classify query type and choose retrieval strategy."""
    try:
        decision = await _get_router().route(
            query=state["query"],
            strategy_override=state.get("strategy_override"),
            metadata_filter=state.get("metadata_filter"),
        )
        return {
            "query_type": decision.query_type.value,
            "primary_strategy": decision.primary_strategy.value,
            "fallback_strategy": decision.fallback_strategy.value,
            "expanded_queries": decision.expanded_queries,
            "routing_reasoning": decision.reasoning,
            "metadata_filter": decision.metadata_filter,
            "metadata": {"strategy": decision.primary_strategy.value},
        }
    except Exception as exc:
        logger.error("router_node failed: %s", exc)
        return {
            "query_type": "semantic",
            "primary_strategy": "semantic",
            "fallback_strategy": "semantic",
            "expanded_queries": [state["query"]],
            "routing_reasoning": f"fallback due to error: {exc}",
            "metadata": {"strategy": "vector"},
        }
