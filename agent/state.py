"""
Agent state — the shared data structure passed between every graph node.

Every node receives the full AgentState and returns a partial dict
with only the fields it updated. LangGraph merges the updates.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from utils.models import QueryType, RetrievalStrategy


class AgentState(TypedDict, total=False):
    # ── Conversation ────────────────────────────────────────────────────────
    messages: list[dict]                # Full conversation turn history
    tool_calls: list[dict]              # Tool invocations made this turn

    # ── Input ──────────────────────────────────────────────────────────────
    query: str                          # Original user question
    metadata_filter: Optional[dict]     # Optional structured filters
    strategy_override: Optional[str]    # Force a specific retrieval strategy
    top_k: int                          # Chunks to retrieve (default: 20)
    metadata: dict                      # Arbitrary key-value metadata

    # ── Routing ────────────────────────────────────────────────────────────
    query_type: str                     # QueryType enum value
    primary_strategy: str               # RetrievalStrategy enum value
    fallback_strategy: str              # Fallback if primary fails
    expanded_queries: list[str]         # Paraphrased query variants
    routing_reasoning: str              # Why this strategy was chosen

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_chunks: list[dict]        # Raw chunks from vector store
    retrieval_latency_ms: float
    retrieval_fallback_used: bool

    # ── Reranking ──────────────────────────────────────────────────────────
    reranked_chunks: list[dict]         # Chunks after cross-encoder reranking
    rerank_skipped: bool                # True when reranking is disabled

    # ── Generation ─────────────────────────────────────────────────────────
    answer: str                         # LLM-generated answer text
    final_answer: str                   # Post-evaluation finalized answer
    citations: list[dict]               # Source references
    model_used: str
    total_tokens: int
    generation_latency_ms: float
    cached: bool

    # ── Evaluation ─────────────────────────────────────────────────────────
    answer_score: float                 # 0.0–1.0 quality estimate
    confidence: float                   # 0.0–1.0 confidence in the final answer
    needs_retry: bool                   # Evaluator requested another retrieval pass
    retry_count: int                    # How many retries have occurred
    eval_reasoning: str                 # Why the answer was accepted/rejected

    # ── Control ────────────────────────────────────────────────────────────
    error: Optional[str]                # Set if any node raised an exception
    total_latency_ms: float


def initial_state(
    query: str,
    top_k: int = 20,
    metadata_filter: Optional[dict] = None,
    strategy_override: Optional[str] = None,
) -> AgentState:
    """Return a clean initial state for a new query."""
    return AgentState(
        messages=[],
        tool_calls=[],
        query=query,
        metadata_filter=metadata_filter,
        strategy_override=strategy_override,
        top_k=top_k,
        metadata={},
        expanded_queries=[query],
        retrieved_chunks=[],
        reranked_chunks=[],
        citations=[],
        answer="",
        final_answer="",
        confidence=0.0,
        answer_score=0.0,
        needs_retry=False,
        retry_count=0,
        retrieval_fallback_used=False,
        rerank_skipped=False,
        cached=False,
        error=None,
    )
