"""
Generator node — builds a prompt from reranked chunks and calls the LLM.
Wraps PromptBuilder + GenerationService; updates answer/citations in AgentState.
Integrates short-term and long-term memory for context.
"""
from __future__ import annotations

import logging
import time

from agent.state import AgentState
from agent.memory.short_term import ShortTermMemory
from agent.memory.long_term import LongTermMemory
from generation.prompts.prompt_builder import PromptBuilder
from generation.llm_service import get_generation_service
from utils.models import QueryType, RetrievalStrategy, RetrievedChunk, Chunk
from uuid import UUID

logger = logging.getLogger(__name__)
_prompt_builder = None
_generation = None
_short_term_memory = None
_long_term_memory = None


def _get_prompt_builder():
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder


def _get_generation():
    global _generation
    if _generation is None:
        _generation = get_generation_service()
    return _generation


def _get_short_term_memory():
    global _short_term_memory
    if _short_term_memory is None:
        _short_term_memory = ShortTermMemory(max_turns=10)
    return _short_term_memory


def _get_long_term_memory():
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory(prefix="ragcore_agent")
    return _long_term_memory


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


async def generator_node(state: AgentState) -> dict:
    """Generate an answer from the reranked context chunks with memory context."""
    start = time.monotonic()
    chunks_dicts = state.get("reranked_chunks") or state.get("retrieved_chunks", [])

    if not chunks_dicts:
        return {
            "answer": (
                "I could not find relevant information in the indexed documents "
                "to answer this question."
            ),
            "citations": [],
            "model_used": "none",
            "total_tokens": 0,
            "generation_latency_ms": 0.0,
            "cached": False,
        }

    try:
        query_type = QueryType(state.get("query_type", "semantic"))
        strategy = RetrievalStrategy(state.get("primary_strategy", "semantic"))
        chunks = [_dict_to_retrieved_chunk(d) for d in chunks_dicts]

        # Build memory context
        short_term_context = _get_short_term_memory().get_context()
        system_extra = ""
        if short_term_context:
            system_extra += f"Previous Conversation:\n{short_term_context}\n"

        prompt = _get_prompt_builder().build(
            query=state["query"],
            chunks=chunks,
            query_type=query_type,
            system_extra=system_extra if system_extra else None,
        )
        result = await _get_generation().generate(
            query=state["query"],
            prompt=prompt,
            query_type=query_type,
            strategy_used=strategy,
        )
        latency = (time.monotonic() - start) * 1000
        citations = [
            {
                "source": c.source,
                "title": c.title,
                "excerpt": c.excerpt,
                "score": round(c.score, 4),
                "doc_id": c.doc_id,
            }
            for c in result.citations
        ]

        # Add this turn to short-term memory
        _get_short_term_memory().add(
            query=state["query"],
            answer=result.answer,
            query_type=state.get("query_type", "semantic"),
            citations=[c.dict() if hasattr(c, "dict") else c for c in result.citations],
        )

        return {
            "answer": result.answer,
            "citations": citations,
            "model_used": result.model_used,
            "total_tokens": result.total_tokens,
            "generation_latency_ms": latency,
            "cached": result.cached,
        }
    except Exception as exc:
        logger.error("generator_node failed: %s", exc)
        latency = (time.monotonic() - start) * 1000
        return {
            "answer": f"Generation failed: {exc}",
            "citations": [],
            "model_used": "error",
            "total_tokens": 0,
            "generation_latency_ms": latency,
            "cached": False,
            "error": str(exc),
        }
