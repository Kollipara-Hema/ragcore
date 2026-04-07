"""
Generator node — builds a prompt from reranked chunks and calls the LLM.
Wraps PromptBuilder + GenerationService; updates answer/citations in AgentState.
"""
from __future__ import annotations

import logging
import time

from agent.state import AgentState
from generation.prompts.prompt_builder import PromptBuilder
from generation.llm_service import get_generation_service
from utils.models import QueryType, RetrievalStrategy, RetrievedChunk, Chunk
from uuid import UUID

logger = logging.getLogger(__name__)
_prompt_builder = None
_generation = None


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
    """Generate an answer from the reranked context chunks."""
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

        prompt = _get_prompt_builder().build(
            query=state["query"],
            chunks=chunks,
            query_type=query_type,
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
