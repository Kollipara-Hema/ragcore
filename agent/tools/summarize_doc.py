"""
Tool: summarize_doc — retrieve all chunks for a document and summarize them.
"""
from __future__ import annotations

from langchain_core.tools import tool
from retrieval.strategies.retrieval_executor import RetrievalExecutor
from retrieval.router.query_router import RoutingDecision
from utils.models import QueryType, RetrievalStrategy

_executor = None


def _get_executor():
    global _executor
    if _executor is None:
        _executor = RetrievalExecutor()
    return _executor


@tool
async def summarize_doc(doc_id: str, max_chunks: int = 10) -> dict:
    """
    Fetch chunks belonging to a specific document and return a summary dict.

    Args:
        doc_id: UUID string of the document to summarize.
        max_chunks: Maximum number of chunks to include.

    Returns:
        Dict with 'doc_id', 'chunk_count', 'combined_text', and 'sources'.
    """
    decision = RoutingDecision(
        query_type=QueryType.SEMANTIC,
        primary_strategy=RetrievalStrategy.METADATA_FILTER,
        fallback_strategy=RetrievalStrategy.SEMANTIC,
        metadata_filter={"doc_id": doc_id},
        expanded_queries=["document summary"],
        reasoning="",
    )
    result = await _get_executor().execute(decision, top_k=max_chunks)
    chunks = result.chunks
    if not chunks:
        return {"doc_id": doc_id, "chunk_count": 0, "combined_text": "", "sources": []}

    combined = "\n\n".join(rc.chunk.content for rc in chunks)
    sources = list({rc.chunk.metadata.get("source", "unknown") for rc in chunks})
    return {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "combined_text": combined,
        "sources": sources,
    }
    """
    Fetch chunks belonging to a specific document and return a summary dict.

    Args:
        doc_id: UUID string of the document to summarize.
        max_chunks: Maximum number of chunks to include.

    Returns:
        Dict with 'doc_id', 'chunk_count', 'combined_text', and 'sources'.
    """
    decision = RoutingDecision(
        query_type=QueryType.SEMANTIC,
        primary_strategy=RetrievalStrategy.METADATA_FILTER,
        fallback_strategy=RetrievalStrategy.SEMANTIC,
        metadata_filter={"doc_id": doc_id},
        expanded_queries=["document summary"],
    )
    result = await _get_executor().execute(decision, top_k=max_chunks)
    chunks = result.chunks
    if not chunks:
        return {"doc_id": doc_id, "chunk_count": 0, "combined_text": "", "sources": []}

    combined = "\n\n".join(rc.chunk.content for rc in chunks)
    sources = list({rc.chunk.metadata.get("source", "unknown") for rc in chunks})
    return {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "combined_text": combined,
        "sources": sources,
    }
