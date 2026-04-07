"""
Tool: compare_docs — retrieve top chunks from two documents and build a
side-by-side comparison dict for the generator to reason over.
"""
from __future__ import annotations

from langchain_core.tools import tool
from agent.tools.summarize_doc import summarize_doc


@tool
async def compare_docs(doc_id_a: str, doc_id_b: str, max_chunks: int = 5) -> dict:
    """
    Fetch content from two documents and return a structured comparison.

    Args:
        doc_id_a: UUID string of the first document.
        doc_id_b: UUID string of the second document.
        max_chunks: Chunks to pull per document.

    Returns:
        Dict with 'doc_a', 'doc_b' summaries ready for the generator node.
    """
    summary_a, summary_b = await _gather(doc_id_a, doc_id_b, max_chunks)
    return {
        "doc_a": summary_a,
        "doc_b": summary_b,
        "comparison_ready": bool(summary_a["chunk_count"] and summary_b["chunk_count"]),
    }


async def _gather(doc_id_a: str, doc_id_b: str, max_chunks: int):
    import asyncio
    return await asyncio.gather(
        summarize_doc(doc_id_a, max_chunks),
        summarize_doc(doc_id_b, max_chunks),
    )
    """
    Fetch content from two documents and return a structured comparison.

    Args:
        doc_id_a: UUID string of the first document.
        doc_id_b: UUID string of the second document.
        max_chunks: Chunks to pull per document.

    Returns:
        Dict with 'doc_a', 'doc_b' summaries ready for the generator node.
    """
    summary_a, summary_b = await _gather(doc_id_a, doc_id_b, max_chunks)
    return {
        "doc_a": summary_a,
        "doc_b": summary_b,
        "comparison_ready": bool(summary_a["chunk_count"] and summary_b["chunk_count"]),
    }


async def _gather(doc_id_a: str, doc_id_b: str, max_chunks: int):
    import asyncio
    return await asyncio.gather(
        summarize_doc(doc_id_a, max_chunks),
        summarize_doc(doc_id_b, max_chunks),
    )
