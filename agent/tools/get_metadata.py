"""
Tool: get_metadata — look up stored metadata for a document by its ID.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool
from vectorstore.vector_store import get_vector_store


@tool
async def get_metadata(doc_id: str) -> Optional[dict]:
    """
    Return the metadata dict stored alongside a document in the vector store.

    Args:
        doc_id: UUID string of the document.

    Returns:
        Metadata dict (source, title, author, doc_type, tags, etc.)
        or None if the document is not found.
    """
    try:
        store = get_vector_store()
        meta = await store.get_document_metadata(doc_id)
        return meta
    except Exception:
        return None
    """
    Return the metadata dict stored alongside a document in the vector store.

    Args:
        doc_id: UUID string of the document.

    Returns:
        Metadata dict (source, title, author, doc_type, tags, etc.)
        or None if the document is not found.
    """
    try:
        store = get_vector_store()
        meta = await store.get_document_metadata(doc_id)
        return meta
    except Exception:
        return None
