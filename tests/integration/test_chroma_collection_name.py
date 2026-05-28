"""Integration tests for ChromaVectorStore's collection_name parameter.

Covers: the param is honored on the underlying Chroma collection; the default
falls back to settings.chroma_collection_name; distinct persist_dirs isolate
two corpora that share a name; same (persist_dir, collection_name) shares
data across instances.

Not covered (deliberate): shared persist_dir + different collection_names.
That configuration corrupts the BM25 sidecar — see ChromaVectorStore's class
docstring. A test demonstrating the corruption would only document a misuse;
the docstring tells callers not to do it.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from utils.models import Chunk
from vectorstore.chroma_store import ChromaVectorStore


# Two non-overlapping vocabularies so BM25 IDF stays > 0 and cross-corpus
# leakage would be unambiguous if it occurred.
_ALPHA_CONTENTS = (
    "the cat sat on the mat",
    "the dog ran across the field",
    "the fox jumps over the lazy hound",
)
_BETA_CONTENTS = (
    "quantum mechanics describes particle wave duality",
    "general relativity bends spacetime around mass",
    "electromagnetism unites electric and magnetic fields",
)


def _upsert(store: ChromaVectorStore, contents: tuple[str, ...]) -> None:
    doc_id = uuid4()
    chunks = [
        Chunk(content=text, doc_id=doc_id, embedding=[0.1 + 0.1 * i] * 8)
        for i, text in enumerate(contents)
    ]
    asyncio.run(store.upsert(chunks))


def test_collection_name_param_is_honored(tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="custom_name")
    assert store._collection.name == "custom_name"
    assert store._collection_name == "custom_name"


def test_default_collection_name_falls_back_to_settings(tmp_path):
    from config.settings import settings
    store = ChromaVectorStore(persist_dir=str(tmp_path))
    assert store._collection.name == settings.chroma_collection_name
    assert store._collection_name == settings.chroma_collection_name


def test_different_persist_dirs_isolated_even_with_shared_collection_name(tmp_path):
    """The recommended multi-corpus pattern: distinct persist_dirs, any name."""
    store_a = ChromaVectorStore(persist_dir=str(tmp_path / "a"), collection_name="shared")
    store_b = ChromaVectorStore(persist_dir=str(tmp_path / "b"), collection_name="shared")

    _upsert(store_a, _ALPHA_CONTENTS)
    _upsert(store_b, _BETA_CONTENTS)

    # Cross-corpus terms must not leak. store_a knows nothing about "quantum";
    # store_b knows nothing about "cat".
    cat_in_a = asyncio.run(store_a.keyword_search("cat", top_k=5))
    assert len(cat_in_a) > 0
    assert "cat" in cat_in_a[0].chunk.content

    cat_in_b = asyncio.run(store_b.keyword_search("cat", top_k=5))
    assert cat_in_b == []

    quantum_in_b = asyncio.run(store_b.keyword_search("quantum", top_k=5))
    assert len(quantum_in_b) > 0
    assert "quantum" in quantum_in_b[0].chunk.content

    quantum_in_a = asyncio.run(store_a.keyword_search("quantum", top_k=5))
    assert quantum_in_a == []


def test_same_persist_dir_same_collection_name_shares_data(tmp_path):
    """A second instance opened against the same (dir, name) must see writes
    from the first. This is the persistence-across-restart guarantee
    test_chroma_vector_store.py already exercises for the default name;
    here we confirm it also holds for an explicit non-default name."""
    first = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="reopen_test")
    _upsert(first, _ALPHA_CONTENTS)

    second = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="reopen_test")
    results = asyncio.run(second.keyword_search("cat", top_k=5))
    assert len(results) > 0
    assert any("cat" in r.chunk.content for r in results)
