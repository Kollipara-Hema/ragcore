"""
SessionStore primitive tests.

Load-bearing properties pinned here:
  - create then get returns the same store object (no duplicate construction).
  - create is idempotent on a known token (no second PersistentClient at the
    same path → no chromadb 0.4.24 duplicate-client ValueError).
  - Session tokens NEVER appear in the public corpus registry. The
    get_corpus error message leaks sorted(_corpus_registry); if sessions
    ever land there, that 400 body enumerates every live session.
  - delete removes the in-memory record, evicts chromadb's cached System
    for that persist_dir, and removes the on-disk directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vectorstore.session_store import SessionStore, new_session_token


@pytest.fixture
def store(tmp_path):
    return SessionStore(root=tmp_path)


def test_create_then_get_returns_same_store(store):
    """get() returns the same SessionRecord instance create() returned."""
    record = store.create()
    fetched = store.get(record.token)
    assert fetched is record
    assert fetched.store is record.store


def test_create_is_idempotent_on_token(store):
    """
    create(token=existing) returns the SAME object — no second
    ChromaVectorStore, no second PersistentClient at the same persist_dir.
    """
    first = store.create()
    again = store.create(token=first.token)
    assert again is first
    assert again.store is first.store


def test_session_not_in_corpus_registry(store):
    """
    LOAD-BEARING: a session token must not appear in list_corpora() or be
    reachable via get_corpus(). Otherwise the public registry's enumerating
    KeyError leaks every live session token in a single bad /query 400.
    """
    from vectorstore.vector_store import get_corpus, list_corpora

    record = store.create()

    assert record.token not in list_corpora()
    with pytest.raises(KeyError):
        get_corpus(record.token)


def test_delete_removes_dict_entry_and_persist_dir(store):
    """After delete: get returns None, on-disk persist_dir is gone."""
    record = store.create()
    persist_dir = record.persist_dir
    assert persist_dir.exists(), "Setup: persist_dir should exist after create()"

    ok = store.delete(record.token)
    assert ok is True
    assert store.get(record.token) is None
    assert not persist_dir.exists()


def test_delete_unknown_token_returns_false(store):
    """delete() on a never-seen token is a no-op returning False."""
    assert store.delete("nonexistent-token") is False


def test_delete_evicts_chromadb_cache_entry(store):
    """
    The property delete()'s docstring promises: after delete, the per-session
    chromadb System is no longer in SharedSystemClient's process-wide cache.
    Without this, the on-disk delete is shadowed by a still-cached System
    holding the sqlite handle and HNSW RAM until process restart.

    Asserts targeted eviction — other cached entries (Apple corpora, etc.)
    must remain untouched.
    """
    from chromadb.api.client import SharedSystemClient

    record = store.create()
    persist_dir_str = str(record.persist_dir.resolve())

    def _cached_paths():
        cache = SharedSystemClient._identifer_to_system
        out = []
        for system in cache.values():
            sd = getattr(getattr(system, "settings", None), "persist_directory", None)
            if sd:
                out.append(str(Path(sd).resolve()))
        return out

    before = _cached_paths()
    assert persist_dir_str in before, (
        "Setup precondition: chromadb should cache the System after create()"
    )
    other_before = [p for p in before if p != persist_dir_str]

    store.delete(record.token)

    after = _cached_paths()
    assert persist_dir_str not in after, (
        "delete() did not evict the per-session chromadb cache entry"
    )
    # Untouched: anything that was cached before delete (except our entry)
    # must still be cached. Catches an accidental switch to the coarse
    # clear_system_cache() that would break Apple corpora.
    for p in other_before:
        assert p in after, (
            f"delete() incorrectly evicted unrelated cache entry: {p}"
        )


def test_new_session_token_is_filesystem_safe():
    """Token charset is base64url ([A-Za-z0-9_-]) — no '/' or '..' so the
    token is safe to use as a directory name with no sanitization."""
    for _ in range(100):
        token = new_session_token()
        assert "/" not in token
        assert ".." not in token
        assert len(token) >= 32
