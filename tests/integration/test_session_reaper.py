"""
Session TTL eviction + use-after-free regression + boot cleanup.

Test groups:
  A. Idle eviction — past TTL evicted; fresh skipped.
  B. Idle, NOT absolute — old created_at + fresh last_access survives.
  C. In-flight pin — pinned session is never evicted; C.4 is the
     load-bearing use-after-free regression (slow ingest under sweep).
  E. RAM-release mechanical assertions (cross-platform).
  F. RAM-release RSS assertion (Linux only, skipped on macOS — see audit
     about ru_maxrss vs VmRSS).
  G. Boot cleanup wipes orphan dirs.
  H. Refuse-to-boot on session-root overlap.
  I. Reaper task lifecycle.

Tests do NOT depend on the reaper TIMER firing. They call sweep_once()
directly so timing is deterministic.
"""
from __future__ import annotations

import asyncio
import gc
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app, _assert_session_root_isolated, _purge_orphaned_session_dirs
from config.settings import settings
from vectorstore.chroma_store import ChromaVectorStore
from vectorstore.session_store import SessionStore, sweep_once


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fresh_store(tmp_path) -> SessionStore:
    return SessionStore(root=tmp_path / "sessions")


def _stale_last_access(record, age_seconds: float) -> None:
    """Push a session's last_access into the past so the next sweep
    considers it expired. Bypasses the get()-side bump by directly
    mutating the record."""
    record.last_access = time.time() - age_seconds


# ──────────────────────────────────────────────────────────────────────────────
# A. Idle eviction
# ──────────────────────────────────────────────────────────────────────────────

def test_A1_past_ttl_evicted(tmp_path):
    store = _fresh_store(tmp_path)
    record = store.create()
    persist_dir = record.persist_dir
    assert persist_dir.exists()

    _stale_last_access(record, 7200)
    evicted = sweep_once(store, ttl_seconds=1800)

    assert record.token in evicted
    assert store.get(record.token) is None
    assert not persist_dir.exists()


def test_A2_fresh_not_evicted(tmp_path):
    store = _fresh_store(tmp_path)
    record = store.create()
    # last_access is now() — well within TTL
    evicted = sweep_once(store, ttl_seconds=1800)

    assert evicted == []
    assert store.get(record.token) is not None


# ──────────────────────────────────────────────────────────────────────────────
# B. Idle, NOT absolute — old created_at survives if last_access is fresh
# ──────────────────────────────────────────────────────────────────────────────

def test_B1_old_created_fresh_access_survives(tmp_path):
    store = _fresh_store(tmp_path)
    record = store.create()
    record.created_at = time.time() - 10 * 86400  # 10 days old
    record.last_access = time.time() - 60         # 1 min ago — fresh

    evicted = sweep_once(store, ttl_seconds=1800)

    assert evicted == []
    assert store.get(record.token) is not None


# ──────────────────────────────────────────────────────────────────────────────
# C. In-flight pin protection
# ──────────────────────────────────────────────────────────────────────────────

def test_C1_pinned_session_not_evicted(tmp_path):
    store = _fresh_store(tmp_path)
    record = store.create(pin=True)  # pinned immediately
    _stale_last_access(record, 7200)

    evicted = sweep_once(store, ttl_seconds=1800)

    assert evicted == []
    assert store.get(record.token) is not None
    assert record.in_flight == 1


def test_C2_unpinned_then_swept_evicts(tmp_path):
    store = _fresh_store(tmp_path)
    record = store.create(pin=True)
    _stale_last_access(record, 7200)

    # First sweep: pin holds the session.
    assert sweep_once(store, ttl_seconds=1800) == []

    # unpin only decrements in_flight (no last_access bump), so the
    # session stays stale and the next sweep evicts it.
    store.unpin(record.token)
    assert record.in_flight == 0

    evicted = sweep_once(store, ttl_seconds=1800)
    assert record.token in evicted


def test_C3_concurrent_pin_thread_blocks_eviction(tmp_path):
    """A handler thread holds the pin for ~0.3s; the main thread tries to
    sweep mid-pin. Eviction is correctly skipped — the pin/sweep
    serialization through SessionStore._lock prevents the race."""
    store = _fresh_store(tmp_path)
    record = store.create()
    _stale_last_access(record, 7200)

    pin_held = threading.Event()
    release_pin = threading.Event()

    def pinning_worker():
        store.get(record.token, pin=True)
        pin_held.set()
        release_pin.wait(timeout=2.0)
        store.unpin(record.token)

    t = threading.Thread(target=pinning_worker)
    t.start()
    assert pin_held.wait(timeout=2.0)

    # Sweep while the pin is held — must skip.
    evicted = sweep_once(store, ttl_seconds=1800)
    assert evicted == []
    assert store.get(record.token) is not None

    # Release pin, sweep again, evict succeeds.
    release_pin.set()
    t.join(timeout=2.0)
    # get() above bumped last_access; re-stale before final sweep.
    _stale_last_access(record, 7200)
    evicted = sweep_once(store, ttl_seconds=1800)
    assert record.token in evicted


def test_C4_use_after_free_regression_slow_ingest_under_sweep(tmp_path, monkeypatch):
    """LOAD-BEARING use-after-free regression test.

    Simulates a multi-second ingest that holds record.store across a reaper
    sweep. Without the pin, the reaper would call delete() mid-ingest,
    setting record.store = None and rmtree-ing the persist_dir while
    upsert() is still writing — sqlite OperationalError on a closed
    connection, or worse. With the pin, the reaper sees in_flight > 0,
    skips eviction, and the ingest completes cleanly.

    Implementation: the test runs the SessionStore-level invariant
    directly (no HTTP handlers) so it doesn't depend on the embedder or
    the full ingest pipeline. The pattern under test — "handler pins,
    holds record.store for seconds, reaper attempts eviction with stale
    last_access, ingest completes successfully" — is exactly what the
    handler does at api/main.py.
    """
    store = _fresh_store(tmp_path)
    record = store.create(pin=True)
    chroma = record.store
    assert chroma is not None

    # Stale the access time so the reaper sees it as a candidate.
    _stale_last_access(record, 7200)

    sweep_results = []

    async def run():
        async def slow_ingest():
            """Simulate a slow upsert: tiny payload, but yield to the loop
            long enough that the reaper's sweep happens mid-flight."""
            # Real chromadb upsert — proves record.store is functional.
            from utils.models import Chunk
            from uuid import uuid4
            chunks = [
                Chunk(
                    content=f"slow chunk {i}",
                    doc_id=uuid4(),
                    chunk_id=uuid4(),
                    chunk_index=i,
                    embedding=[0.01 * j for j in range(1024)],
                )
                for i in range(5)
            ]
            # Three small upserts with yields in between simulate the
            # multi-step nature of a real ingest (chunk, embed, upsert,
            # bm25 save) — each step gives the reaper a chance to fire.
            for batch in (chunks[:2], chunks[2:4], chunks[4:]):
                if batch:
                    await chroma.upsert(batch)
                await asyncio.sleep(0.05)

        async def attempt_sweep():
            # Two sweeps during the ingest — both must skip eviction.
            for _ in range(2):
                await asyncio.sleep(0.06)
                sweep_results.append(sweep_once(store, ttl_seconds=1800))

        await asyncio.gather(slow_ingest(), attempt_sweep())

    asyncio.run(run())

    # Critical assertion: NO sweep evicted the pinned session.
    assert all(r == [] for r in sweep_results), (
        f"reaper evicted a pinned session: {sweep_results}"
    )
    # Session and persist_dir still alive.
    assert store.get(record.token) is not None
    assert record.persist_dir.exists()
    # Upsert wrote chunks — store wasn't yanked out from under us.
    assert record.store.count() == 5

    # Cleanup: unpin and let the next sweep evict cleanly.
    store.unpin(record.token)
    _stale_last_access(record, 7200)
    assert record.token in sweep_once(store, ttl_seconds=1800)


# ──────────────────────────────────────────────────────────────────────────────
# E. RAM-release mechanical assertions (cross-platform)
# ──────────────────────────────────────────────────────────────────────────────

def test_E1_delete_removes_chromadb_cache_entry(tmp_path):
    store = _fresh_store(tmp_path)
    record = store.create()
    persist_dir = record.persist_dir

    # Confirm the chromadb cache has an entry for this persist_dir.
    from chromadb.api.client import SharedSystemClient
    cache = SharedSystemClient._identifer_to_system
    has_entry = any(
        Path(getattr(getattr(s, "settings", None), "persist_directory", "")).resolve()
        == persist_dir.resolve()
        for s in cache.values()
    )
    assert has_entry, "chromadb did not cache the session's System"

    assert store.delete(record.token) is True
    assert store.get(record.token) is None
    assert not persist_dir.exists()

    # Cache entry is gone.
    has_entry_after = any(
        Path(getattr(getattr(s, "settings", None), "persist_directory", "")).resolve()
        == persist_dir.resolve()
        for s in cache.values()
    )
    assert not has_entry_after, "chromadb cache entry not released"


def test_E2_fresh_client_at_same_path_after_delete(tmp_path):
    """After delete(), constructing a new ChromaVectorStore at the same
    persist_dir must NOT hit chromadb 0.4.24's duplicate-client ValueError
    — that's the failure mode the cache release exists to prevent."""
    store = _fresh_store(tmp_path)
    record = store.create()
    persist_dir = record.persist_dir
    same_path = str(persist_dir)

    assert store.delete(record.token) is True

    # Path is gone (rmtree); recreate parent so the new client has somewhere
    # to write.
    Path(same_path).mkdir(parents=True, exist_ok=True)
    fresh = ChromaVectorStore(persist_dir=same_path, collection_name="session")
    # If we got here, no ValueError — the cache release worked.
    assert fresh.count() == 0


# ──────────────────────────────────────────────────────────────────────────────
# F. RAM-release RSS — Linux-only, skipped on macOS
# ──────────────────────────────────────────────────────────────────────────────

def _vmrss_mb() -> float:
    """Current VmRSS from /proc/self/status, in MB.
    NOT ru_maxrss — see the audit and scripts/rss_probe_linux.py docstring
    for why ru_maxrss is structurally wrong for before/after comparisons."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS not found")


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="RSS reclamation requires glibc malloc_trim — Linux only. "
           "On macOS the cache release runs but RSS is not measurable via "
           "the same API. See audit Commit 7 item 2.",
)
def test_F1_rss_reclaimed_on_linux(tmp_path):
    """End-to-end RSS check on Linux: after delete + malloc_trim, RSS
    should drop by ≥50% of the post-upsert increment. Probe-verified
    baseline is ~90% reclamation — the 50% threshold leaves margin for
    allocator overhead variation."""
    store = _fresh_store(tmp_path)
    record = store.create()

    rss_before = _vmrss_mb()

    # Upsert a meaningful payload — enough to register on RSS.
    from utils.models import Chunk
    from uuid import uuid4
    n, dim = 2000, 1024
    # Cast to plain Python floats — chromadb 1.x rejects numpy.float32
    # scalars; 0.4.24 accepts them. The plain-float form works everywhere.
    rng = np.random.default_rng(0)
    chunks = [
        Chunk(
            content=f"chunk {i} " + ("payload word " * 40),
            doc_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=i,
            embedding=rng.standard_normal(dim).astype(np.float32).tolist(),
        )
        for i in range(n)
    ]
    asyncio.run(record.store.upsert(chunks))
    del chunks
    gc.collect()
    rss_after_upsert = _vmrss_mb()

    upsert_delta = rss_after_upsert - rss_before
    # Sanity: upsert actually moved RSS (>= 30 MB for 2000×1024).
    assert upsert_delta > 30, f"upsert moved RSS only {upsert_delta:.1f} MB"

    # Evict.
    assert store.delete(record.token) is True
    gc.collect()
    time.sleep(0.5)
    rss_after_evict = _vmrss_mb()

    reclaimed = rss_after_upsert - rss_after_evict
    reclaim_fraction = reclaimed / upsert_delta
    assert reclaim_fraction >= 0.5, (
        f"Reclaimed only {reclaimed:.1f} MB of {upsert_delta:.1f} MB ({reclaim_fraction:.1%}). "
        f"Expected ≥50%. Is malloc_trim being called? rss_before={rss_before:.1f}, "
        f"rss_after_upsert={rss_after_upsert:.1f}, rss_after_evict={rss_after_evict:.1f}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# G. Boot cleanup wipes orphan dirs
# ──────────────────────────────────────────────────────────────────────────────

def test_G1_purges_orphan_session_dirs(tmp_path):
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    (sessions_root / "abc123").mkdir()
    (sessions_root / "def456").mkdir()
    (sessions_root / "def456" / "chroma.sqlite3").touch()

    removed = _purge_orphaned_session_dirs(sessions_root)

    assert removed == 2
    assert sessions_root.exists()
    assert not (sessions_root / "abc123").exists()
    assert not (sessions_root / "def456").exists()


def test_G2_curated_corpus_paths_untouched(tmp_path, monkeypatch):
    """Sibling chroma_collections/ dir must be untouched by session cleanup."""
    sessions_root = tmp_path / "sessions"
    chroma_root = tmp_path / "chroma_collections"
    sessions_root.mkdir()
    chroma_root.mkdir()
    (sessions_root / "orphan_token").mkdir()
    (chroma_root / "apple_10k_fixed").mkdir()
    (chroma_root / "apple_10k_fixed" / "chroma.sqlite3").touch()

    _purge_orphaned_session_dirs(sessions_root)

    # Session orphan gone.
    assert not (sessions_root / "orphan_token").exists()
    # Curated corpus untouched.
    assert (chroma_root / "apple_10k_fixed" / "chroma.sqlite3").exists()


# ──────────────────────────────────────────────────────────────────────────────
# H. Refuse to boot on session-root overlap
# ──────────────────────────────────────────────────────────────────────────────

def test_H1_session_root_equals_chroma_persist_dir_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ragcore_session_root", str(tmp_path / "shared"))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "shared"))

    with pytest.raises(RuntimeError, match="overlaps"):
        _assert_session_root_isolated()


def test_H2_session_root_inside_chroma_persist_dir_refuses(monkeypatch, tmp_path):
    chroma = tmp_path / "chroma_collections"
    chroma.mkdir()
    monkeypatch.setattr(settings, "chroma_persist_dir", str(chroma))
    monkeypatch.setattr(settings, "ragcore_session_root", str(chroma / "sessions"))

    with pytest.raises(RuntimeError, match="overlaps"):
        _assert_session_root_isolated()


def test_H3_session_root_contains_faiss_data_dir_refuses(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(settings, "ragcore_session_root", str(sessions))
    monkeypatch.setattr(settings, "faiss_data_dir", str(sessions / "faiss"))

    with pytest.raises(RuntimeError, match="overlaps"):
        _assert_session_root_isolated()


def test_H4_distinct_roots_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ragcore_session_root", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "faiss_data_dir", str(tmp_path / "faiss"))
    _assert_session_root_isolated()  # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# I. Reaper task lifecycle — starts in lifespan, cancels cleanly on shutdown
# ──────────────────────────────────────────────────────────────────────────────

def test_I1_reaper_task_runs_during_lifespan_and_cancels_on_shutdown(monkeypatch):
    """Boot the real app via TestClient(app) inside `with` so lifespan
    fires, assert the reaper task is alive, exit `with` so lifespan
    shutdown runs, assert the task is cancelled.
    """
    # Fast sweep so we can observe the task entering its loop quickly.
    monkeypatch.setattr(settings, "ragcore_session_sweep_interval_seconds", 1)
    with TestClient(app) as _:
        # Reaper task is set during startup.
        task = api_main._reaper_task
        assert task is not None
        assert not task.done()
    # After `with` exits, lifespan shutdown ran and cancelled the task.
    assert api_main._reaper_task is None
