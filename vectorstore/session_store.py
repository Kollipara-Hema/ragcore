"""
Session-scoped vector store registry.

Lives BESIDE the public corpus registry in vector_store.py, never inside it.
This separation is load-bearing for two reasons:

1. The public registry's get_corpus error message echoes
   sorted(_corpus_registry) into /query 400 bodies (vector_store.py:256-258).
   If session tokens were registered there, a single bad-corpus query would
   enumerate every live session. SessionStore is never enumerated and never
   reachable via the client-supplied corpus string.

2. Session corpora have different lifetimes (TTL, eviction, per-session byte
   budgets) from curated Apple/FiQA corpora. Keeping them in distinct data
   structures prevents accidental cleanup of curated corpora.

Sessions persist at {root}/<token>/, where root is a SEPARATE filesystem
root from chroma_persist_dir (Apple corpora) and faiss_data_dir (FiQA), so
session cleanup can never touch curated data. Token charset is base64url
([A-Za-z0-9_-]) — no path traversal possible.
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from vectorstore.chroma_store import ChromaVectorStore

# Structlog so the kwargs (ttl_seconds, evicted_count, session_prefix, ...)
# are rendered as structured fields, consistent with api/main.py. A stdlib
# Logger would TypeError on the kwargs.
logger = structlog.get_logger(__name__)


def new_session_token() -> str:
    """32 bytes of entropy → 43-char base64url string. Charset is
    [A-Za-z0-9_-] so the token is safe to use as a directory name with no
    sanitization — no '..', no '/'."""
    return secrets.token_urlsafe(32)


class SessionCapacityError(Exception):
    """Raised by SessionStore.create() when minting a NEW session would
    exceed max_concurrent. Get-or-create on an existing token never raises
    this — reads don't pay the concurrency check."""


@dataclass
class SessionRecord:
    token: str
    # Optional because delete() nulls it to drop the client reference before
    # rmtree. Records in the live registry always have a non-None store; the
    # None state is transient inside delete().
    store: Optional[ChromaVectorStore]
    persist_dir: Path
    created_at: float          # time.time() at create
    last_access: float         # time.time() updated by get()/create() hits
    byte_count: int = 0        # populated by the ingest handlers on success
    file_count: int = 0        # populated by the ingest handlers on success
    # In-flight pin count (use-after-free guard). Bumped atomically with
    # get()/create() when pin=True; decremented by unpin(). The reaper's
    # delete() refuses to evict while in_flight > 0 — see delete().
    in_flight: int = 0


class SessionStore:
    """
    Token -> SessionRecord. In-process dict, never persisted; process restart
    loses all tokens. Boot cleanup of orphaned persist_dirs is not yet
    implemented.

    Concurrency model:
      - Single-worker asyncio (the deployed shape): the no-await rule alone
        prevents coroutine interleaving between dict lookup and dict insert.
      - Sync FastAPI handler in the threadpool: the threading.Lock covers it.
      - Multi-worker uvicorn (N > 1): out of scope. SessionStore is per-
        process by design; cross-worker sessions would need Redis-backed
        state. Render runs --workers 1.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create(
        self,
        token: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        *,
        pin: bool = False,
    ) -> SessionRecord:
        """
        Get-or-create. If `token` is provided and exists, returns the
        existing record. If `token` is provided and is new, OR `token` is
        None, mints a fresh session under the lock.

        ChromaVectorStore construction (a few hundred ms — sqlite init,
        HNSW segment load, BM25 sidecar restore) happens INSIDE the lock.
        This is deliberate: if we released the lock before construction,
        two threads both observing "token absent" would both build a
        PersistentClient at the same persist_dir, and the second would
        raise chromadb 0.4.24's duplicate-client ValueError. The slow
        critical section is correctness-required; on a single-worker
        uvicorn the contention is negligible.

        If `max_concurrent` is provided, raises SessionCapacityError when
        MINTING a fresh session would push the total session count over
        the cap. Get-or-create on an EXISTING token bypasses the check
        (read paths don't shed load).

        If `pin=True`, the returned record's in_flight counter is bumped
        atomically with the get/create — closing the window where the
        reaper could evict between resolve and pin. Callers MUST call
        unpin(token) in a finally to release the pin.
        """
        with self._lock:
            if token and token in self._records:
                existing = self._records[token]
                existing.last_access = time.time()
                if pin:
                    existing.in_flight += 1
                return existing

            new_token = token or new_session_token()
            # Defend against the statistically-impossible minted-token
            # collision; bail without constructing a second client.
            if new_token in self._records:
                return self._records[new_token]

            # Concurrent-session cap — applied only when minting fresh.
            # Atomic with the dict insert below so two concurrent mint
            # requests can't both observe "under cap" and both proceed.
            if max_concurrent is not None and len(self._records) >= max_concurrent:
                raise SessionCapacityError(
                    f"Process at max_concurrent={max_concurrent} sessions"
                )

            persist_dir = self._root / new_token
            store = ChromaVectorStore(
                persist_dir=str(persist_dir),
                collection_name="session",
            )
            now = time.time()
            record = SessionRecord(
                token=new_token,
                store=store,
                persist_dir=persist_dir,
                created_at=now,
                last_access=now,
                in_flight=1 if pin else 0,
            )
            self._records[new_token] = record
            return record

    def get(self, token: str, *, pin: bool = False) -> Optional[SessionRecord]:
        """Look up a session. If pin=True, bump in_flight atomically with the
        lookup so the reaper cannot evict between get() returning and the
        caller acting on the record. Callers MUST unpin(token) in a finally.
        """
        with self._lock:
            record = self._records.get(token)
            if record is not None:
                record.last_access = time.time()
                if pin:
                    record.in_flight += 1
            return record

    def unpin(self, token: str) -> None:
        """Decrement in_flight. No-op if the session is gone (delete() refuses
        to evict pinned sessions, so this should only see a missing record
        if the caller called unpin twice; logged as a warning)."""
        with self._lock:
            record = self._records.get(token)
            if record is None:
                logger.warning(
                    "unpin called on missing session %s", token[:6],
                )
                return
            if record.in_flight <= 0:
                logger.warning(
                    "unpin underflow on session %s (in_flight=%d)",
                    token[:6], record.in_flight,
                )
                return
            record.in_flight -= 1

    def try_reserve_file(self, token: str, max_files: int) -> bool:
        """Atomic check-and-increment for file_count. Returns False if the
        session is unknown OR already at max_files. On True, file_count is
        incremented and last_access bumped; the caller MUST call
        release_file_reservation() if the subsequent ingest fails — the
        reserve+rollback pattern is what makes the cap a real limit under
        concurrent ingests (without it, two parallel uploads can both
        observe count == max-1 and both succeed)."""
        with self._lock:
            record = self._records.get(token)
            if record is None or record.file_count >= max_files:
                return False
            record.file_count += 1
            record.last_access = time.time()
            return True

    def release_file_reservation(self, token: str) -> None:
        """Roll back a try_reserve_file() that preceded a failed ingest.
        No-op if the session is gone or the counter is already at zero."""
        with self._lock:
            record = self._records.get(token)
            if record is None or record.file_count <= 0:
                return
            record.file_count -= 1

    def record_bytes(self, token: str, byte_delta: int) -> None:
        """Track total bytes uploaded to a session. Telemetry today; future
        per-session byte-budget enforcement and TTL/LRU decisions will read
        this field. No-op if the session is gone."""
        with self._lock:
            record = self._records.get(token)
            if record is None:
                return
            record.byte_count += byte_delta
            record.last_access = time.time()

    def delete(self, token: str) -> bool:
        """
        Drop the dict entry, release the per-session chromadb System from
        chromadb's process-wide cache, then rm -rf the persist_dir.

        Use-after-free guard: returns False (without evicting) when the
        session is in-flight (in_flight > 0). The reaper retries on the
        next sweep; an explicit caller can poll or wait for the pin to
        release. There is intentionally no force=True bypass — every
        current caller (reaper) MUST honor pins.

        Releasing the cached System is what actually closes the sqlite
        handle and frees the HNSW segment from RAM; without it, deleting
        the on-disk files leaves chromadb's SharedSystemClient holding a
        dangling sqlite connection (and the associated RAM) until process
        restart. We do a TARGETED eviction (only this session's entry)
        instead of clear_system_cache's process-wide wipe — clearing all
        would also stop the six Apple ChromaVectorStore objects' Systems
        and break their next query.

        After the targeted cache eviction, malloc_trim(0) is called inside
        _release_chroma_cache_for_path on glibc systems to return freed
        arena pages to the OS. Probe-verified: ~90% of the per-session
        working set is reclaimed (~54 MB of a ~60 MB cycle). Without
        malloc_trim only ~33% reclaims (the cache release frees Python
        objects but pages stay with the allocator). See
        scripts/rss_probe_linux.py.

        rmtree runs OUTSIDE the lock (filesystem ops are slow and the
        dict entry is already gone, so no other caller can find this
        token).
        """
        with self._lock:
            record = self._records.get(token)
            if record is None:
                return False
            if record.in_flight > 0:
                logger.info(
                    "session_delete_skipped_pinned",
                    session_prefix=token[:6],
                    in_flight=record.in_flight,
                )
                return False
            del self._records[token]

        # Drop our reference so refcount can hit zero on the
        # ChromaVectorStore once the chromadb cache entry is also gone.
        record.store = None
        _release_chroma_cache_for_path(record.persist_dir)
        shutil.rmtree(record.persist_dir, ignore_errors=True)
        return True


def _release_chroma_cache_for_path(persist_dir: Path) -> None:
    """Targeted eviction of chromadb's cached System for this persist_dir,
    followed by a glibc malloc_trim to return arena pages to the OS.

    Best-effort: depends on chromadb 0.4.24 internals
    (SharedSystemClient._identifer_to_system [sic — typo'd in 0.4.24 source],
    system.settings.persist_directory). If those layouts change in a future
    chromadb upgrade, we log and continue rather than fail — the on-disk
    delete still runs, the consequence is a transient RAM leak until
    process restart.

    RAM reclamation is LOAD-BEARING for the 2 GB Render box: probe-verified
    on Linux/glibc 2.41 that the cache-release alone reclaims ~33% of the
    per-session working set, while cache-release + malloc_trim reclaims
    ~90%. Without malloc_trim, a few dozen evict-and-respawn cycles would
    OOM the process. See scripts/rss_probe_linux.py for the measurement,
    including the ru_maxrss-vs-VmRSS gotcha that makes naive probes report
    a false-negative.

    On non-glibc systems (macOS dev), libc.so.6 doesn't load and the
    malloc_trim call is skipped — local dev sees the cache release work
    but no RSS return; production (Linux) gets both.
    """
    try:
        from chromadb.api.client import SharedSystemClient
    except ImportError:
        logger.warning(
            "chromadb SharedSystemClient not importable; "
            "skipping per-path cache release for %s",
            persist_dir,
        )
        return

    cache = getattr(SharedSystemClient, "_identifer_to_system", None)
    if cache is None:
        logger.warning(
            "chromadb cache layout unexpected; skipping per-path release for %s",
            persist_dir,
        )
        return

    target = Path(persist_dir).resolve()
    for key in list(cache.keys()):
        system = cache[key]
        sd = getattr(getattr(system, "settings", None), "persist_directory", None)
        if not sd:
            continue
        if Path(sd).resolve() == target:
            try:
                system.stop()
            except Exception as exc:
                logger.warning("system.stop() failed during session delete: %s", exc)
            cache.pop(key, None)
            break

    # Return freed arena pages to the OS (glibc only). On macOS this raises
    # OSError because libc.so.6 doesn't exist; the fall-through is
    # intentional — dev sees the cache release without RSS reclamation.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def sweep_once(
    store: "SessionStore",
    *,
    ttl_seconds: float,
    now: Optional[float] = None,
) -> list[str]:
    """One reaper sweep iteration. Evicts every session whose
    (now - last_access) > ttl_seconds AND in_flight == 0. Returns the list
    of evicted tokens (full tokens — caller MUST truncate before logging).

    Two-phase walk: candidates are identified under the lock, then deleted
    outside the lock so the slow rmtree+cache-release doesn't block handler
    reads. Between the two phases, a candidate can become pinned by a
    handler — delete() re-checks in_flight under the lock and returns
    False, leaving the session alive for the next sweep.
    """
    now = now if now is not None else time.time()
    with store._lock:
        candidates = [
            token
            for token, record in store._records.items()
            if (now - record.last_access) > ttl_seconds and record.in_flight == 0
        ]

    evicted: list[str] = []
    for token in candidates:
        if store.delete(token):
            evicted.append(token)
    return evicted


async def reaper_loop(
    store: "SessionStore",
    *,
    ttl_seconds: float,
    sweep_interval_seconds: float,
) -> None:
    """Background asyncio task. Sleeps sweep_interval_seconds between
    sweeps. Cancellation-aware — exits cleanly on asyncio.CancelledError.
    Any other exception in a sweep is logged but does NOT kill the loop,
    so a single transient failure doesn't disable eviction until the next
    process bounce.
    """
    logger.info(
        "session_reaper_started",
        ttl_seconds=ttl_seconds,
        sweep_interval_seconds=sweep_interval_seconds,
    )
    try:
        while True:
            await asyncio.sleep(sweep_interval_seconds)
            try:
                evicted = sweep_once(store, ttl_seconds=ttl_seconds)
                if evicted:
                    logger.info(
                        "session_reaper_swept",
                        evicted_count=len(evicted),
                        # truncated prefixes only — never the full tokens
                        evicted_prefixes=[t[:6] for t in evicted],
                    )
            except Exception as exc:
                logger.error(
                    "session_reaper_sweep_failed: %s", exc, exc_info=True,
                )
    except asyncio.CancelledError:
        logger.info("session_reaper_cancelled")
        raise
