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
   budgets — landing in Commits 5-7) from curated Apple/FiQA corpora.
   Keeping them in distinct data structures prevents accidental cleanup of
   curated corpora.

Sessions persist at {root}/<token>/, where root is a SEPARATE filesystem
root from chroma_persist_dir (Apple corpora) and faiss_data_dir (FiQA), so
session cleanup can never touch curated data. Token charset is base64url
([A-Za-z0-9_-]) — no path traversal possible.
"""
from __future__ import annotations

import logging
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def new_session_token() -> str:
    """32 bytes of entropy → 43-char base64url string. Charset is
    [A-Za-z0-9_-] so the token is safe to use as a directory name with no
    sanitization — no '..', no '/'."""
    return secrets.token_urlsafe(32)


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
    byte_count: int = 0        # populated by Commit 4 ingest wiring
    file_count: int = 0        # populated by Commit 4 ingest wiring


class SessionStore:
    """
    Token -> SessionRecord. In-process dict, never persisted; process restart
    loses all tokens. Boot cleanup of orphaned persist_dirs lands in Commit 7.

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

    def create(self, token: Optional[str] = None) -> SessionRecord:
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
        """
        with self._lock:
            if token and token in self._records:
                existing = self._records[token]
                existing.last_access = time.time()
                return existing

            new_token = token or new_session_token()
            # Defend against the statistically-impossible minted-token
            # collision; bail without constructing a second client.
            if new_token in self._records:
                return self._records[new_token]

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
            )
            self._records[new_token] = record
            return record

    def get(self, token: str) -> Optional[SessionRecord]:
        with self._lock:
            record = self._records.get(token)
            if record is not None:
                record.last_access = time.time()
            return record

    def delete(self, token: str) -> bool:
        """
        Drop the dict entry, release the per-session chromadb System from
        chromadb's process-wide cache, then rm -rf the persist_dir.

        Releasing the cached System is what actually closes the sqlite
        handle and frees the HNSW segment from RAM; without it, deleting
        the on-disk files leaves chromadb's SharedSystemClient holding a
        dangling sqlite connection (and the associated RAM) until process
        restart. We do a TARGETED eviction (only this session's entry)
        instead of clear_system_cache's process-wide wipe — clearing all
        would also stop the six Apple ChromaVectorStore objects' Systems
        and break their next query.

        rmtree runs OUTSIDE the lock (filesystem ops are slow and the
        dict entry is already gone, so no other caller can find this
        token).

        TODO: RSS-actually-freed-after-delete verified in Commit 7.
        """
        with self._lock:
            record = self._records.pop(token, None)
        if record is None:
            return False

        # Drop our reference so refcount can hit zero on the
        # ChromaVectorStore once the chromadb cache entry is also gone.
        record.store = None
        _release_chroma_cache_for_path(record.persist_dir)
        shutil.rmtree(record.persist_dir, ignore_errors=True)
        return True


def _release_chroma_cache_for_path(persist_dir: Path) -> None:
    """Targeted eviction of chromadb's cached System for this persist_dir.

    Best-effort: depends on chromadb 0.4.24 internals
    (SharedSystemClient._identifer_to_system [sic — typo'd in 0.4.24 source],
    system.settings.persist_directory). If those layouts change in a future
    chromadb upgrade, we log and continue rather than fail — the on-disk
    delete still runs, the consequence is a transient RAM leak until
    process restart. Commit 7 verifies the property end-to-end.
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
            return
