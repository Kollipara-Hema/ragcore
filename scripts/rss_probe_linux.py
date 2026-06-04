"""RSS reclamation probe — Linux/glibc edition.

Inlines _release_chroma_cache_for_path so the probe needs only chromadb +
numpy + the standard library. Run inside a Linux container (python:3.11-slim
matches the prod base image) to answer the single question:

    Does malloc_trim(0) return HNSW/Chroma pages to the OS after
    _release_chroma_cache_for_path + del + gc.collect()?

Output is a sequence of RSS readings (MB) with deltas. The decision number is
"after release+gc+malloc_trim" minus "after upsert" — negative means pages
returned, ~zero means leaked.

MEASUREMENT GOTCHA (do not repeat this bug):
    The OBVIOUS source for "process RSS" is resource.getrusage(RUSAGE_SELF)
    .ru_maxrss. DO NOT USE IT for reclamation probes. ru_maxrss is the PEAK
    RSS over the process lifetime — it monotonically rises and structurally
    CANNOT show a drop, even when the kernel reclaims pages back. A first
    pass of this probe used ru_maxrss, observed "+0 MB delta" everywhere,
    and concluded malloc_trim was useless. That was wrong: the readings were
    just the peak. Read /proc/self/status:VmRSS for current resident pages
    (Linux) — that IS the value malloc_trim affects. On macOS there is no
    /proc; ps(1) or mach_task_basic_info would be the equivalent. Either
    way, never trust ru_maxrss for a before/after RSS comparison.
"""
from __future__ import annotations

import ctypes
import gc
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np


def rss_mb() -> float:
    """CURRENT RSS in MiB, read from /proc/self/status (Linux only).

    Critical: do NOT use resource.getrusage(RUSAGE_SELF).ru_maxrss — that
    reports PEAK RSS over the process lifetime, which can never decrease.
    VmRSS in /proc/self/status reflects the actual currently-resident pages
    and DOES decrease when malloc returns pages to the OS.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kib = int(line.split()[1])
                return kib / 1024
    raise RuntimeError("VmRSS not found in /proc/self/status")


def release_chroma_cache_for_path(persist_dir: Path) -> bool:
    """Inlined from vectorstore/session_store.py:_release_chroma_cache_for_path.
    Returns True if a cache entry was found and stopped."""
    try:
        from chromadb.api.client import SharedSystemClient
    except ImportError:
        print("  [warn] SharedSystemClient not importable")
        return False
    cache = getattr(SharedSystemClient, "_identifer_to_system", None)
    if cache is None:
        print("  [warn] cache layout unexpected")
        return False
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
                print(f"  [warn] system.stop() failed: {exc}")
            cache.pop(key, None)
            return True
    return False


def malloc_trim() -> int | None:
    """Return malloc_trim(0)'s return value, or None if unavailable."""
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return None
    return libc.malloc_trim(0)


def main() -> None:
    print(f"platform: {sys.platform}")
    print(f"glibc: {os.confstr('CS_GNU_LIBC_VERSION') if hasattr(os, 'confstr') else 'unknown'}")

    print(f"\nbaseline RSS:                                {rss_mb():.1f} MB")

    # --- numpy import overhead already paid above ---
    print(f"after numpy import:                          {rss_mb():.1f} MB")

    # Import chromadb (heavy)
    import chromadb
    print(f"after chromadb import:                       {rss_mb():.1f} MB")

    tmpdir = Path(tempfile.mkdtemp(prefix="rss_probe_linux_"))
    persist = tmpdir / "session_probe"
    persist.mkdir()

    client = chromadb.PersistentClient(path=str(persist))
    collection = client.get_or_create_collection(
        name="probe", metadata={"hnsw:space": "cosine"},
    )
    print(f"after PersistentClient + collection:         {rss_mb():.1f} MB")

    # Synthesize 2000 chunks × 1024-dim — matches BGE output dimensionality.
    n, dim = 2000, 1024
    ids = [str(uuid4()) for _ in range(n)]
    embeddings = np.random.default_rng(42).standard_normal((n, dim), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    documents = [f"chunk #{i} " + ("payload word " * 40) for i in range(n)]
    metadatas = [{"chunk_id": ids[i], "chunk_index": i} for i in range(n)]
    print(f"after building {n}×{dim} embeddings in numpy:  {rss_mb():.1f} MB")

    # Upsert into Chroma — triggers HNSW segment write + sqlite + index in RAM.
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    rss_after_upsert = rss_mb()
    print(f"after upsert (HNSW + sqlite + metadata):     {rss_after_upsert:.1f} MB")

    # Drop our Python references.
    del embeddings, ids, documents, metadatas, collection, client

    # Targeted cache release (mirrors session_store.delete()).
    released = release_chroma_cache_for_path(persist)
    print(f"cache release returned:                      {released}")

    gc.collect()
    # A second GC pass — generational collector sometimes needs two.
    gc.collect()
    time.sleep(0.5)
    rss_after_release_gc = rss_mb()
    print(f"after release + gc:                          {rss_after_release_gc:.1f} MB")

    trim_ret = malloc_trim()
    if trim_ret is None:
        print("malloc_trim:                                 N/A (libc.so.6 not loadable)")
    else:
        # malloc_trim returns 1 if memory was actually released, 0 otherwise.
        print(f"malloc_trim(0) return value:                 {trim_ret} (1=released, 0=nothing to release)")

    rss_after_trim = rss_mb()
    print(f"after malloc_trim:                           {rss_after_trim:.1f} MB")

    print()
    print(f"DELTA (after_upsert -> after_release_gc):    {rss_after_release_gc - rss_after_upsert:+.1f} MB")
    print(f"DELTA (after_upsert -> after_malloc_trim):   {rss_after_trim - rss_after_upsert:+.1f} MB")
    print()
    print("Decision: negative delta means pages returned to OS.")
    print("Zero/positive delta means RSS leaked despite cache release.")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
