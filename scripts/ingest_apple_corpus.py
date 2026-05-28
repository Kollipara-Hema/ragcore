"""
Ingest the 5 Apple corpus files into 6 distinct Chroma collections (one
per CORPORA_CONFIG entry). Each entry gets its own ChromaVectorStore at
config["persist_dir"] with collection_name=<corpus_name>, and its own
IngestionPipeline using the configured chunker.

Run BEFORE this script:
    python scripts/fetch_apple_corpus.py   # populates data/apple_demo/

Run THIS script (in venv311):
    python scripts/ingest_apple_corpus.py

Expected wall-clock: 7-12 minutes total. The embedding phase dominates;
log lines per corpus make progress visible.

Exits non-zero on any per-corpus failure — does not silently skip.

Note on the numpy 2.x / chromadb 0.4.24 shim: importing chromadb at module
level here would fail under NumPy 2.x. The shim lives inside
ChromaVectorStore.__init__, so DO NOT add a bare `import chromadb` to this
file. Importing ChromaVectorStore (which we do) is enough.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make repo root importable when the script is run as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.corpora import CORPORA_CONFIG  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from vectorstore.chroma_store import ChromaVectorStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ingest_apple_corpus")


def _sources_for(config: dict) -> list[str]:
    """Return the list of source files for a corpus entry. Supports both
    single-file entries (`source: str`) and the multi-file
    apple_financial_csvs entry (`sources: list[str]`)."""
    if "sources" in config:
        return list(config["sources"])
    return [config["source"]]


async def _ingest_one_corpus(corpus_name: str, config: dict) -> tuple[int, float]:
    """Ingest one corpus end-to-end. Returns (total_chunks, wall_seconds)."""
    persist_dir = config["persist_dir"]
    chunker = config["chunker"]
    sources = _sources_for(config)

    logger.info(
        "Starting %s ingestion (chunker=%s, persist_dir=%s, files=%d)",
        corpus_name, chunker, persist_dir, len(sources),
    )
    start = time.monotonic()

    store = ChromaVectorStore(persist_dir=persist_dir, collection_name=corpus_name)
    pipeline = IngestionPipeline(chunking_strategy=chunker, vector_store=store)

    total_chunks = 0
    for source in sources:
        source_abs = REPO_ROOT / source
        if not source_abs.is_file():
            raise FileNotFoundError(
                f"Corpus {corpus_name}: source not found at {source_abs}. "
                f"Did scripts/fetch_apple_corpus.py run first?"
            )
        _, chunk_count = await pipeline.ingest_file(source_abs)
        total_chunks += chunk_count
        logger.info(
            "  %s file=%s chunks=%d",
            corpus_name, Path(source).name, chunk_count,
        )

    elapsed = time.monotonic() - start
    logger.info(
        "Done %s: %d chunks across %d file(s) in %.1fs",
        corpus_name, total_chunks, len(sources), elapsed,
    )
    return total_chunks, elapsed


async def _main() -> int:
    overall_start = time.monotonic()
    results: list[tuple[str, int, float]] = []

    for corpus_name, config in CORPORA_CONFIG.items():
        try:
            chunks, elapsed = await _ingest_one_corpus(corpus_name, config)
        except Exception:
            logger.exception("Ingestion failed for corpus %s", corpus_name)
            return 1
        results.append((corpus_name, chunks, elapsed))

    overall_elapsed = time.monotonic() - overall_start

    print()
    name_w = max(len(r[0]) for r in results)
    print(f"{'corpus':<{name_w}}  {'chunks':>7}  {'seconds':>8}")
    print("-" * (name_w + 20))
    for name, chunks, elapsed in results:
        print(f"{name:<{name_w}}  {chunks:>7}  {elapsed:>8.1f}")
    print("-" * (name_w + 20))
    total_chunks = sum(c for _, c, _ in results)
    print(f"{'TOTAL':<{name_w}}  {total_chunks:>7}  {overall_elapsed:>8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
