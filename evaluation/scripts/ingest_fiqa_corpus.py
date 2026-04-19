"""
Ingest the FiQA corpus subset into RAGCore's FAISS vector store.

Reads evaluation/datasets/fiqa_corpus.json and ingests each document
via IngestionPipeline.ingest_text() — same code path as the API, no
HTTP server required.

Prerequisites:
  - faiss_index.idx and faiss_metadata.pkl must NOT exist (clean store).
  - evaluation/datasets/fiqa_corpus.json must exist (run load_fiqa.py first).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("GENERATION_STRATEGY", "basic")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def main() -> None:
    from ingestion.pipeline import IngestionPipeline
    from vectorstore.vector_store import get_vector_store

    corpus_path = REPO_ROOT / "evaluation" / "datasets" / "fiqa_corpus.json"
    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found. Run load_fiqa.py first.")
        sys.exit(1)

    with open(corpus_path) as f:
        corpus: list[dict] = json.load(f)

    print(f"Corpus loaded: {len(corpus)} documents")
    print("Initialising ingestion pipeline …")

    pipeline = IngestionPipeline()

    total_chunks = 0
    failed = 0
    start = time.monotonic()

    for i, doc in enumerate(corpus, start=1):
        doc_id_label = doc.get("doc_id", f"idx-{i}")
        text = doc.get("text", "").strip()
        if not text:
            logger.warning("Skipping doc %s — empty text", doc_id_label)
            failed += 1
            continue

        metadata = {
            "doc_id": doc_id_label,
            "title": doc.get("title", ""),
            "source": "fiqa",
        }

        try:
            _, chunks_indexed = await pipeline.ingest_text(text, metadata=metadata)
            total_chunks += chunks_indexed
        except Exception as e:
            logger.error("Failed to ingest doc %s: %s", doc_id_label, e)
            failed += 1
            continue

        if i % 10 == 0 or i == len(corpus):
            elapsed = time.monotonic() - start
            print(f"  [{i:>3}/{len(corpus)}] ingested — chunks so far: {total_chunks}  ({elapsed:.1f}s)")

    elapsed = time.monotonic() - start

    # FAISS index stats
    store = get_vector_store()
    index_total = store.index.ntotal if store.index is not None else 0

    idx_file = REPO_ROOT / "faiss_index.idx"
    meta_file = REPO_ROOT / "faiss_metadata.pkl"
    idx_size = idx_file.stat().st_size / 1024 if idx_file.exists() else 0
    meta_size = meta_file.stat().st_size / 1024 if meta_file.exists() else 0

    print(f"\n=== Ingestion complete ===")
    print(f"Documents ingested:   {len(corpus) - failed} / {len(corpus)}")
    print(f"Failed / skipped:     {failed}")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"FAISS ntotal:         {index_total}")
    print(f"Time taken:           {elapsed:.1f}s")
    print(f"faiss_index.idx:      {idx_size:.1f} KB")
    print(f"faiss_metadata.pkl:   {meta_size:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())
