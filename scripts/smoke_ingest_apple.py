"""
Smoke-ingest the 5 Apple corpus files through the loader layer only.

Verifies each Apple corpus file loads cleanly via loader_registry.load(...).
Touches no downstream stage: no chunker, no embedder, no vector store.
Run before any indexing or query work to confirm the loader layer is healthy.

Usage:
    python scripts/smoke_ingest_apple.py

Exit code is the number of files that failed to load (0 on full success).
"""
from __future__ import annotations
import os
import traceback
from pathlib import Path

from ingestion.loaders.document_loaders import loader_registry


CORPUS_DIR = Path(
    os.environ.get(
        "RAGCORE_APPLE_CORPUS_DIR",
        "/Users/hemakollipara/Desktop/files_ragcore",
    )
)

CORPUS_FILES = [
    "apple__sec__form_10k__2025.pdf",
    "apple__sec__q4_earnings_release__2025.html",
    "apple__corporate__environmental_progress_report__2025.pdf",
    "apple__sec__financial_metrics_quarterly__2026.csv",
    "apple__sec__financial_metrics_annual__2026.csv",
]


def main() -> int:
    failures = 0
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        print("=" * 78)
        print(f"file:   {path}")
        if not path.exists():
            failures += 1
            print("MISSING: file not found at expected path")
            continue
        try:
            loader = loader_registry.get_loader(path)
            loader_cls = type(loader).__name__
            docs = loader_registry.load(path)
            total_chars = sum(len(d.content) for d in docs)
            joined = "\n".join(d.content for d in docs)
            print(f"loader: {loader_cls}")
            print(f"docs:   {len(docs)}")
            print(f"chars:  {total_chars}")
            print(f"first 100: {joined[:100]!r}")
            print(f"last 100:  {joined[-100:]!r}")
        except Exception:
            failures += 1
            print("FAILED:")
            traceback.print_exc()
    print("=" * 78)
    ok = len(CORPUS_FILES) - failures
    print(f"summary: {ok}/{len(CORPUS_FILES)} files loaded successfully")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
