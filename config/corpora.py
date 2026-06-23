"""
Static registry of the Apple multi-corpus demo: provenance label, offline
ingest path, chunking strategy, and Chroma persist_dir for each corpus.

Consumed by:
  - scripts/ingest_apple_corpus.py — builds a ChromaVectorStore + an
    IngestionPipeline per entry and runs ingest_file() against `ingest_path`
    (or each path in `ingest_paths`). Offline regeneration only.
  - api/main.py lifespan — for each entry whose persist_dir exists on disk,
    constructs a ChromaVectorStore(persist_dir=..., collection_name=<key>)
    and calls register_corpus(<key>, store). Missing persist_dirs are
    skipped with a warning so deploys without ingested data still boot.

FiQA (the "default" corpus) is intentionally NOT in this dict — it lives on
FAISS and is registered separately in lifespan.

Per-entry schema:
    source:      human-readable provenance label advertised verbatim by
                 GET /corpora. NOT a path into the image — deployed corpora
                 are served from the bundled `persist_dir`, so the source
                 documents need not exist at runtime.
    ingest_path: local path to the file passed to ingest_file() by the
                 offline scripts/ingest_apple_corpus.py. Never read on a live
                 request path. apple_financial_csvs uses `ingest_paths` (a
                 list) instead, ingesting two CSVs into one collection.
    chunker:     the factory key for ingestion.chunkers.chunkers.get_chunker
    persist_dir: per-corpus Chroma directory. Distinct per corpus because
                 the BM25 sidecar (bm25_state.pkl) lives at persist_dir
                 granularity, not collection granularity — sharing would
                 cross-contaminate BM25 state across corpora.

Chunker-key gotcha: the factory key for DocumentStructureChunker is
"structure", not "document_structure". The corpus name
apple_10k_document_structure is our naming (it identifies which corpus is
being served); the `chunker` field below MUST be the factory key
"structure" or get_chunker raises ValueError at ingestion time.
"""
from __future__ import annotations

from config.settings import settings

# Root for per-corpus Chroma persist directories. Local dev resolves to
# ./data/chroma_collections (settings.chroma_persist_dir default); production
# sets CHROMA_PERSIST_DIR=/var/data/chroma_db so collections live on the
# persistent disk instead of the ephemeral container filesystem.
_CHROMA_ROOT = settings.chroma_persist_dir

CORPORA_CONFIG: dict[str, dict] = {
    "apple_10k_fixed": {
        "source": "SEC EDGAR — Apple FY2025 Form 10-K",
        "ingest_path": "data/apple_demo/apple__sec__form_10k__2025.pdf",
        "chunker": "fixed",
        "persist_dir": f"{_CHROMA_ROOT}/apple_10k_fixed",
    },
    # NOTE: "hierarchical" here names the CHUNKER class
    # (HierarchicalChunker), not retrieve-child-return-parent semantics.
    # The ingestion pipeline filters parent chunks out before embedding
    # and no retrieval code expands children back to parents. With
    # default settings this corpus is effectively a FixedSize child
    # corpus (256-char windows, 32-char overlap). Wiring real parent-
    # child retrieval is a deferred follow-up.
    "apple_10k_hierarchical": {
        "source": "SEC EDGAR — Apple FY2025 Form 10-K",
        "ingest_path": "data/apple_demo/apple__sec__form_10k__2025.pdf",
        "chunker": "hierarchical",
        "persist_dir": f"{_CHROMA_ROOT}/apple_10k_hierarchical",
    },
    "apple_10k_document_structure": {
        "source": "SEC EDGAR — Apple FY2025 Form 10-K",
        "ingest_path": "data/apple_demo/apple__sec__form_10k__2025.pdf",
        "chunker": "structure",
        "persist_dir": f"{_CHROMA_ROOT}/apple_10k_document_structure",
    },
    "apple_environmental": {
        "source": "Apple.com Environment — Apple 2025 Environmental Progress Report",
        "ingest_path": "data/apple_demo/apple__corporate__environmental_progress_report__2025.pdf",
        "chunker": "structure",
        "persist_dir": f"{_CHROMA_ROOT}/apple_environmental",
    },
    "apple_earnings_html": {
        "source": "SEC EDGAR — Apple FY2025 Q4 earnings release (8-K exhibit)",
        "ingest_path": "data/apple_demo/apple__sec__q4_earnings_release__2025.html",
        "chunker": "fixed",
        "persist_dir": f"{_CHROMA_ROOT}/apple_earnings_html",
    },
    # Two CSVs, one corpus. The ingest script special-cases this entry by
    # iterating `ingest_paths` instead of using `ingest_path` directly, so the
    # same ChromaVectorStore + IngestionPipeline absorbs both files into a
    # single collection (each CSV gets its own doc_id).
    "apple_financial_csvs": {
        "source": "SEC companyfacts XBRL — Apple financial metrics (quarterly + annual)",
        "ingest_paths": [
            "data/apple_demo/apple__sec__financial_metrics_quarterly__2026.csv",
            "data/apple_demo/apple__sec__financial_metrics_annual__2026.csv",
        ],
        "chunker": "fixed",
        "persist_dir": f"{_CHROMA_ROOT}/apple_financial_csvs",
    },
}
