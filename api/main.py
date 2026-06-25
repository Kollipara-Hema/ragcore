"""
=============================================================================
api/main.py — The FastAPI Web Application (Your REST API)
=============================================================================
PURPOSE:
    This file defines all the HTTP endpoints (URLs) that users and front-ends
    call to interact with the RAG system.

ENDPOINTS:
    POST /ingest/file         Upload a document file (PDF, DOCX, TXT, etc.)
    POST /ingest/text         Submit raw text directly for indexing
    DELETE /documents/{id}    Remove a document from the index
    POST /query               Ask a question, get a full answer
    POST /query/stream        Ask a question, get a streaming answer
    GET  /health              Check if the API is running
    GET  /metrics             Performance metrics

HOW TO RUN:
    uvicorn rag_system.api.main:app --reload --port 8000

HOW TO TEST:
    Visit http://localhost:8000/docs for the interactive API documentation
=============================================================================
"""

# --- Standard library ---
from __future__ import annotations
import logging     # For writing log messages
import os          # For file path operations and environment
import shutil      # For copytree() when seeding Apple collections onto the persist disk
import tempfile    # For saving uploaded files temporarily
import uuid        # For generating unique IDs

import structlog

# contextlib.asynccontextmanager — used for startup/shutdown lifecycle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

# --- FastAPI framework ---
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
# FastAPI     = the web framework
# File        = marks a parameter as a file upload
# Form        = marks a parameter as a form field
# HTTPException = raises HTTP errors (400, 404, 500, etc.)
# UploadFile  = represents an uploaded file

from fastapi.middleware.cors import CORSMiddleware
# CORS = Cross-Origin Resource Sharing
# Allows browsers on other domains to call this API
# e.g. your React frontend at localhost:3000 calling this API at localhost:8000

from fastapi.responses import JSONResponse, StreamingResponse
# StreamingResponse = streams data token by token (for chat-like UX)

# --- Internal modules ---
from config.settings import settings, LLMProvider    # App configuration from .env

# Data models for request/response shapes
from utils.models import (
    IngestRequest, IngestResponse, DocumentStatus, QueryRequest,
    AgentQueryRequest, AgentQueryResponse, QueryTrace,
    CorpusInfo, CorporaListResponse,
    RetrievalOnlyResponse,
)

# The main RAG pipeline orchestrator
from orchestrator import RAGOrchestrator

# Agent graph for LangGraph-based agents
from agent.graph import build_graph
from agent.state import initial_state

# The document ingestion pipeline (loading, chunking, embedding, indexing)
from ingestion.pipeline import IngestionPipeline

# Vector store (for document deletion and health checks)
from vectorstore.vector_store import (
    BaseVectorStore,
    FAISSVectorStore,
    get_corpus,
    get_vector_store,
    list_corpora,
    register_corpus,
)
from vectorstore.chroma_store import ChromaVectorStore
from vectorstore.session_store import (
    SessionStore,
    SessionRecord,
    SessionCapacityError,
    reaper_loop,
)

# Standard library — added for the reaper task and boot cleanup
import asyncio

# Apple multi-corpus demo configuration
from config.corpora import CORPORA_CONFIG

# Embedder (for health checks)
from embeddings.embedder import get_embedder

# Rate limiting middleware (prevents abuse)
from api.middleware.rate_limit import RateLimitMiddleware

# API key authentication middleware (optional, gated by settings.auth_enabled)
from api.middleware.auth import APIKeyAuthMiddleware

# Body-size limit middleware for /ingest/* (outermost; see onion order below)
from api.middleware.body_size import BodySizeLimitMiddleware

# Structured logging setup and request-ID middleware
from monitoring.logging_config import configure_logging
from api.middleware.request_id import RequestIdMiddleware

# Monitoring tracer
from monitoring.tracer import get_tracer

# Prometheus instrumentation — importing metrics registers the ResourceCollector
from prometheus_fastapi_instrumentator import Instrumentator
import monitoring.metrics  # noqa: F401

# Logger for this module
logger = structlog.get_logger(__name__)

# =============================================================================
# Global instances — created once at startup, reused for all requests
# =============================================================================
# These are declared here but initialized in the lifespan function below.
# Using globals avoids recreating expensive objects (like loading ML models)
# on every request.

orchestrator: Optional[RAGOrchestrator] = None    # The RAG pipeline
ingestion_pipeline: Optional[IngestionPipeline] = None  # Document indexing
session_store: Optional[SessionStore] = None      # Per-session vector stores


# =============================================================================
# Apple-corpus seeding — copy repo-shipped collections to the persist dir
# =============================================================================
# The 6 Apple Chroma collections are tracked in-repo at data/chroma_collections/
# and bundled into the Docker image. In production, settings.chroma_persist_dir
# points at the persistent disk (e.g. /var/data/chroma_db), which starts empty
# on a fresh disk. _seed_apple_collections() copies each per-corpus directory
# from the repo path to the persist path on first boot; subsequent boots
# detect the existing chroma.sqlite3 and skip. In local dev where the two
# paths resolve to the same location, the function no-ops.

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_CHROMA_SOURCE = _REPO_ROOT / "data" / "chroma_collections"
_REPO_FAISS_SOURCE = _REPO_ROOT / "data" / "faiss_seed"
_FAISS_INDEX_FILE = "faiss_index.idx"
_FAISS_METADATA_FILE = "faiss_metadata.pkl"


def _seed_apple_collections() -> None:
    """Idempotent, atomic first-boot copy of repo-shipped Chroma collections
    onto the runtime persist_dir. Copies the ENTIRE per-corpus directory —
    copying only chroma.sqlite3 would leave bm25_state.pkl and the HNSW
    binaries behind, breaking hybrid retrieval.

    Atomicity matters because the next-boot skip-check keys off
    `dest/chroma.sqlite3`. A crashed mid-copy that left a bare sqlite at dest
    would be mistaken for a complete seed forever — bricking that corpus. We
    copy into a sibling temp dir and os.replace() it into the final dest, so
    chroma.sqlite3 only appears at dest when the whole tree is in place.
    """
    for corpus_name, config in CORPORA_CONFIG.items():
        source = _REPO_CHROMA_SOURCE / corpus_name
        dest = Path(config["persist_dir"])

        if source.resolve() == dest.resolve():
            logger.info(
                "Seed skipped for %s: source == dest (%s)", corpus_name, source,
            )
            continue
        if (dest / "chroma.sqlite3").is_file():
            logger.info(
                "Seed skipped for %s: %s already seeded", corpus_name, dest,
            )
            continue
        if not source.is_dir():
            logger.warning(
                "Seed source missing for %s at %s; corpus will be skipped at registration",
                corpus_name, source,
            )
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".{corpus_name}.seeding-tmp"
        logger.info("Seeding %s: %s -> %s (via %s)", corpus_name, source, dest, tmp)
        try:
            # Clean up any leftover temp from a prior crashed seed before we
            # start, so copytree has a clean dst to create.
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(source, tmp)
            # A partial dest (no sqlite, the skip-check already passed) gets
            # removed so os.replace can land the fully-populated temp atomically.
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(tmp, dest)
        except Exception as exc:
            logger.error(
                "Seed failed for %s (%s); corpus will be skipped at registration",
                corpus_name, exc, exc_info=True,
            )
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            continue


def _seed_faiss_default() -> None:
    """Seed the bundled FiQA FAISS artifacts onto FAISS_DATA_DIR before the
    default corpus is registered. Idempotent (skip if dest index exists),
    atomic per-file (tmp + os.replace, index written LAST so the marker only
    appears when both files are in place).

    Idempotency boundary: PRESENCE, not CORRECTNESS. A stale or corrupt
    pre-existing dest index is SKIPPED, not validated. Validating existing
    state is the readiness-probe doc_count check's job (deferred).

    DIVERGES from _seed_apple_collections: any failure here raises
    RuntimeError, NOT a logged warning. The default corpus is what every
    unscoped /query hits; an empty default that boots silently serves empty
    answers and /health/ready keeps passing — the documented silent-outage
    mode (debugging-notes 2026-05-12). The Apple seed soft-fallbacks because
    losing one explicit-name corpus degrades gracefully; the default cannot.
    Do not unify.

    This is the LIFESPAN contract for the default corpus. The lazy
    get_vector_store() shim constructs a FAISSVectorStore directly from
    settings.faiss_data_dir without going through this seed — tests that
    bypass the lifespan are responsible for their own state setup.
    """
    source_dir = _REPO_FAISS_SOURCE
    source_index = source_dir / _FAISS_INDEX_FILE
    source_metadata = source_dir / _FAISS_METADATA_FILE
    dest_dir = Path(settings.faiss_data_dir)
    dest_index = dest_dir / _FAISS_INDEX_FILE
    dest_metadata = dest_dir / _FAISS_METADATA_FILE

    if source_dir.resolve() == dest_dir.resolve():
        # Warning, not info — source==dest in prod is a misconfiguration.
        logger.warning("faiss_seed_skipped_source_eq_dest", path=str(source_dir))
        return

    if dest_index.is_file():
        logger.info("faiss_seed_skipped_already_seeded", dest=str(dest_dir))
        return

    if not source_index.is_file():
        raise RuntimeError(
            f"Default-corpus seed failed: source index missing at {source_index}. "
            f"Refusing to boot — an empty-but-serving default is the documented "
            f"silent-outage mode."
        )
    if not source_metadata.is_file():
        raise RuntimeError(
            f"Default-corpus seed failed: source metadata missing at {source_metadata}. "
            f"Refusing to boot — an empty-but-serving default is the documented "
            f"silent-outage mode."
        )

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Default-corpus seed failed: cannot create dest dir {dest_dir} ({exc}). "
            f"Refusing to boot — an empty-but-serving default is the documented "
            f"silent-outage mode."
        ) from exc

    tmp_index = dest_dir / f"{_FAISS_INDEX_FILE}.seed-tmp"
    tmp_metadata = dest_dir / f"{_FAISS_METADATA_FILE}.seed-tmp"

    # Pre-cleanup leftover tmps from a prior crashed seed so copy2 has clean targets.
    for stale in (tmp_index, tmp_metadata):
        stale.unlink(missing_ok=True)

    logger.info("faiss_seed_starting", source=str(source_dir), dest=str(dest_dir))
    try:
        shutil.copy2(source_index, tmp_index)
        shutil.copy2(source_metadata, tmp_metadata)
        # Metadata first, index LAST. Idempotency keys off index existence;
        # writing it last ensures a crash between replaces leaves index missing
        # (next boot re-seeds) — never a half-state that looks complete.
        os.replace(tmp_metadata, dest_metadata)
        os.replace(tmp_index, dest_index)
    except Exception as exc:
        for stale in (tmp_index, tmp_metadata):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError(
            f"Default-corpus seed failed during copy: {exc}. "
            f"Refusing to boot — an empty-but-serving default is the documented "
            f"silent-outage mode."
        ) from exc

    logger.info("faiss_seed_complete", dest=str(dest_dir))


# =============================================================================
# Boot-cleanup helpers — session-root isolation + orphaned-dir purge
# =============================================================================

def _assert_session_root_isolated() -> None:
    """Refuse to boot when the session root overlaps any curated-corpus path.

    `_purge_orphaned_session_dirs` rmtrees children of `session_root` at
    every startup. If a misconfiguration ever pointed `RAGCORE_SESSION_ROOT`
    at a path that contains (or is contained by) `faiss_data_dir` or
    `chroma_persist_dir`, that startup purge would silently delete the
    Apple seed corpora or FiQA index. Belt-and-suspenders: refuse to boot
    rather than risk it. The settings docstring at config/settings.py:100
    states the separation as a design contract; this assert enforces it
    at runtime.
    """
    session = Path(settings.ragcore_session_root).resolve()
    forbidden = {
        "faiss_data_dir": Path(settings.faiss_data_dir).resolve(),
        "chroma_persist_dir": Path(settings.chroma_persist_dir).resolve(),
    }
    for name, other in forbidden.items():
        # Overlap = same path, OR one contains the other. Path.is_relative_to
        # returns True for equal paths too (a path is relative to itself), so
        # both directions catch all overlap cases.
        if session == other or session.is_relative_to(other) or other.is_relative_to(session):
            raise RuntimeError(
                f"ragcore_session_root ({session}) overlaps with {name} "
                f"({other}). Refusing to boot — boot cleanup would delete "
                f"curated corpus data."
            )


def _purge_orphaned_session_dirs(root: Path) -> int:
    """Wipe every direct child directory of `root`. At boot, the in-memory
    SessionStore._records is empty — every on-disk per-token dir is an
    orphan from the previous process and useless (the in-memory token is
    gone; no client can address it).

    Safety: caller MUST have run _assert_session_root_isolated() first.
    This function trusts `root` is not a curated-corpus path. Returns the
    count of dirs removed for telemetry.
    """
    if not root.exists():
        return 0
    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            shutil.rmtree(child)
            removed += 1
        except Exception as exc:
            logger.warning(
                "boot_orphan_purge_failed for %s: %s", child, exc,
            )
    return removed


# Reaper task handle — set in lifespan startup, cancelled in shutdown. Module-
# level so the shutdown side of lifespan can find it.
_reaper_task: Optional[asyncio.Task] = None


# =============================================================================
# LIFESPAN — Startup and Shutdown Logic
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the server starts (before any requests).
    Code after 'yield' runs ONCE when the server shuts down.

    This is where we initialize expensive resources like:
    - Loading ML models into memory
    - Creating database connections
    - Setting up the orchestrator
    """
    global orchestrator, ingestion_pipeline, session_store, _reaper_task

    configure_logging()
    logger.info("Starting RAG system...")

    # Refuse-to-boot if session_root overlaps any curated-corpus path. Runs
    # BEFORE any other boot step, including seeding — a misconfigured
    # session_root that points at chroma_persist_dir would otherwise let
    # _purge_orphaned_session_dirs delete the Apple seeds.
    _assert_session_root_isolated()

    # Seed FiQA before constructing the FAISS store — the store reads the
    # index on construction, and a fail-hard seed prevents empty-but-serving.
    _seed_faiss_default()

    # Register the FAISS-backed "default" corpus (FiQA). Unconditional —
    # /query requests omitting `corpus` route here via the get_vector_store
    # shim, which would lazy-construct on first call anyway; doing it here
    # makes the registered set visible at startup and keeps the registration
    # path uniform with the Chroma corpora below.
    register_corpus("default", FAISSVectorStore())

    # Seed Apple collections onto the runtime persist_dir before registration.
    # No-op in local dev (source == dest) and on subsequent prod boots
    # (chroma.sqlite3 already present on the persistent disk).
    _seed_apple_collections()

    # Register each Apple Chroma corpus whose persist_dir contains an
    # ingested AND non-empty collection. Missing persist_dirs are skipped
    # with a warning so a deploy without ingested Apple data still starts
    # and serves the default corpus. Empty collections (count() == 0) are
    # also skipped — guards against a partial seed leaving an unusable
    # store registered.
    for corpus_name, config in CORPORA_CONFIG.items():
        persist_dir = Path(config["persist_dir"])
        if not (persist_dir / "chroma.sqlite3").is_file():
            logger.warning(
                "Corpus %s not found at %s, skipping", corpus_name, persist_dir,
            )
            continue
        store = ChromaVectorStore(
            persist_dir=str(persist_dir), collection_name=corpus_name,
        )
        if store.count() == 0:
            logger.warning(
                "Corpus %s at %s is empty (count=0); skipping registration",
                corpus_name, persist_dir,
            )
            continue
        register_corpus(corpus_name, store)
        logger.info("Registered corpus %s from %s", corpus_name, persist_dir)

    # Create the main RAG orchestrator — this loads ML models into memory
    # (cross-encoder reranker, BGE embedder if local — may take 10-30 seconds)
    orchestrator = RAGOrchestrator()

    # Create the ingestion pipeline — used for synchronous file processing
    ingestion_pipeline = IngestionPipeline()

    # Construct the per-session vector store registry. The session root is
    # intentionally a SEPARATE filesystem subtree from the FAISS/Chroma
    # curated-corpus dirs (see settings docstring) so session cleanup
    # cannot touch curated data — enforced at boot by
    # _assert_session_root_isolated above.
    session_store = SessionStore(root=settings.ragcore_session_root)

    # Wipe orphaned per-token dirs from prior processes. _records is empty
    # at boot, so every on-disk dir is unreachable (no in-memory token can
    # address it). Runs BEFORE the reaper task starts to avoid racing with
    # the first sweep.
    purged = _purge_orphaned_session_dirs(Path(settings.ragcore_session_root))
    if purged:
        logger.info("boot_orphan_purge_complete", purged_dirs=purged)

    # Spawn the reaper background task. Single-worker uvicorn means one
    # reaper per process — no cross-worker coordination needed. Cancelled
    # on shutdown below.
    _reaper_task = asyncio.create_task(
        reaper_loop(
            session_store,
            ttl_seconds=settings.ragcore_session_idle_ttl_seconds,
            sweep_interval_seconds=settings.ragcore_session_sweep_interval_seconds,
        ),
        name="session_reaper",
    )

    logger.info("RAG system ready. Visit http://localhost:8000/docs")

    yield  # Server is now running — handle requests

    # Cleanup on shutdown (close connections, etc.)
    logger.info("Shutting down RAG system...")
    if _reaper_task is not None:
        _reaper_task.cancel()
        try:
            await _reaper_task
        except asyncio.CancelledError:
            pass
        _reaper_task = None


# =============================================================================
# CREATE THE FASTAPI APP
# =============================================================================

app = FastAPI(
    title="DocIntel RAG API",                    # Shown in /docs
    description="Production-grade multi-strategy RAG system",  # Shown in /docs
    version="1.0.0",
    lifespan=lifespan,  # Use our startup/shutdown function above
)

Instrumentator().instrument(app).expose(app)

# =============================================================================
# MIDDLEWARE — Code that runs on EVERY request before and after the handler
# =============================================================================

# CORS Middleware — allows the configured origins (settings.cors_origins) to
# call this API. Default is ["http://localhost:8501"]; override via the
# CORS_ORIGINS env var (comma-separated, e.g. "https://app.example.com").
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Request-Id", "X-API-Key", "X-Session-Id"],
    # expose_headers is REQUIRED so browsers can read X-Session-Id off the
    # response cross-origin. Without it, the header is delivered by FastAPI
    # but stripped from JavaScript's view of the response — clients would
    # have no way to capture the minted session token.
    expose_headers=["X-Session-Id", "X-Request-Id"],
)

# API Key Authentication Middleware — only registered when auth is enabled.
# Sits inside RateLimit so bad-key floods are throttled before key validation
# runs. See the onion-order comment at BodySizeLimitMiddleware below for the
# full pipeline.
if settings.ragcore_auth_enabled:
    if not settings.ragcore_api_key:
        raise RuntimeError(
            "RAGCORE_AUTH_ENABLED=true but RAGCORE_API_KEY is unset or empty. "
            "Set a non-empty API key or disable auth."
        )
    app.add_middleware(APIKeyAuthMiddleware, api_key=settings.ragcore_api_key)
    logger.info("API key authentication enabled")
else:
    logger.info("API key authentication disabled (RAGCORE_AUTH_ENABLED=false)")

# Rate Limiting Middleware — prevents a single user from spamming the API.
# Defined in api/middleware/rate_limit.py.
# Configurable via RAGCORE_RATE_LIMIT_MAX_REQUESTS / RAGCORE_RATE_LIMIT_WINDOW_SECONDS.
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.ragcore_rate_limit_max_requests,
    window_seconds=settings.ragcore_rate_limit_window_seconds,
    trust_proxy_headers=settings.ragcore_trust_proxy_headers,
    proxy_hop_count=settings.ragcore_proxy_hop_count,
)

# Request-ID Middleware — assigns a request_id and binds it to structlog
# contextvars so downstream log calls carry it automatically. Note: 413/400
# responses from BodySizeLimitMiddleware do NOT carry a request_id because
# BodySize is outermost; acceptable, since those rejections never reach any
# logic that would benefit from correlation.
app.add_middleware(RequestIdMiddleware)

# Body-size limit — OUTERMOST middleware. add_middleware is LIFO (last added
# runs first), so registering this last makes it the first thing every
# request hits. A single integer compare on Content-Length rejects oversized
# /ingest/* bodies before any downstream component buffers a byte; the
# wrapped receive() catches missing/lying Content-Length (chunked uploads).
#
# Outermost position is LOAD-BEARING. This guard's whole value is being
# cheaper than anything it protects — moving it inward (e.g. inside
# RateLimit or Auth) lets the protected components pay setup cost for
# requests this would have rejected for free. A regression that reorders
# middleware silently is exactly what tests/unit/test_body_size.py guards
# against.
#
# Final onion order, outermost → innermost:
#   BodySize → RequestId → RateLimit → Auth → CORS → handler
app.add_middleware(
    BodySizeLimitMiddleware,
    max_bytes=settings.ragcore_ingest_max_body_bytes,
    protected_prefixes=("/ingest/file", "/ingest/text"),
)


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================

@app.get("/health", tags=["system"])
async def health():
    """
    Health check endpoint.
    Returns 200 OK if the API is running.
    Used by Docker, load balancers, and monitoring tools to check if app is alive.

    Example: curl http://localhost:8000/health
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": settings.environment,  # "development" or "production"
    }


# Map each LLM provider to its settings field carrying the API key.
_LLM_KEY_MAP: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "openai_api_key",
    LLMProvider.ANTHROPIC: "anthropic_api_key",
    LLMProvider.GROQ: "groq_api_key",
}


def _check_llm_config() -> None:
    """Raise RuntimeError with a one-line reason if the active LLM provider has no API key."""
    provider = settings.llm_provider
    field = _LLM_KEY_MAP.get(provider)
    if field is None:
        raise RuntimeError(f"{provider.value} provider is not fully configured")
    key = getattr(settings, field, None)
    if not key:
        raise RuntimeError(f"{field.upper()} is not set")


@app.get("/health/live", tags=["system"])
async def health_live():
    """Process-alive check. Returns 200 immediately with no dependency checks."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["system"])
async def health_ready():
    """
    Readiness check. Verifies vector store reachability, embedder responsiveness,
    and LLM API key presence. Returns 200 if all pass, 503 if any fail.
    Response body includes per-check results and a reason string for any failure.
    """
    checks: dict = {}
    all_ok = True

    # Check 1: vector store reachable
    try:
        get_vector_store().ping()
        checks["vector_store"] = {"ok": True}
    except Exception as exc:
        checks["vector_store"] = {"ok": False, "reason": str(exc)}
        all_ok = False

    # Check 2: embedder loaded and responsive
    try:
        embedding = await get_embedder().embed_query("a")
        if not embedding:
            raise RuntimeError("embed_query returned empty result")
        checks["embedder"] = {"ok": True}
    except Exception as exc:
        checks["embedder"] = {"ok": False, "reason": str(exc)}
        all_ok = False

    # Check 3: LLM API key present (no live call)
    try:
        _check_llm_config()
        checks["llm_config"] = {"ok": True}
    except Exception as exc:
        checks["llm_config"] = {"ok": False, "reason": str(exc)}
        all_ok = False

    body = {"status": "ready" if all_ok else "not_ready", "checks": checks}
    return JSONResponse(content=body, status_code=200 if all_ok else 503)


# =============================================================================
# INGESTION ENDPOINTS — For uploading and indexing documents
# =============================================================================

def _resolve_or_mint_session(request: Request) -> SessionRecord:
    """Resolve an existing session from the X-Session-Id request header, or
    mint a new server-generated session if the header is missing OR names a
    session this process no longer holds (e.g. after a restart, or after
    the session reaper evicted it).

    Returns a PINNED record (in_flight bumped atomically with the get/create
    under the SessionStore lock). The caller MUST call session_store.unpin
    in a finally block; otherwise the session is leaked from eviction
    forever. Pinning here closes the use-after-free window between resolve
    and the long-running ingest — the reaper cannot evict between the lock
    release inside get/create and the handler's downstream operations.

    Raises HTTPException(503) when minting would exceed
    ragcore_session_max_concurrent. Get-or-create on an existing token does
    not pay the concurrency check (read paths don't shed load)."""
    token = request.headers.get("X-Session-Id")
    if token:
        existing = session_store.get(token, pin=True)
        if existing is not None:
            return existing
    try:
        return session_store.create(
            max_concurrent=settings.ragcore_session_max_concurrent,
            pin=True,
        )
    except SessionCapacityError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Server at session capacity "
                f"({settings.ragcore_session_max_concurrent}). "
                f"Try again later."
            ),
        )


# Magic-byte sniffer constants
_SNIFF_PREFIX_BYTES = 8192  # filetype only needs the first few hundred bytes


def _sniff_file_kind(content: bytes) -> str:
    """Return the AUTHORITATIVE type of the uploaded bytes. One of:
      "pdf"    — filetype.guess identified as PDF
      "text"   — filetype.guess returned None AND bytes decode as UTF-8
      "reject" — any recognized non-PDF binary (docx/zip/png/...) OR
                 unrecognized AND non-UTF-8-decodable bytes

    The three-way classification is deliberate. The final allowlist is PDF +
    text only, so HTML/CSV-specific heuristics would (a) do security work for
    types already eliminated by routing, and (b) false-reject valid Markdown
    with inline <div> blocks or TXT with commas.

    Safety property: an HTML or CSV file lands here as "text" and routes to
    TextLoader (Path.read_text — no BeautifulSoup, no pandas). The parser
    pathology that motivated dropping HTML/CSV from the user-input path is
    eliminated structurally by removing the dangerous loaders from this
    flow, not by a classifier that can misfire.
    """
    import filetype
    kind = filetype.guess(content[:_SNIFF_PREFIX_BYTES])
    if kind is not None:
        return "pdf" if kind.extension == "pdf" else "reject"
    try:
        content.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "reject"


# Map sniffed_kind to the temp-file suffix used for loader dispatch. This is
# the LOAD-BEARING seam: by deriving the suffix from sniffed_kind rather than
# from file.filename, the loader registry's extension-based dispatch
# (see ingestion/loaders/document_loaders.py LoaderRegistry) cannot be
# steered by a client-supplied extension. A spoofed filename has no effect.
_SNIFFED_KIND_TO_SUFFIX: dict[str, str] = {
    "pdf": ".pdf",
    "text": ".txt",
}


def _resolve_query_target(request: Request) -> Optional[SessionRecord]:
    """Resolves the query's vector-store target from the X-Session-Id header.

    Routing precedence (load-bearing isolation invariant):
      - X-Session-Id present AND resolves to a live session → return that
        session's record. The request body's `corpus` field is IGNORED
        ENTIRELY. Session routing wins unconditionally — a session header
        commits the request to the session store even if `corpus` names a
        valid public corpus.
      - X-Session-Id present BUT unknown/expired → raise HTTPException(404).
        Presence of the header commits to the session path; we do NOT
        silently fall back to the public corpus. Falling back would answer
        "what does my doc say?" with default-corpus chunks the user never
        uploaded — a worse failure mode than a clean 404.
      - X-Session-Id absent → return None. Caller routes via body.corpus
        through the public registry.

    The 404 body intentionally does NOT echo the submitted token. Logs use a
    6-char truncated prefix only; full tokens never appear in logs or
    responses.

    INVARIANT (use-after-free, mirrors ingest sites):
      record.store is captured here for the lifetime of the query. The
      returned record is PINNED (in_flight bumped) atomically with the
      get() under the SessionStore lock, so the reaper cannot evict
      between resolve and the orchestrator returning. The caller MUST
      unpin in a finally; query_stream pins synchronously and unpins
      inside the generator's finally to span the StreamingResponse
      consumption. Grep "use-after-free" to find all three sites.
    """
    token = request.headers.get("X-Session-Id")
    if token is None:
        return None

    existing = session_store.get(token, pin=True)
    if existing is not None:
        logger.info(
            "query_routed",
            route="session",
            session_prefix=token[:6],
        )
        return existing

    logger.info(
        "query_routed",
        route="session_unknown",
        session_prefix=token[:6],
    )
    raise HTTPException(
        status_code=404,
        detail="Session not found or expired",
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["ingestion"])
async def ingest_file(
    request: Request,
    response: Response,
    file: UploadFile = File(...),               # The uploaded file — REQUIRED
    title: Optional[str] = Form(None),          # Optional document title
    tags: Optional[str] = Form(None),           # Optional comma-separated tags
):
    """
    Upload a document file to be indexed into the caller's per-session
    corpus. The session is identified by an X-Session-Id request header;
    if absent, a server-generated token is minted and returned in the
    X-Session-Id RESPONSE header. The default FiQA store is NEVER touched
    by this endpoint.

    Supported file types: PDF and plain text (TXT/MD). The client-supplied
    Content-Type and filename are IGNORED for type decisions — the actual
    bytes are sniffed (magic-byte for PDF; UTF-8 decode for text). See
    _sniff_file_kind. PDFs are also capped at ragcore_pdf_max_pages (default
    100); larger PDFs return 413 before the embed pipeline runs.

    Blocks until indexing is complete. May take 10-60 seconds for large files.

    Example:
        curl -X POST http://localhost:8000/ingest/file
             -F "file=@report.pdf"
             -F "title=Q3 Report"
             -i  # show the X-Session-Id response header on first call
    """
    content = await file.read()

    # Per-file size guard — tighter than the outer 10 MB body cap. Body cap
    # is transport protection; this one is per-session abuse protection.
    if len(content) > settings.ragcore_session_max_file_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds per-file limit "
                f"({settings.ragcore_session_max_file_bytes} bytes)."
            ),
        )

    # Magic-byte sniff is the AUTHORITATIVE type. Drives both (a) the
    # allowlist gate immediately below and (b) the temp-file suffix used
    # for loader dispatch below — see _SNIFFED_KIND_TO_SUFFIX. The client's
    # Content-Type and filename play NO role in routing.
    sniffed_kind = _sniff_file_kind(content)
    if sniffed_kind not in _SNIFFED_KIND_TO_SUFFIX:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. Allowed: PDF, TXT, MD. "
                f"Detected: {sniffed_kind!r}."
            ),
        )

    record = _resolve_or_mint_session(request)  # PINNED — unpin in finally below

    # Reserve a file slot atomically. If two parallel uploads to one session
    # race the cap, only N succeed; the (N+1)th gets 409 here without the
    # ingest pipeline running.
    if not session_store.try_reserve_file(
        record.token, max_files=settings.ragcore_session_max_files,
    ):
        session_store.unpin(record.token)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session file limit reached "
                f"({settings.ragcore_session_max_files}). "
                f"Start a new session to upload more."
            ),
        )

    # Save the bytes to disk — loaders read from a file path. CRITICAL: the
    # suffix is derived from sniffed_kind, NOT from file.filename. This is
    # the seam that makes the sniffed type drive loader dispatch: the loader
    # registry's get_loader() walks loaders and calls supports() which
    # checks str(source).lower().endswith(".ext"). A client-supplied
    # extension cannot reach this codepath.
    suffix = _SNIFFED_KIND_TO_SUFFIX[sniffed_kind]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    metadata = {
        "title": title or file.filename,
        "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],
        "original_filename": file.filename,
    }

    try:
        # PDF page-cap probe — fires AFTER cheap pymupdf metadata read,
        # BEFORE pipeline.ingest_file() begins any chunk/embed work. A file
        # whose bytes passed sniff (valid %PDF magic) but whose body is
        # truncated/corrupt raises inside pymupdf.open and is classified
        # as 400, NOT swallowed into a 500 from deep inside to_markdown().
        # The existing except HTTPException branch below releases the
        # session reservation and the finally unlinks the temp file, so
        # this path needs no bespoke cleanup.
        if sniffed_kind == "pdf":
            import pymupdf
            try:
                with pymupdf.open(tmp_path) as _pdf:
                    n_pages = _pdf.page_count
            except Exception as exc:
                logger.warning("PDF page-count probe failed: %s", exc)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "PDF could not be parsed "
                        f"({exc.__class__.__name__}). The file may be "
                        "truncated or corrupt."
                    ),
                )
            if n_pages > settings.ragcore_pdf_max_pages:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"PDF exceeds page limit "
                        f"({settings.ragcore_pdf_max_pages} pages, "
                        f"got {n_pages})."
                    ),
                )

        # Per-request IngestionPipeline bound to THIS session's store.
        # __init__ is cheap (chunker is a factory call with no ML state,
        # embedder is the module-level singleton — see pipeline.py:78-86
        # and embedder.py:199-230), so allocating one per request is fine.
        #
        # INVARIANT: record.store is captured here for the duration of the
        # ingest (use-after-free). The session reaper (spawned at startup,
        # see reaper_loop) could otherwise evict this session between this
        # line and pipeline.ingest_file() returning and upsert into a
        # deleted store. The session is pinned for the request duration —
        # acquired with a pin above and released via session_store.unpin()
        # in the finally below — so the reaper cannot evict mid-ingest. The
        # same pinning applies at _resolve_query_target AND at ingest_text
        # below. Grep "use-after-free" to find all three.
        pipeline = IngestionPipeline(vector_store=record.store)
        actual_doc_id, chunk_count = await pipeline.ingest_file(tmp_path, metadata)
    except HTTPException:
        session_store.release_file_reservation(record.token)
        raise
    except Exception as e:
        session_store.release_file_reservation(record.token)
        logger.error("Session ingest_file failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        # Unpin AFTER the pipeline has fully returned. Until this point the
        # reaper cannot evict this session — record.store is safe to use.
        session_store.unpin(record.token)

    session_store.record_bytes(record.token, len(content))
    response.headers["X-Session-Id"] = record.token
    return IngestResponse(
        job_id="sync",
        doc_id=actual_doc_id,
        status=DocumentStatus.INDEXED,
        message=f"Successfully indexed {chunk_count} chunks.",
    )


@app.post("/ingest/text", response_model=IngestResponse, tags=["ingestion"])
async def ingest_text(
    request: Request,
    response: Response,
    body: IngestRequest,
):
    """
    Index raw text into the caller's per-session corpus. Same session
    contract as /ingest/file — X-Session-Id request header in, X-Session-Id
    response header out. The default FiQA store is NEVER touched.
    """
    if not body.text_content:
        raise HTTPException(status_code=400, detail="text_content is required")

    content_bytes = body.text_content.encode("utf-8")
    if len(content_bytes) > settings.ragcore_session_max_file_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Text payload exceeds per-file limit "
                f"({settings.ragcore_session_max_file_bytes} bytes)."
            ),
        )

    record = _resolve_or_mint_session(request)  # PINNED — unpin in finally below

    if not session_store.try_reserve_file(
        record.token, max_files=settings.ragcore_session_max_files,
    ):
        session_store.unpin(record.token)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session file limit reached "
                f"({settings.ragcore_session_max_files}). "
                f"Start a new session to upload more."
            ),
        )

    try:
        # Per-request IngestionPipeline — see ingest_file() for the
        # __init__-is-cheap rationale and the session-reaper
        # use-after-free invariant. Same constraint applies at
        # _resolve_query_target. Grep "use-after-free" to find all sites.
        pipeline = IngestionPipeline(vector_store=record.store)
        doc_id, chunk_count = await pipeline.ingest_text(body.text_content, body.metadata)
    except HTTPException:
        session_store.release_file_reservation(record.token)
        raise
    except Exception as e:
        session_store.release_file_reservation(record.token)
        logger.error("Session ingest_text failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session_store.unpin(record.token)

    session_store.record_bytes(record.token, len(content_bytes))
    response.headers["X-Session-Id"] = record.token
    return IngestResponse(
        job_id="sync",
        doc_id=doc_id,
        status=DocumentStatus.INDEXED,
        message=f"Successfully indexed {chunk_count} chunks.",
    )


@app.delete("/documents/{doc_id}", tags=["ingestion"])
async def delete_document(doc_id: str):
    """
    Remove all chunks of a document from the index.

    Use this to:
    - Update a document (delete then re-upload)
    - Remove outdated or incorrect documents
    - Clean up test data

    Example:
        curl -X DELETE http://localhost:8000/documents/abc-123-def
    """
    try:
        store = get_vector_store()   # Get the vector store instance
        # Delete all chunks with this doc_id
        count = await store.delete_document(uuid.UUID(doc_id))
        return {"doc_id": doc_id, "chunks_deleted": count}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid doc_id format: {doc_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# QUERY ENDPOINTS — For asking questions about indexed documents
# =============================================================================

@app.post("/query", tags=["query"])
async def query(request: Request, body: QueryRequest):
    """
    Ask a question about your indexed documents.

    Routing:
      - With a valid X-Session-Id header: retrieval reads ONLY from that
        session's store. The `corpus` body field is ignored.
      - Without the header: retrieval reads from the public corpus selected
        by `corpus` (defaults to the FiQA "default" corpus).
      - With an unknown/expired X-Session-Id: 404. No silent fallback.

    Request body options:
        query           (required) — your question
        top_k           (optional) — how many chunks to retrieve (default from .env)
        metadata_filter (optional) — filter by doc type, author, etc.
        strategy_override (optional) — force a specific retrieval strategy
        corpus          (optional) — public corpus name. IGNORED when
                                     X-Session-Id resolves.
        stream          (optional) — use /query/stream endpoint instead for streaming

    Example:
        curl -X POST http://localhost:8000/query
             -H "Content-Type: application/json"
             -d '{"query": "What were the main findings in the Q3 report?"}'
    """
    session_record = _resolve_query_target(request)  # PINNED if non-None
    store_override: Optional[BaseVectorStore] = (
        session_record.store if session_record else None
    )
    if session_record is None:
        logger.info("query_routed", route="public", corpus=body.corpus)

    try:
        # Run the full RAG pipeline and return the complete answer
        result = await orchestrator.query(body, store_override=store_override)
        return result
    except KeyError as ke:
        # Raised by get_corpus() when body.corpus names an unregistered
        # corpus. Message is curated upstream and names the registered set
        # so callers can spot a typo. Cannot fire on the session path —
        # store_override bypasses get_corpus() entirely.
        raise HTTPException(status_code=400, detail=str(ke))
    except Exception as e:
        logger.error("query_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
    finally:
        if session_record is not None:
            session_store.unpin(session_record.token)


@app.post("/retrieve", response_model=RetrievalOnlyResponse, tags=["query"])
async def retrieve(request: Request, body: QueryRequest):
    """
    Retrieval-only endpoint: router → retrieve → rerank, NO generation.

    Consumes ZERO LLM tokens — generation is skipped entirely. Used by the
    chunker comparison view and other retrieval-debugging surfaces where the
    generated answer isn't needed. Accepts the same QueryRequest as /query
    (verify_claims is a no-op here, since no generation runs). Returns
    retrieval_candidates, stage_timings (retrieve + rerank only),
    strategy_used, query_type, and a top_rerank_score convenience field.

    Same session-routing contract as /query — see _resolve_query_target.

    Example:
        curl -X POST http://localhost:8000/retrieve
             -H "Content-Type: application/json"
             -d '{"query": "What is a 401k?", "corpus": "default"}'
    """
    session_record = _resolve_query_target(request)  # PINNED if non-None
    store_override: Optional[BaseVectorStore] = (
        session_record.store if session_record else None
    )
    if session_record is None:
        logger.info("retrieve_routed", route="public", corpus=body.corpus)

    try:
        return await orchestrator.retrieve_only(body, store_override=store_override)
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=str(ke))
    except Exception as e:
        logger.error("retrieve_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval processing failed: {str(e)}"
        )
    finally:
        if session_record is not None:
            session_store.unpin(session_record.token)


@app.post("/query/stream", tags=["query"])
async def query_stream(request: Request, body: QueryRequest):
    """
    Ask a question with a streaming response (tokens appear as they generate).

    This creates a ChatGPT-like experience where the answer appears
    word by word rather than waiting for the full response.

    Uses Server-Sent Events (SSE) format:
        data: First words of the answer\n\n
        data: ...more words...\n\n
        data: [DONE]\n\n

    How to consume in JavaScript:
        const evtSource = new EventSource('/query/stream');
        evtSource.onmessage = (e) => {
            if (e.data === '[DONE]') evtSource.close();
            else document.write(e.data);
        };

    Example:
        curl -X POST http://localhost:8000/query/stream
             -H "Content-Type: application/json"
             -d '{"query": "Summarize the key points."}'
             --no-buffer
    """
    # Resolve session BEFORE the StreamingResponse is constructed. An unknown
    # X-Session-Id raises 404 here — committed as the HTTP status, not
    # surfaced as an in-band SSE [ERROR] event after a 200 has shipped. Same
    # ordering rationale as the public-corpus pre-check below.
    #
    # PIN: _resolve_query_target returns a PINNED record (in_flight bumped)
    # synchronously here, BEFORE we return StreamingResponse to the client.
    # That closes the use-after-free window across the entire stream
    # consumption. The matching unpin lives inside event_stream's finally
    # because the generator runs AFTER this function returns — a normal
    # try/finally around `return StreamingResponse(...)` would unpin before
    # the stream is consumed.
    session_record = _resolve_query_target(request)
    store_override: Optional[BaseVectorStore] = (
        session_record.store if session_record else None
    )

    # Pre-validate corpus membership ONLY on the public path. On the session
    # path body.corpus is ignored entirely by routing (precedence rule); a
    # 400 for an "unknown corpus" in a session request would mislead callers
    # into thinking the corpus mattered.
    if session_record is None:
        logger.info("query_routed", route="public", corpus=body.corpus)
        if body.corpus != "default" and body.corpus not in list_corpora():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown corpus: {body.corpus!r}. Registered: {sorted(list_corpora())}",
            )

    async def event_stream() -> AsyncIterator[str]:
        """
        Generator function that yields SSE-formatted tokens.
        Each token is sent as:  data: <token text>\n\n
        """
        try:
            # Call the streaming version of the RAG pipeline
            async for token in orchestrator.stream_query(
                body, store_override=store_override,
            ):
                # Format as Server-Sent Event
                yield f"data: {token}\n\n"

            # Signal to the client that the stream is complete
            yield "data: [DONE]\n\n"

        except Exception as e:
            # Send error message to client then close stream
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            # Unpin AFTER the stream is fully consumed (or errored). Until
            # this point the reaper cannot evict — record.store remains
            # valid for orchestrator.stream_query's retrieval calls.
            if session_record is not None:
                session_store.unpin(session_record.token)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",  # Required for SSE
        headers={
            "Cache-Control": "no-cache",      # Don't cache streaming responses
            "X-Accel-Buffering": "no",        # Disable nginx buffering for streams
        },
    )


# =============================================================================
# CORPORA ENDPOINTS — Multi-corpus discovery
# =============================================================================

@app.get("/corpora", response_model=CorporaListResponse, tags=["query"])
async def list_registered_corpora():
    """
    List corpora registered at startup with static metadata and live counts.

    `source` (a human-readable provenance label) and `chunker` are populated
    for corpora that have a config entry in `config/corpora.py` (the six Apple
    corpora). The `default` corpus is runtime-bound to a FAISS-backed FiQA
    collection and has no config record, so its `source` and `chunker` are null.

    `doc_count` is read live from each registered store via its count()
    method — for FAISS this is index.ntotal, for Chroma it is the
    collection's row count.
    """
    items: list[CorpusInfo] = []
    for name in list_corpora():
        cfg = CORPORA_CONFIG.get(name)
        store = get_vector_store() if name == "default" else get_corpus(name)
        # `source` is a human-readable provenance label (see config/corpora.py),
        # not a path into the image, so it is safe to advertise verbatim. The
        # local ingest paths live in `ingest_path`/`ingest_paths` and are never
        # exposed here.
        items.append(CorpusInfo(
            name=name,
            source=cfg.get("source") if cfg else None,
            chunker=cfg.get("chunker") if cfg else None,
            doc_count=store.count(),
        ))
    return CorporaListResponse(corpora=items)


# =============================================================================
# AGENT ENDPOINTS — LangGraph-based agent API
# =============================================================================

# In-memory trace store (can be upgraded to Redis for distributed systems)
_trace_store: dict[str, QueryTrace] = {}

_agent_graph = None


def _get_agent_graph():
    """Get or initialize the agent graph."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_graph()
    return _agent_graph


@app.post(
    "/agent/query",
    response_model=AgentQueryResponse,
    tags=["agent"],
    summary="Query the LangGraph agent",
    description="Send a question to the intelligent agent graph with memory and tool support."
)
async def agent_query(request: AgentQueryRequest):
    """
    Query the LangGraph agent with support for:
    - Multi-step reasoning with tools
    - Conversation history and memory
    - Configurable retrieval strategies
    - Confidence-based retry logic

    Request schema:
        query: str — the user's question
        session_id: optional str — session ID for memory context
        metadata_filter: optional dict — document filters
        trace_enabled: bool — enable request tracing

    Response includes:
        answer: str — generated answer
        citations: list — source references
        confidence: float — answer confidence (0-1)
        retry_count: int — number of retries
        model_used: str — LLM model name
        latency_ms: float — processing time
        trace_id: str — unique trace identifier
    """
    import uuid
    from datetime import datetime as dt

    trace_id = str(uuid.uuid4())
    request_start = dt.utcnow()

    try:
        # Build initial state from request
        state = initial_state(
            query=request.query,
            metadata_filter=request.metadata_filter,
        )
        if request.session_id:
            state["metadata"] = {"session_id": request.session_id}

        graph = _get_agent_graph()
        result = await graph.ainvoke(state)

        # Build response
        end_time = dt.utcnow()
        latency_ms = (end_time - request_start).total_seconds() * 1000

        response = AgentQueryResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.0),
            retry_count=result.get("retry_count", 0),
            model_used=result.get("model_used", "unknown"),
            latency_ms=latency_ms,
            trace_id=trace_id,
        )

        # Store trace if enabled
        if request.trace_enabled:
            trace = QueryTrace(
                trace_id=trace_id,
                query=request.query,
                session_id=request.session_id,
                start_time=request_start,
                end_time=end_time,
                total_duration_ms=latency_ms,
                events=[],
                final_answer=result.get("answer", ""),
                confidence=result.get("confidence", 0.0),
                status="completed",
            )
            _trace_store[trace_id] = trace

        return response

    except Exception as e:
        logger.error("Agent query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent query failed: {str(e)}"
        )


@app.get(
    "/agent/trace/{trace_id}",
    response_model=QueryTrace,
    tags=["agent"],
    summary="Get trace for agent query",
    description="Retrieve detailed execution trace for a previous agent query."
)
async def get_agent_trace(trace_id: str):
    """
    Retrieve the execution trace for a specific agent query.

    The trace includes:
    - Query and response information
    - Sequence of node executions
    - Timing and status for each step
    - Final answer and confidence

    Args:
        trace_id: Unique identifier returned in the query response

    Returns:
        QueryTrace object with full execution details

    Raises:
        404: If trace_id not found
    """
    trace = _trace_store.get(trace_id)
    if not trace:
        raise HTTPException(
            status_code=404,
            detail=f"Trace {trace_id} not found"
        )
    return trace


@app.get(
    "/trace/{trace_id}",
    response_model=QueryTrace,
    tags=["tracing"],
    summary="Get trace for any query",
    description="Retrieve detailed execution trace for a query (orchestrator or agent)."
)
async def get_trace(trace_id: str):
    """
    Retrieve the execution trace for a specific query.

    Checks both agent traces and orchestrator traces.

    Args:
        trace_id: Unique identifier returned in the query response

    Returns:
        QueryTrace object with full execution details

    Raises:
        404: If trace_id not found
    """
    # Check agent traces first
    trace = _trace_store.get(trace_id)
    if trace:
        return trace

    # Check orchestrator traces via tracer
    tracer = get_tracer()
    trace = tracer.get_trace(trace_id)
    if trace:
        return trace

    raise HTTPException(
        status_code=404,
        detail=f"Trace {trace_id} not found"
    )
