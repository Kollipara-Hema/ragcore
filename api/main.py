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
    GET  /ingest/status/{id}  Check if a file upload has finished processing
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
from ingestion.pipeline import IngestionPipeline, ingest_file_task

# Vector store (for document deletion and health checks)
from vectorstore.vector_store import (
    FAISSVectorStore,
    get_corpus,
    get_vector_store,
    list_corpora,
    register_corpus,
)
from vectorstore.chroma_store import ChromaVectorStore

# Apple multi-corpus demo configuration
from config.corpora import CORPORA_CONFIG

# Embedder (for health checks)
from embeddings.embedder import get_embedder

# Rate limiting middleware (prevents abuse)
from api.middleware.rate_limit import RateLimitMiddleware

# API key authentication middleware (optional, gated by settings.auth_enabled)
from api.middleware.auth import APIKeyAuthMiddleware

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
    global orchestrator, ingestion_pipeline

    configure_logging()
    logger.info("Starting RAG system...")

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

    logger.info("RAG system ready. Visit http://localhost:8000/docs")

    yield  # Server is now running — handle requests

    # Cleanup on shutdown (close connections, etc.)
    logger.info("Shutting down RAG system...")


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

# CORS Middleware — allows browsers from any origin to call this API
# In production, replace allow_origins=["*"] with your specific frontend URL
# e.g. allow_origins=["https://your-app.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Request-Id", "X-API-Key"],
)

# API Key Authentication Middleware — only registered when auth is enabled.
# Placed between CORS and RateLimit: execution order is
# RequestId → RateLimit → Auth → CORS → handler, so rate-limiting applies
# before key validation and bad-key floods are still throttled.
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

# Rate Limiting Middleware — prevents a single user from spamming the API
# Defined in api/middleware/rate_limit.py
# Configurable via RAGCORE_RATE_LIMIT_MAX_REQUESTS / RAGCORE_RATE_LIMIT_WINDOW_SECONDS
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.ragcore_rate_limit_max_requests,
    window_seconds=settings.ragcore_rate_limit_window_seconds,
    trust_proxy_headers=settings.ragcore_trust_proxy_headers,
)

# Request-ID Middleware — registered last so it executes first (outermost).
# All downstream log calls run inside its bound_contextvars context and
# therefore automatically carry request_id in their JSON output.
app.add_middleware(RequestIdMiddleware)


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
# TOGETHER is defined in LLMProvider but has no key field in Settings — handled explicitly.
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

@app.post("/ingest/file", response_model=IngestResponse, tags=["ingestion"])
async def ingest_file(
    file: UploadFile = File(...),               # The uploaded file — REQUIRED
    title: Optional[str] = Form(None),          # Optional document title
    tags: Optional[str] = Form(None),           # Optional comma-separated tags
    async_processing: bool = Form(True),        # True = background, False = wait
):
    """
    Upload a document file to be indexed.

    Supported file types: PDF, TXT, MD, DOCX, HTML, CSV

    async_processing=true  (default):
        Returns immediately with a job_id.
        Use GET /ingest/status/{job_id} to check when it's done.
        Best for large files or production use.

    async_processing=false:
        Waits until indexing is complete before responding.
        Best for small files and demos where you want immediate feedback.

    Example:
        curl -X POST http://localhost:8000/ingest/file
             -F "file=@report.pdf"
             -F "title=Q3 Report"
             -F "async_processing=false"
    """

    # List of allowed MIME types — reject anything else
    # MIME types are the official file type identifiers
    allowed_types = {
        "application/pdf",          # PDF files
        "text/plain",               # .txt files
        "text/markdown",            # .md files
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "text/html",                # .html files
        "text/csv",                 # .csv files
    }

    # Check if the uploaded file type is allowed
    if file.content_type not in allowed_types:
        # Return 400 Bad Request with a descriptive error message
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: PDF, TXT, MD, DOCX, HTML, CSV"
        )

    # Save the uploaded file to a temporary location on disk
    # We need a physical file path because our loaders read from disk
    suffix = os.path.splitext(file.filename or "upload")[1]  # e.g. ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()   # Read all bytes from the upload
        tmp.write(content)            # Write to temp file
        tmp_path = tmp.name           # Save the temp file path

    # Build metadata dict from the form fields
    metadata = {
        "title": title or file.filename,  # Use provided title or filename
        "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],  # Parse tags
        "original_filename": file.filename,  # Keep the original name for reference
    }

    # Generate a placeholder doc_id (the real one comes from the pipeline)
    doc_id = str(uuid.uuid4())

    if async_processing and ingest_file_task:
        # ── ASYNC PATH: Queue as background Celery task ───────────────────
        # Returns immediately — the file will be processed in the background
        # The user polls /ingest/status/{job_id} to check progress
        job = ingest_file_task.delay(tmp_path, metadata)
        # .delay() sends the task to the Celery worker via Redis

        return IngestResponse(
            job_id=job.id,             # Celery task ID for status polling
            doc_id=doc_id,             # Placeholder — real ID set by pipeline
            status=DocumentStatus.PENDING,  # Not indexed yet
            message=f"File queued for processing. "
                    f"Poll /ingest/status/{job.id} to check progress.",
        )

    else:
        # ── SYNC PATH: Process immediately, wait for completion ───────────
        # Blocks until the file is fully indexed. May take 10-60 seconds for large files.
        try:
            actual_doc_id, chunk_count = await ingestion_pipeline.ingest_file(
                tmp_path, metadata
            )
            return IngestResponse(
                job_id="sync",              # No background job
                doc_id=actual_doc_id,       # Real doc_id from the pipeline
                status=DocumentStatus.INDEXED,  # Fully indexed
                message=f"Successfully indexed {chunk_count} chunks.",
            )
        except Exception as e:
            logger.error("Synchronous ingestion failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
        finally:
            # Always delete the temp file, even if ingestion failed
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@app.post("/ingest/text", response_model=IngestResponse, tags=["ingestion"])
async def ingest_text(request: IngestRequest):
    """
    Index raw text content directly without uploading a file.

    Useful for:
    - Indexing text extracted from your own data sources
    - Testing the system without preparing files
    - Indexing dynamic content (web scrapes, database records)

    Example:
        curl -X POST http://localhost:8000/ingest/text
             -H "Content-Type: application/json"
             -d '{"text_content": "Your document text here...",
                  "metadata": {"title": "My Document", "tags": ["test"]}}'
    """
    # Validate that text_content was actually provided
    if not request.text_content:
        raise HTTPException(status_code=400, detail="text_content is required")

    try:
        # Run ingestion pipeline directly on the text string
        doc_id, chunk_count = await ingestion_pipeline.ingest_text(
            request.text_content,
            request.metadata  # Any extra metadata (title, author, tags, etc.)
        )
        return IngestResponse(
            job_id="sync",
            doc_id=doc_id,
            status=DocumentStatus.INDEXED,
            message=f"Successfully indexed {chunk_count} chunks.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/status/{job_id}", tags=["ingestion"])
async def ingest_status(job_id: str):
    """
    Check the processing status of an async file ingestion job.

    States:
        PENDING   = Task queued, not started yet
        STARTED   = Currently being processed
        SUCCESS   = Finished successfully — document is now searchable
        FAILURE   = Something went wrong

    Example:
        curl http://localhost:8000/ingest/status/abc-123-def
    """
    try:
        from celery.result import AsyncResult
        from ingestion.pipeline import celery_app

        # Check if Celery is configured
        if not celery_app:
            raise HTTPException(
                status_code=503,
                detail="Async processing not configured. Start with async_processing=false."
            )

        # Look up the task status in Redis
        result = AsyncResult(job_id, app=celery_app)

        return {
            "job_id": job_id,
            "state": result.state,   # PENDING | STARTED | SUCCESS | FAILURE
            # result.result contains the return value if SUCCESS,
            # or the error message if FAILURE
            "result": result.result if result.ready() else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
async def query(request: QueryRequest):
    """
    Ask a question about your indexed documents.

    The system automatically:
    1. Classifies the query type (factual, analytical, multi-hop, etc.)
    2. Selects the best retrieval strategy (hybrid, keyword, semantic)
    3. Retrieves relevant document chunks
    4. Reranks chunks by relevance
    5. Generates a grounded, cited answer

    Request body options:
        query           (required) — your question
        top_k           (optional) — how many chunks to retrieve (default from .env)
        metadata_filter (optional) — filter by doc type, author, etc.
        strategy_override (optional) — force a specific retrieval strategy
        stream          (optional) — use /query/stream endpoint instead for streaming

    Example:
        curl -X POST http://localhost:8000/query
             -H "Content-Type: application/json"
             -d '{"query": "What were the main findings in the Q3 report?"}'
    """
    try:
        # Run the full RAG pipeline and return the complete answer
        result = await orchestrator.query(request)
        return result
    except KeyError as ke:
        # Raised by get_corpus() when request.corpus names an unregistered
        # corpus. Message is curated upstream and names the registered set
        # so callers can spot a typo.
        raise HTTPException(status_code=400, detail=str(ke))
    except Exception as e:
        logger.error("query_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/retrieve", response_model=RetrievalOnlyResponse, tags=["query"])
async def retrieve(request: QueryRequest):
    """
    Retrieval-only endpoint: router → retrieve → rerank, NO generation.

    Consumes ZERO LLM tokens — generation is skipped entirely. Used by the
    chunker comparison view and other retrieval-debugging surfaces where the
    generated answer isn't needed. Accepts the same QueryRequest as /query
    (verify_claims is a no-op here, since no generation runs). Returns
    retrieval_candidates, stage_timings (retrieve + rerank only),
    strategy_used, query_type, and a top_rerank_score convenience field.

    Example:
        curl -X POST http://localhost:8000/retrieve
             -H "Content-Type: application/json"
             -d '{"query": "What is a 401k?", "corpus": "default"}'
    """
    try:
        return await orchestrator.retrieve_only(request)
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=str(ke))
    except Exception as e:
        logger.error("retrieve_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval processing failed: {str(e)}"
        )


@app.post("/query/stream", tags=["query"])
async def query_stream(request: QueryRequest):
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
    # Pre-validate corpus membership before constructing the StreamingResponse:
    # once streaming begins the HTTP status is committed as 200, so an
    # unknown-corpus KeyError raised mid-stream can only be surfaced as an
    # in-band [ERROR] event, not as the 400 the caller deserves. /query and
    # /query/stream therefore differ here by HTTP-semantic necessity, not by
    # an accidental inconsistency — /query catches KeyError post-hoc; this
    # endpoint must check up front.
    if request.corpus != "default" and request.corpus not in list_corpora():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown corpus: {request.corpus!r}. Registered: {sorted(list_corpora())}",
        )

    async def event_stream() -> AsyncIterator[str]:
        """
        Generator function that yields SSE-formatted tokens.
        Each token is sent as:  data: <token text>\n\n
        """
        try:
            # Call the streaming version of the RAG pipeline
            async for token in orchestrator.stream_query(request):
                # Format as Server-Sent Event
                yield f"data: {token}\n\n"

            # Signal to the client that the stream is complete
            yield "data: [DONE]\n\n"

        except Exception as e:
            # Send error message to client then close stream
            yield f"data: [ERROR] {str(e)}\n\n"

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

    `source` and `chunker` are populated for corpora that have a config entry
    in `config/corpora.py` (the six Apple corpora). The `default` corpus is
    runtime-bound to a FAISS-backed FiQA collection and has no config record,
    so its `source` and `chunker` are null.

    `doc_count` is read live from each registered store via its count()
    method — for FAISS this is index.ntotal, for Chroma it is the
    collection's row count.
    """
    items: list[CorpusInfo] = []
    for name in list_corpora():
        cfg = CORPORA_CONFIG.get(name)
        store = get_vector_store() if name == "default" else get_corpus(name)
        # Some CORPORA_CONFIG entries (apple_financial_csvs) use a list of
        # `sources` rather than a single `source`. Serialise as a CSV string
        # so the response field stays scalar.
        if cfg is None:
            source = None
        elif "sources" in cfg:
            source = ", ".join(cfg["sources"])
        else:
            source = cfg.get("source")
        items.append(CorpusInfo(
            name=name,
            source=source,
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
