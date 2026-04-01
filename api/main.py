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
import tempfile    # For saving uploaded files temporarily
import time        # For measuring response time
import uuid        # For generating unique IDs

# contextlib.asynccontextmanager — used for startup/shutdown lifecycle
from contextlib import asynccontextmanager
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
# JSONResponse     = returns data as JSON
# StreamingResponse = streams data token by token (for chat-like UX)

# --- Internal modules ---
from config.settings import settings    # App configuration from .env

# Data models for request/response shapes
from utils.models import IngestRequest, IngestResponse, DocumentStatus, QueryRequest

# The main RAG pipeline orchestrator
from orchestrator import RAGOrchestrator

# The document ingestion pipeline (loading, chunking, embedding, indexing)
from ingestion.pipeline import IngestionPipeline, ingest_file_task

# Vector store (for document deletion)
from vectorstore.vector_store import get_vector_store

# Rate limiting middleware (prevents abuse)
from api.middleware.rate_limit import RateLimitMiddleware

# Logger for this module
logger = logging.getLogger(__name__)

# =============================================================================
# Global instances — created once at startup, reused for all requests
# =============================================================================
# These are declared here but initialized in the lifespan function below.
# Using globals avoids recreating expensive objects (like loading ML models)
# on every request.

orchestrator: Optional[RAGOrchestrator] = None    # The RAG pipeline
ingestion_pipeline: Optional[IngestionPipeline] = None  # Document indexing


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

    logger.info("Starting RAG system...")

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

# =============================================================================
# MIDDLEWARE — Code that runs on EVERY request before and after the handler
# =============================================================================

# CORS Middleware — allows browsers from any origin to call this API
# In production, replace allow_origins=["*"] with your specific frontend URL
# e.g. allow_origins=["https://your-app.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (fine for demos)
    allow_credentials=True,     # Allow cookies and auth headers
    allow_methods=["*"],        # Allow GET, POST, DELETE, etc.
    allow_headers=["*"],        # Allow all request headers
)

# Rate Limiting Middleware — prevents a single user from spamming the API
# Defined in api/middleware/rate_limit.py
# Default: 60 requests per minute per IP address
app.add_middleware(RateLimitMiddleware)


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


@app.get("/metrics", tags=["system"])
async def metrics():
    """
    Metrics endpoint for monitoring.
    In production, integrate prometheus_client library for real metrics.

    Example: curl http://localhost:8000/metrics
    """
    return {"message": "Integrate prometheus_client for production metrics."}


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
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
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
