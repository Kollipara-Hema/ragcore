"""
Shared domain models used across ingestion, retrieval, and generation layers.
Using dataclasses + Pydantic for clean serialization and validation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field as PField


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    FACTUAL = "factual"          # "What is X?" — direct lookup
    SEMANTIC = "semantic"        # "Explain how X relates to Y"
    MULTI_HOP = "multi_hop"      # requires chaining across docs
    ANALYTICAL = "analytical"   # synthesis across multiple sources
    LOOKUP = "lookup"            # exact keyword / metadata filter
    COMPARATIVE = "comparative"  # compare A vs B


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"         # pure vector search
    KEYWORD = "keyword"           # BM25 / full-text
    HYBRID = "hybrid"             # vector + keyword fusion
    METADATA_FILTER = "metadata"  # structured filter + vector
    MULTI_QUERY = "multi_query"   # query expansion + merge
    PARENT_CHILD = "parent_child" # small-to-big retrieval


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Documents & Chunks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentMetadata:
    source: str                    # original file path or URL
    doc_type: str                  # "pdf", "docx", "txt", "html", "csv"
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    page_count: Optional[int] = None
    language: str = "en"
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    content: str
    metadata: DocumentMetadata
    doc_id: UUID = field(default_factory=uuid4)
    status: DocumentStatus = DocumentStatus.PENDING

    def to_dict(self) -> dict:
        return {
            "doc_id": str(self.doc_id),
            "content": self.content,
            "metadata": {
                "source": self.metadata.source,
                "doc_type": self.metadata.doc_type,
                "title": self.metadata.title,
                "author": self.metadata.author,
                "language": self.metadata.language,
                "tags": self.metadata.tags,
                **self.metadata.custom,
            },
        }


@dataclass
class Chunk:
    content: str
    doc_id: UUID
    chunk_id: UUID = field(default_factory=uuid4)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    # For hierarchical chunking: link to parent chunk
    parent_chunk_id: Optional[UUID] = None


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    strategy_used: RetrievalStrategy
    rank: int = 0


@dataclass
class RetrievalRequest:
    query: str
    query_type: QueryType
    strategy: RetrievalStrategy
    top_k: int = 20
    metadata_filter: Optional[dict[str, Any]] = None
    expanded_queries: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    request: RetrievalRequest
    chunks: list[RetrievedChunk]
    latency_ms: float = 0.0
    fallback_used: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    source: str
    title: Optional[str]
    excerpt: str        # short snippet shown in UI
    score: float


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]
    query_type: QueryType
    strategy_used: RetrievalStrategy
    model_used: str
    total_tokens: int
    latency_ms: float
    cached: bool = False
    fallback_used: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# API Request/Response models (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = PField(..., min_length=3, max_length=2000)
    top_k: Optional[int] = PField(default=None, ge=1, le=50)
    metadata_filter: Optional[dict[str, Any]] = None
    strategy_override: Optional[RetrievalStrategy] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    query_type: str
    strategy_used: str
    model_used: str
    total_tokens: int
    latency_ms: float
    cached: bool
    self_rag_stats: Optional[dict] = None


class IngestRequest(BaseModel):
    source_url: Optional[str] = None
    text_content: Optional[str] = None
    metadata: dict[str, Any] = PField(default_factory=dict)


class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    status: DocumentStatus
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Agent API models — for the LangGraph-based agent endpoints
# ─────────────────────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    query: str = PField(..., min_length=3, max_length=2000, description="User question")
    session_id: Optional[str] = PField(None, description="Optional session ID for memory context")
    metadata_filter: Optional[dict[str, Any]] = PField(None, description="Optional document filters")
    trace_enabled: bool = PField(True, description="Enable tracing for this query")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What are the main findings?",
            "session_id": "user-123",
            "metadata_filter": {"doc_type": "report"},
            "trace_enabled": True
        }
    }}


class AgentQueryResponse(BaseModel):
    answer: str = PField(..., description="Generated answer")
    citations: list[dict] = PField(default_factory=list, description="Source references")
    confidence: float = PField(..., ge=0.0, le=1.0, description="Answer confidence (0-1)")
    retry_count: int = PField(default=0, description="Number of retries performed")
    model_used: str = PField(..., description="LLM model used for generation")
    latency_ms: float = PField(..., description="Total processing latency in milliseconds")
    trace_id: str = PField(..., description="Unique trace ID for this request")


class TraceEvent(BaseModel):
    node: str = PField(..., description="Node name (router, retriever, etc.)")
    timestamp: datetime = PField(default_factory=datetime.utcnow)
    duration_ms: float = PField(default=0.0, description="Node execution time")
    status: str = PField(default="success", description="Status: success, error, skipped")
    details: dict[str, Any] = PField(default_factory=dict, description="Node-specific details")


class QueryTrace(BaseModel):
    trace_id: str = PField(..., description="Unique trace ID")
    query: str = PField(..., description="Original query")
    session_id: Optional[str] = None
    start_time: datetime = PField(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_duration_ms: float = PField(default=0.0, description="Total duration in milliseconds")
    events: list[TraceEvent] = PField(default_factory=list, description="Sequence of node events")
    final_answer: Optional[str] = None
    confidence: float = PField(default=0.0, description="Answer confidence")
    status: str = PField(default="pending", description="Query status: pending, completed, error")
