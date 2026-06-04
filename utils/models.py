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
    pre_rerank_score: Optional[float] = None


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
    input_tokens: int = 0
    output_tokens: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# API Request/Response models (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = PField(..., min_length=3, max_length=2000)
    top_k: Optional[int] = PField(default=None, ge=1, le=50)
    metadata_filter: Optional[dict[str, Any]] = None
    strategy_override: Optional[RetrievalStrategy] = None
    stream: bool = False
    verify_claims: bool = PField(
        default=False,
        description="Opt in to Self-RAG per-claim verification. ~3x slower.",
    )
    corpus: str = PField(
        default="default",
        description="Which registered corpus to query. Omitting routes to the 'default' corpus.",
    )


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
    follow_up_questions: Optional[list[str]] = None
    stage_timings: Optional[dict] = None
    # Shape: {"router_ms": float, "retrieve_ms": float, "rerank_ms": float,
    #         "prompt_ms": float, "generate_ms": float, "total_ms": float}
    retrieval_candidates: Optional[list[dict]] = None
    # Shape: [{"rank": int, "doc_id": str, "chunk_id": str, "source": str,
    #          "score": float, "used_in_answer": bool, "excerpt": str}]
    # Candidates from retrieval BEFORE reranking; used_in_answer = True when
    # the chunk_id appears in final citations.
    attributed_spans: Optional[list[dict]] = None
    # Shape: [{"source": int, "start": int, "end": int, "text": str}]
    # Character offsets into answer. Produced by trailing-marker heuristic;
    # None when LLM emitted no <cite source="N"> markers.


class RetrievalOnlyResponse(BaseModel):
    """Response from POST /retrieve — retrieval + rerank only, no generation,
    no LLM tokens consumed. Mirrors the retrieval-side fields of QueryResponse
    so clients comparing retrieval (e.g. the chunker comparison view) can
    share their projection code."""
    query_type: str
    strategy_used: str
    latency_ms: float
    stage_timings: Optional[dict] = None
    # Shape: {"router_ms", "retrieve_ms", "rerank_ms", "total_ms"}.
    # No prompt_ms / generate_ms — those stages don't run here.
    retrieval_candidates: list[dict]
    # Same shape as QueryResponse.retrieval_candidates. CRITICAL: the
    # used_in_answer field has a DIFFERENT meaning here than in /query.
    # In /query, True means "the LLM cited this chunk in its answer." In
    # /retrieve, no LLM is called — True means "this chunk survived rerank
    # into the top-K that would have been packed into the prompt." Same
    # field name across both endpoints, but the semantics diverge; this
    # comment is the only thing preventing future conflation.
    top_rerank_score: Optional[float] = None
    # Convenience: max(retrieval_candidates[].score) — the post-rerank
    # cross-encoder logit of the top chunk. Equivalent to citations[0].score
    # in /query responses, but citations aren't built when generation is
    # skipped, so callers reading "top rerank score" from /query need to
    # source it from retrieval_candidates here instead.


class CorpusInfo(BaseModel):
    name: str
    source: Optional[str] = None
    # `source` and `chunker` come from config/corpora.py; both are None for
    # the runtime-bound "default" corpus which has no static config record.
    chunker: Optional[str] = None
    doc_count: int


class CorporaListResponse(BaseModel):
    corpora: list[CorpusInfo]


class IngestRequest(BaseModel):
    source_url: Optional[str] = None
    text_content: Optional[str] = PField(default=None, max_length=1_000_000)
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
