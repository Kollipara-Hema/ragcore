"""
=============================================================================
config/settings.py — Central Configuration (Updated for Phases 1-5)
=============================================================================
YOU DO NOT CHANGE THIS FILE.
All your settings go in the .env file.

New settings added for Phases 1-5:
    CHUNKING_STRATEGY   — now includes: propositional, table_aware, structure
    GENERATION_STRATEGY — new: basic | self_rag | flare | agentic
=============================================================================
"""

from enum import Enum
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, EnvSettingsSource, DotEnvSettingsSource


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"   # Paid, high quality
    BGE = "bge"         # Free, local, recommended
    COHERE = "cohere"   # Paid, multilingual


class VectorStoreProvider(str, Enum):
    WEAVIATE = "weaviate"   # Free self-hosted, best features
    PINECONE = "pinecone"   # Paid managed cloud
    QDRANT = "qdrant"       # Free self-hosted, fast
    CHROMA = "chroma"       # Free local (dev only)
    FAISS = "faiss"         # In-process local (deployed default)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    GROQ = "groq"


# pydantic-settings v2 treats list[str] as "complex" and tries to JSON-parse env
# var values before field validators run, making field_validator ineffective.
# These source subclasses intercept decode_complex_value for cors_origins so that
# comma-separated strings (e.g. "https://a.com,https://b.com") work alongside
# the JSON format (e.g. '["https://a.com"]') that pydantic-settings natively supports.
class _CorsAwareMixin:
    def decode_complex_value(self, field_name, field_info, value):
        if field_name == "cors_origins" and isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return super().decode_complex_value(field_name, field_info, value)


class _CorsEnvSource(_CorsAwareMixin, EnvSettingsSource):
    pass


class _CorsDotEnvSource(_CorsAwareMixin, DotEnvSettingsSource):
    pass


class Settings(BaseSettings):

    # App
    app_name: str = "DocIntel RAG"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = False

    # Security
    cors_origins: list[str] = Field(
        default=["http://localhost:8501"],
        description="Allowed CORS origins. Comma-separated in env var.",
    )
    ragcore_auth_enabled: bool = Field(
        default=False,
        description="Require X-API-Key header on protected endpoints.",
    )
    ragcore_api_key: Optional[str] = Field(
        default=None,
        description="Shared API key. Required when ragcore_auth_enabled=True.",
    )
    ragcore_rate_limit_max_requests: int = Field(
        default=60,
        description="Max requests per IP per window. Env: RAGCORE_RATE_LIMIT_MAX_REQUESTS.",
    )
    ragcore_rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limit sliding window in seconds. Env: RAGCORE_RATE_LIMIT_WINDOW_SECONDS.",
    )
    ragcore_trust_proxy_headers: bool = Field(
        default=False,
        description="Trust X-Forwarded-For for client IP. Only enable behind a trusted reverse proxy.",
    )
    ragcore_ingest_max_body_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Hard ceiling on /ingest/* request body size. Returns 413 above this.",
    )
    ragcore_session_root: str = Field(
        default="./data/sessions",
        description="Root directory for per-session corpora. SEPARATE from "
                    "faiss_data_dir and chroma_persist_dir so session cleanup "
                    "can never touch curated corpora.",
    )
    # The three caps below are scaffolding shaped for the 2 GB Render box.
    # HF migration bumps these via env vars (RAGCORE_SESSION_MAX_*), not by
    # editing this file — keep values out of code.
    ragcore_session_max_file_bytes: int = Field(
        default=1 * 1024 * 1024,
        description="Per-file size cap inside a session ingest. Tighter than "
                    "ragcore_ingest_max_body_bytes (which guards transport). "
                    "Returns 413 above this.",
    )
    ragcore_session_max_files: int = Field(
        default=3,
        description="Max files per session. Returns 409 above this.",
    )
    ragcore_session_max_concurrent: int = Field(
        default=3,
        description="Max concurrent sessions per process. Returns 503 above this.",
    )
    ragcore_pdf_max_pages: int = Field(
        default=100,
        description=(
            "Max page count for an uploaded PDF. Probed via pymupdf.open "
            "(metadata read only — no text extraction) BEFORE the embed "
            "pipeline runs. Returns 413 when exceeded. Inclusive — a PDF "
            "with exactly N pages is accepted."
        ),
    )

    # Embedding
    embedding_provider: EmbeddingProvider = EmbeddingProvider.BGE
    embedding_model_openai: str = "text-embedding-3-large"
    embedding_model_bge: str = "BAAI/bge-large-en-v1.5"
    embedding_model_cohere: str = "embed-english-v3.0"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 64

    # Vector Store
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.FAISS
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "docintel"
    qdrant_url: str = "http://localhost:6333"
    chroma_persist_dir: str = "./data/chroma_collections"
    chroma_collection_name: str = "ragcore"
    faiss_data_dir: str = "./faiss"

    # LLM
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_streaming: bool = True
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # Chunking
    # Basic:    fixed | semantic | hierarchical | sentence
    # Advanced: propositional | table_aware | structure
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: str = "semantic"

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_query_expansion: bool = False
    max_expanded_queries: int = 3
    hybrid_alpha: float = 0.7

    # Generation Strategy
    # basic    = standard RAG (fast, cheapest)                         ← start here
    # self_rag = verify claims after generation (2-3x cost, better accuracy)
    # flare    = retrieve mid-generation when uncertain (best for long answers)
    # agentic  = multi-turn agent (most powerful, slowest)
    generation_strategy: str = "basic"

    # Self-RAG settings
    self_rag_max_additional_retrievals: int = 2

    # FLARE settings
    flare_confidence_threshold: float = 0.6
    flare_max_retrieval_rounds: int = 3

    # Agentic settings
    agentic_max_turns: int = 5

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_query: int = 3600
    cache_ttl_embedding: int = 86400

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    ingestion_concurrency: int = 4

    # Monitoring
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = False

    # Evaluation
    enable_evaluation: bool = False
    eval_strategy: str = "heuristic"  # heuristic | ragas
    ragas_enabled: bool = False

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (
            init_settings,
            _CorsEnvSource(settings_cls),
            _CorsDotEnvSource(settings_cls),
            file_secret_settings,
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
