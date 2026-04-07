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
from pydantic_settings import BaseSettings


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"   # Paid, high quality
    BGE = "bge"         # Free, local, recommended
    COHERE = "cohere"   # Paid, multilingual


class VectorStoreProvider(str, Enum):
    WEAVIATE = "weaviate"   # Free self-hosted, best features
    PINECONE = "pinecone"   # Paid managed cloud
    QDRANT = "qdrant"       # Free self-hosted, fast
    CHROMA = "chroma"       # Free local (dev only)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    GROQ = "groq"


class Settings(BaseSettings):

    # App
    app_name: str = "DocIntel RAG"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = False

    # Embedding
    embedding_provider: EmbeddingProvider = EmbeddingProvider.BGE
    embedding_model_openai: str = "text-embedding-3-large"
    embedding_model_bge: str = "BAAI/bge-large-en-v1.5"
    embedding_model_cohere: str = "embed-english-v3.0"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 64

    # Vector Store
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.WEAVIATE
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "docintel"
    qdrant_url: str = "http://localhost:6333"
    chroma_persist_dir: str = "./chroma_db"

    # LLM
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_streaming: bool = True
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    # Chunking
    # Original: fixed | semantic | hierarchical | sentence
    # Phase 1:  propositional | table_aware | structure
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
    hybrid_alpha: float = 0.7  # Note: set strategy_override to semantic when using Chroma

    # Generation Strategy — Phase 5
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
