"""
Embedding service with three provider adapters:
  - OpenAI (text-embedding-3-large) — cloud, high quality
  - BGE (BAAI/bge-large-en-v1.5) — open source, on-prem
  - Cohere (embed-v3) — multilingual

All adapters implement BaseEmbedder.
Includes batching, caching, and retry logic.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config.settings import settings, EmbeddingProvider
from utils.models import Chunk

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """All embedding providers implement this interface."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of float vectors."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Can be overridden for query-specific instructions."""
        results = await self.embed_texts([query])
        return results[0]

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all chunks in-place, skip already embedded ones."""
        to_embed = [c for c in chunks if c.embedding is None]
        if not to_embed:
            return chunks

        texts = [c.content for c in to_embed]
        embeddings = await self.embed_texts(texts)

        for chunk, embedding in zip(to_embed, embeddings):
            chunk.embedding = embedding

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI adapter
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = None, batch_size: int = None):
        self.model = model or settings.embedding_model_openai
        self.batch_size = batch_size or settings.embedding_batch_size

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        all_embeddings: list[list[float]] = []

        # Batch to respect rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=settings.embedding_dimension if "3" in self.model else None,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        # OpenAI: no special instruction prefix needed
        return (await self.embed_texts([query]))[0]


# ─────────────────────────────────────────────────────────────────────────────
# BGE (local sentence-transformers) adapter
# ─────────────────────────────────────────────────────────────────────────────

class BGEEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers (BAAI/bge-large-en-v1.5).
    BGE requires a task instruction prefix for queries (not for docs).
    """
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = None, batch_size: int = None):
        self.model_name = model_name or settings.embedding_model_bge
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading BGE model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, texts)

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        # BGE prepends instruction for query embeddings
        prefixed = self.QUERY_PREFIX + query
        return (await self.embed_texts([prefixed]))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Cohere adapter
# ─────────────────────────────────────────────────────────────────────────────

class CohereEmbedder(BaseEmbedder):
    """
    Uses Cohere's embed-english-v3.0 or multilingual model.
    Cohere embed v3 uses input_type for document vs query distinction.
    """

    def __init__(self, model: str = None, batch_size: int = None):
        self.model = model or settings.embedding_model_cohere
        self.batch_size = batch_size or settings.embedding_batch_size

    async def embed_texts(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        try:
            import cohere
        except ImportError:
            raise ImportError("Run: pip install cohere")

        client = cohere.AsyncClient(api_key=settings.openai_api_key)  # reuse key field
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_texts([query], input_type="search_query"))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Cached wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CachedEmbedder(BaseEmbedder):
    """
    Wraps any embedder with Redis-based caching.
    Cache key = SHA256(model_name + text).
    """

    def __init__(self, embedder: BaseEmbedder):
        self._embedder = embedder
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                r = aioredis.from_url(settings.redis_url, decode_responses=False)
                await r.ping()
                self._redis = r
            except Exception:
                self._redis = "unavailable"
        return None if self._redis == "unavailable" else self._redis

    def _cache_key(self, text: str) -> str:
        model_name = type(self._embedder).__name__
        payload = f"{model_name}:{text}".encode()
        return f"embed:{hashlib.sha256(payload).hexdigest()}"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        redis = await self._get_redis()
        results: list[Optional[list[float]]] = [None] * len(texts)
        missing_indices: list[int] = []

        # Skip cache if Redis not available
        if redis is None:
            return await self._embedder.embed_texts(texts)

        # Check cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            try:
                cached = await redis.get(key)
                if cached:
                    results[i] = json.loads(cached)
                else:
                    missing_indices.append(i)
            except Exception:
                missing_indices.append(i)

        # Embed only cache misses
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = await self._embedder.embed_texts(missing_texts)
            for idx, embedding in zip(missing_indices, new_embeddings):
                results[idx] = embedding
                key = self._cache_key(texts[idx])
                try:
                    await redis.setex(key, settings.cache_ttl_embedding, json.dumps(embedding))
                except Exception:
                    pass

        return results  # type: ignore

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_texts([query]))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder(provider: EmbeddingProvider = None, cached: bool = True) -> BaseEmbedder:
    provider = provider or settings.embedding_provider
    mapping = {
        EmbeddingProvider.OPENAI: OpenAIEmbedder,
        EmbeddingProvider.BGE: BGEEmbedder,
        EmbeddingProvider.COHERE: CohereEmbedder,
    }
    cls = mapping.get(provider)
    if not cls:
        raise ValueError(f"Unknown embedding provider: {provider}")

    embedder = cls()
    if cached:
        return CachedEmbedder(embedder)
    return embedder
