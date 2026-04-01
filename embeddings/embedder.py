"""
Multi-provider embedding service.
Defaults to free sentence-transformers (no API key required).
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
from typing import List, Optional, Union

from config.settings import settings, EmbeddingProvider
from utils.models import Chunk

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """Base embedding interface."""

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return embeddings."""
        raise NotImplementedError

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        raise NotImplementedError

    async def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Embed chunk objects in place."""
        embeddings = await self.embed_texts([chunk.content for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small."""

    def __init__(self, model: str = None, batch_size: int = None):
        self.model = model or settings.embedding_model_openai
        self.batch_size = batch_size or settings.embedding_batch_size

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        all_embeddings: List[List[float]] = []

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

    async def embed_query(self, query: str) -> List[float]:
        return (await self.embed_texts([query]))[0]


class BGEEmbedder(BaseEmbedder):
    """Uses sentence-transformers (BAAI/bge-large-en-v1.5)."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = None, batch_size: int = None):
        self.model_name = model_name or settings.embedding_model_bge
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading BGE model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, texts)

    def _sync_embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        prefixed = self.QUERY_PREFIX + query
        return (await self.embed_texts([prefixed]))[0]


class CohereEmbedder(BaseEmbedder):
    """Cohere embedder for multilingual use."""

    def __init__(self, model: str = None, batch_size: int = None):
        self.model = model or settings.embedding_model_cohere
        self.batch_size = batch_size or settings.embedding_batch_size

    async def embed_texts(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        try:
            import cohere
        except ImportError:
            raise ImportError("Run: pip install cohere")

        client = cohere.AsyncClient(api_key=settings.openai_api_key)
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        return (await self.embed_texts([query], input_type="search_query"))[0]


class CachedEmbedder(BaseEmbedder):
    """Wraps any embedder with Redis caching."""

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

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        redis = await self._get_redis()
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []

        if redis is None:
            return await self._embedder.embed_texts(texts)

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

    async def embed_query(self, query: str) -> List[float]:
        return (await self.embed_texts([query]))[0]


def get_embedder(provider: Union[str, EmbeddingProvider, None] = None, cached: bool = True) -> BaseEmbedder:
    if provider is None:
        provider = settings.embedding_provider

    if isinstance(provider, str):
        provider_key = provider.lower()
        if provider_key in ("minilm", "mini", "sentence-transformer", "sentence_transformer"):
            provider = EmbeddingProvider.BGE
        else:
            provider = EmbeddingProvider(provider_key)

    mapping = {
        EmbeddingProvider.OPENAI: OpenAIEmbedder,
        EmbeddingProvider.BGE: BGEEmbedder,
        EmbeddingProvider.COHERE: CohereEmbedder,
    }

    embedder_cls = mapping.get(provider)
    if not embedder_cls:
        raise ValueError(f"Unknown embedding provider: {provider}")

    embedder = embedder_cls()
    if cached:
        return CachedEmbedder(embedder)
    return embedder
