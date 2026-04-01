"""
LLM generation layer.

Features:
  - Multi-provider (OpenAI, Anthropic, Together)
  - Streaming support
  - Response-level caching (Redis)
  - Token usage tracking
  - Structured output (answer + citations JSON)
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from config.settings import settings, LLMProvider
from utils.models import Citation, GenerationResult, QueryType, RetrievalStrategy
from generation.prompts.prompt_builder import ConstructedPrompt

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: ConstructedPrompt) -> tuple[str, int]:
        """Returns (answer_text, total_tokens)."""
        ...

    @abstractmethod
    async def stream(self, prompt: ConstructedPrompt) -> AsyncIterator[str]:
        """Yields answer text chunks as they stream."""
        ...


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = None):
        self.model = model or settings.llm_model

    async def generate(self, prompt: ConstructedPrompt) -> tuple[str, int]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=self.model,
            messages=prompt.messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = response.choices[0].message.content or ""
        total_tokens = response.usage.total_tokens if response.usage else 0
        return answer, total_tokens

    async def stream(self, prompt: ConstructedPrompt) -> AsyncIterator[str]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=prompt.messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    async def generate(self, prompt: ConstructedPrompt) -> tuple[str, int]:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        # Convert OpenAI-format messages to Anthropic format
        system = next((m["content"] for m in prompt.messages if m["role"] == "system"), "")
        user_messages = [m for m in prompt.messages if m["role"] != "system"]

        response = await client.messages.create(
            model=self.model,
            system=system,
            messages=user_messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        answer = response.content[0].text
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        return answer, total_tokens

    async def stream(self, prompt: ConstructedPrompt) -> AsyncIterator[str]:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        system = next((m["content"] for m in prompt.messages if m["role"] == "system"), "")
        user_messages = [m for m in prompt.messages if m["role"] != "system"]

        async with client.messages.stream(
            model=self.model,
            system=system,
            messages=user_messages,
            max_tokens=settings.llm_max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text



class GroqLLM(BaseLLM):
    """Free LLM via Groq API. Get key at console.groq.com"""

    def __init__(self, model: str = None):
        self.model = model or settings.llm_model or "llama-3.3-70b-versatile"

    async def generate(self, prompt: ConstructedPrompt) -> tuple[str, int]:
        try:
            from groq import AsyncGroq
        except ImportError:
            raise ImportError("Run: pip install groq")
        client = AsyncGroq(api_key=settings.groq_api_key)
        response = await client.chat.completions.create(
            model=self.model,
            messages=prompt.messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = response.choices[0].message.content or ""
        total_tokens = response.usage.total_tokens if response.usage else 0
        return answer, total_tokens

    async def stream(self, prompt: ConstructedPrompt):
        try:
            from groq import AsyncGroq
        except ImportError:
            raise ImportError("Run: pip install groq")
        client = AsyncGroq(api_key=settings.groq_api_key)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=prompt.messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GenerationService:
    """
    Orchestrates prompt → LLM → formatted response.
    Handles caching, token tracking, and citation formatting.
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        self._llm = llm or self._build_llm()
        self._redis = None

    def _build_llm(self) -> BaseLLM:
        provider = settings.llm_provider.value if hasattr(settings.llm_provider, 'value') else str(settings.llm_provider)
        if provider == "groq":
            return GroqLLM()
        elif provider == "anthropic":
            return AnthropicLLM()
        return OpenAILLM()

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                r = aioredis.from_url(settings.redis_url)
                await r.ping()
                self._redis = r
            except Exception:
                self._redis = "unavailable"
        return None if self._redis == "unavailable" else self._redis

    def _cache_key(self, query: str, chunks_hash: str) -> str:
        payload = f"{settings.llm_model}:{query}:{chunks_hash}"
        return f"gen:{hashlib.sha256(payload.encode()).hexdigest()}"

    async def generate(
        self,
        query: str,
        prompt: ConstructedPrompt,
        query_type: QueryType,
        strategy_used: RetrievalStrategy,
    ) -> GenerationResult:
        start = time.monotonic()

        # Check cache
        chunks_hash = hashlib.md5(
            "".join(c.chunk_id for c in prompt.citations).encode()
        ).hexdigest()
        cache_key = self._cache_key(query, chunks_hash)

        try:
            redis = await self._get_redis()
            cached = await redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return GenerationResult(
                    answer=data["answer"],
                    citations=prompt.citations,
                    query_type=query_type,
                    strategy_used=strategy_used,
                    model_used=settings.llm_model,
                    total_tokens=data.get("total_tokens", 0),
                    latency_ms=(time.monotonic() - start) * 1000,
                    cached=True,
                )
        except Exception as e:
            logger.debug("Cache read failed: %s", e)

        # Generate
        answer, total_tokens = await self._llm.generate(prompt)

        latency_ms = (time.monotonic() - start) * 1000

        # Cache result
        try:
            redis = await self._get_redis()
            await redis.setex(
                cache_key,
                settings.cache_ttl_query,
                json.dumps({"answer": answer, "total_tokens": total_tokens}),
            )
        except Exception as e:
            logger.debug("Cache write failed: %s", e)

        return GenerationResult(
            answer=answer,
            citations=prompt.citations,
            query_type=query_type,
            strategy_used=strategy_used,
            model_used=settings.llm_model,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cached=False,
        )

    async def stream(
        self,
        query: str,
        prompt: ConstructedPrompt,
    ) -> AsyncIterator[str]:
        async for chunk in self._llm.stream(prompt):
            yield chunk


def get_generation_service() -> GenerationService:
    return GenerationService()
