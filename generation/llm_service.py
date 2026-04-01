"""
LLM generation service.
"""
from __future__ import annotations
import logging
import os
import re
from typing import List, Dict, Any

import openai

logger = logging.getLogger(__name__)


class LLMService:
    """LLM generation with citations and streaming."""

    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, query: str, chunks: List[Dict[str, Any]], stream: bool = False) -> Dict[str, Any]:
        """Generate answer with citations."""
        if not os.getenv("OPENAI_API_KEY"):
            return self._mock_generate(query, chunks)

        # Build prompt
        context = "\n\n".join([f"[{i+1}] {chunk['text_preview']}" for i, chunk in enumerate(chunks)])
        prompt = f"""
Answer the question based on the provided context. Include citations in the format [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        # Extract citations
        citations = self._extract_citations(answer, chunks)

        # Hallucination risk: average similarity (mock as 0.8)
        avg_similarity = 0.8
        risk = "Low" if avg_similarity > 0.7 else "Medium" if avg_similarity > 0.5 else "High"

        return {
            "answer": answer,
            "citations": citations,
            "token_usage": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": tokens
            },
            "hallucination_risk": risk
        }

    def _mock_generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock generation for demo."""
        answer = f"This is a demo answer for: {query}. Based on the provided documents."
        citations = [{"source": chunk["source_doc"], "page": chunk["page"]} for chunk in chunks[:3]]
        return {
            "answer": answer,
            "citations": citations,
            "token_usage": {"prompt": 100, "completion": 50, "total": 150},
            "hallucination_risk": "Low"
        }

    def _extract_citations(self, answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations from answer."""
        citations = []
        for match in re.finditer(r'\[(\d+)\]', answer):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(chunks):
                citations.append({
                    "source": chunks[idx]["source_doc"],
                    "page": chunks[idx]["page"]
                })
        return citations
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
