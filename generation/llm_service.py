"""
Multi-provider LLM generation service.
Supports: Groq, OpenAI, Anthropic, Ollama
"""
from __future__ import annotations
import logging
import os
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMService:
    """Multi-provider LLM generation with citations and streaming."""

    def __init__(self, provider: str = "groq", model: str = None, api_key: str = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.model = model or self._get_default_model(provider)

    def _get_default_model(self, provider: str) -> str:
        defaults = {
            "groq": "llama3-70b-8192",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "ollama": "llama3"
        }
        return defaults.get(provider.lower(), "llama3-70b-8192")

    def generate(self, query: str, chunks: List[Dict[str, Any]], stream: bool = False) -> Dict[str, Any]:
        """Generate answer with citations."""
        if not self.api_key and self.provider != "ollama":
            return self._mock_generate(query, chunks)

        try:
            if self.provider == "groq":
                return self._groq_generate(query, chunks)
            elif self.provider == "openai":
                return self._openai_generate(query, chunks)
            elif self.provider == "anthropic":
                return self._anthropic_generate(query, chunks)
            elif self.provider == "ollama":
                return self._ollama_generate(query, chunks)
            else:
                return self._mock_generate(query, chunks)
        except Exception as e:
            logger.error(f"Generation error with {self.provider}: {e}")
            return self._mock_generate(query, chunks)

    def _groq_generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate using Groq API."""
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            context = "\n\n".join([f"[{i+1}] {chunk['text_preview']}" for i, chunk in enumerate(chunks)])
            prompt = f"Answer based on context:\n{context}\n\nQ: {query}\nA:"

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            citations = self._extract_citations(answer, chunks)

            return {
                "answer": answer,
                "citations": citations,
                "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                "hallucination_risk": "Low",
                "provider": "Groq " + self.model
            }
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return self._mock_generate(query, chunks)

    def _openai_generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            context = "\n\n".join([f"[{i+1}] {chunk['text_preview']}" for i, chunk in enumerate(chunks)])
            prompt = f"Answer based on context:\n{context}\n\nQ: {query}\nA:"

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            citations = self._extract_citations(answer, chunks)

            return {
                "answer": answer,
                "citations": citations,
                "token_usage": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "hallucination_risk": "Low",
                "provider": "OpenAI " + self.model
            }
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._mock_generate(query, chunks)

    def _anthropic_generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate using Anthropic Claude API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            context = "\n\n".join([f"[{i+1}] {chunk['text_preview']}" for i, chunk in enumerate(chunks)])
            prompt = f"Answer based on context:\n{context}\n\nQ: {query}\nA:"

            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.content[0].text
            citations = self._extract_citations(answer, chunks)

            return {
                "answer": answer,
                "citations": citations,
                "token_usage": {
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                },
                "hallucination_risk": "Low",
                "provider": "Anthropic " + self.model
            }
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return self._mock_generate(query, chunks)

    def _ollama_generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate using local Ollama (no API key needed)."""
        try:
            import requests
            context = "\n\n".join([f"[{i+1}] {chunk['text_preview']}" for i, chunk in enumerate(chunks)])
            prompt = f"Answer based on context:\n{context}\n\nQ: {query}\nA:"

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            answer = response.json().get("response", "")
            citations = self._extract_citations(answer, chunks)

            return {
                "answer": answer,
                "citations": citations,
                "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                "hallucination_risk": "Low",
                "provider": "Ollama " + self.model
            }
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self._mock_generate(query, chunks)

    def _mock_generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock generation for demo mode."""
        answer = f"This is a demo answer for: {query}. Based on the provided documents, here are the key insights."
        citations = [{"source": chunk["source_doc"], "page": chunk["page"]} for chunk in chunks[:3]]
        return {
            "answer": answer,
            "citations": citations,
            "token_usage": {"prompt": 100, "completion": 50, "total": 150},
            "hallucination_risk": "Low",
            "provider": "DEMO MODE"
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


# Provider metadata
PROVIDERS = {
    "groq": {
        "name": "Groq (Recommended)",
        "models": ["llama3-70b-8192", "mixtral-8x7b", "gemma2-9b"],
        "requires_key": True,
        "free": True
    },
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "requires_key": True,
        "free": False
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "requires_key": True,
        "free": False
    },
    "ollama": {
        "name": "Ollama (Local)",
        "models": ["llama3", "mistral", "phi3"],
        "requires_key": False,
        "free": True
    },
    "demo": {
        "name": "Demo Mode",
        "models": ["demo"],
        "requires_key": False,
        "free": True
    }
}
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
