"""
Unit tests for Self-RAG orchestrator integration.

These tests verify that:
  - generation_strategy == "self_rag"  → SelfRAGGenerator.generate() is called
  - generation_strategy == "basic"     → SelfRAGGenerator is never instantiated

External services (LLM, vector store, reranker) are fully mocked.
"""
from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — build the minimal fakes the orchestrator needs
# ---------------------------------------------------------------------------

def _make_chunk():
    """Minimal RetrievedChunk-like object."""
    chunk = MagicMock()
    chunk.chunk.chunk_id = "chunk-001"
    chunk.chunk.content = "test content"
    chunk.score = 0.9
    return chunk


def _make_retrieval_result(chunks=None):
    result = MagicMock()
    result.chunks = chunks if chunks is not None else [_make_chunk()]
    return result


def _make_routing_decision():
    from utils.models import QueryType, RetrievalStrategy
    decision = MagicMock()
    decision.query_type = QueryType.FACTUAL
    decision.primary_strategy = RetrievalStrategy.HYBRID
    decision.metadata_filter = None
    return decision


def _make_prompt(citations=None):
    prompt = MagicMock()
    prompt.citations = citations or []
    prompt.messages = [{"role": "user", "content": "test"}]
    # chunk_id needed for cache key calculation in GenerationService
    for c in prompt.citations:
        c.chunk_id = "cite-001"
    return prompt


def _make_generation_result():
    from utils.models import QueryType, RetrievalStrategy
    result = MagicMock()
    result.answer = "basic answer"
    result.citations = []
    result.model_used = "gpt-4o-mini"
    result.total_tokens = 100
    result.cached = False
    result.query_type = QueryType.FACTUAL
    result.strategy_used = RetrievalStrategy.HYBRID
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelfRAGOrchestration:

    def test_self_rag_path_is_taken_when_strategy_is_self_rag(self):
        """When generation_strategy == 'self_rag', SelfRAGGenerator.generate() must be called."""
        from orchestrator import RAGOrchestrator
        from utils.models import QueryRequest

        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=_make_routing_decision())

        mock_executor = AsyncMock()
        mock_executor.execute = AsyncMock(return_value=_make_retrieval_result())

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(return_value=[_make_chunk()])

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build = MagicMock(return_value=_make_prompt())

        mock_generation = AsyncMock()

        orch = RAGOrchestrator(
            router=mock_router,
            executor=mock_executor,
            generation=mock_generation,
        )
        orch._reranker = mock_reranker
        orch._prompt_builder = mock_prompt_builder

        from generation.advanced_generation import SelfRAGResult
        fake_self_rag_result = SelfRAGResult(
            answer="self-rag answer",
            original_answer="original",
            verified_claims=["claim 1"],
            unsupported_claims=[],
            additional_retrievals=0,
            total_tokens=200,
            faithfulness_score=1.0,
        )

        with patch("orchestrator.settings") as mock_settings:
            mock_settings.generation_strategy = "self_rag"
            mock_settings.self_rag_max_additional_retrievals = 2
            mock_settings.retrieval_top_k = 20
            mock_settings.rerank_top_k = 5
            mock_settings.llm_model = "gpt-4o-mini"

            with patch(
                "generation.advanced_generation.SelfRAGGenerator.generate",
                new_callable=AsyncMock,
                return_value=fake_self_rag_result,
            ) as mock_self_rag_generate:
                response = asyncio.run(orch.query(QueryRequest(query="What is X?")))

        mock_self_rag_generate.assert_called_once()
        assert response.answer == "self-rag answer"
        # Basic GenerationService.generate() must NOT have been called
        mock_generation.generate.assert_not_called()

    def test_basic_path_taken_when_strategy_is_basic(self):
        """When generation_strategy == 'basic', SelfRAGGenerator must never be instantiated."""
        from orchestrator import RAGOrchestrator
        from utils.models import QueryRequest

        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=_make_routing_decision())

        mock_executor = AsyncMock()
        mock_executor.execute = AsyncMock(return_value=_make_retrieval_result())

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(return_value=[_make_chunk()])

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build = MagicMock(return_value=_make_prompt())

        mock_generation = AsyncMock()
        mock_generation.generate = AsyncMock(return_value=_make_generation_result())

        orch = RAGOrchestrator(
            router=mock_router,
            executor=mock_executor,
            generation=mock_generation,
        )
        orch._reranker = mock_reranker
        orch._prompt_builder = mock_prompt_builder

        with patch("orchestrator.settings") as mock_settings:
            mock_settings.generation_strategy = "basic"
            mock_settings.retrieval_top_k = 20
            mock_settings.rerank_top_k = 5
            mock_settings.llm_model = "gpt-4o-mini"

            with patch(
                "generation.advanced_generation.SelfRAGGenerator.generate",
                new_callable=AsyncMock,
            ) as mock_self_rag_generate:
                response = asyncio.run(orch.query(QueryRequest(query="What is X?")))

        mock_generation.generate.assert_called_once()
        mock_self_rag_generate.assert_not_called()
        assert response.answer == "basic answer"
