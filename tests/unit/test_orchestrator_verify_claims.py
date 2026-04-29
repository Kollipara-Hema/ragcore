"""
Unit tests for the verify_claims orchestrator override.

Verifies that:
  - verify_claims=False, strategy=basic  → basic generation path
  - verify_claims=True,  strategy=basic  → self_rag generation path
  - verify_claims=True,  strategy=self_rag → self_rag (no-op, already set)
  - QueryRequest.verify_claims defaults to False when omitted
"""
from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_self_rag.py)
# ---------------------------------------------------------------------------

def _make_chunk():
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
    return prompt


def _make_generation_result():
    result = MagicMock()
    result.answer = "basic answer"
    result.citations = []
    result.model_used = "test-model"
    result.total_tokens = 100
    result.cached = False
    return result


def _make_self_rag_result():
    from generation.advanced_generation import SelfRAGResult
    return SelfRAGResult(
        answer="self-rag answer",
        original_answer="original",
        verified_claims=["claim 1"],
        unsupported_claims=[],
        additional_retrievals=0,
        total_tokens=200,
        faithfulness_score=1.0,
    )


def _build_orchestrator(mock_generation):
    """Build a fully-mocked RAGOrchestrator with injected dependencies."""
    from orchestrator import RAGOrchestrator

    mock_router = AsyncMock()
    mock_router.route = AsyncMock(return_value=_make_routing_decision())

    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=_make_retrieval_result())

    mock_reranker = AsyncMock()
    mock_reranker.rerank = AsyncMock(return_value=[_make_chunk()])

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build = MagicMock(return_value=_make_prompt())

    orch = RAGOrchestrator(
        router=mock_router,
        executor=mock_executor,
        generation=mock_generation,
    )
    orch._reranker = mock_reranker
    orch._prompt_builder = mock_prompt_builder
    return orch


def _base_settings(strategy: str):
    """Return a mock settings object with the given generation_strategy."""
    mock_settings = MagicMock()
    mock_settings.generation_strategy = strategy
    mock_settings.self_rag_max_additional_retrievals = 2
    mock_settings.retrieval_top_k = 20
    mock_settings.rerank_top_k = 5
    mock_settings.llm_model = "test-model"
    return mock_settings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVerifyClaimsOverride:

    def test_verify_claims_false_basic_strategy(self):
        """verify_claims=False + strategy=basic → basic generation path, no self_rag."""
        from utils.models import QueryRequest

        mock_generation = AsyncMock()
        mock_generation.generate = AsyncMock(return_value=_make_generation_result())
        mock_generation.generate_followups = AsyncMock(return_value=[])

        orch = _build_orchestrator(mock_generation)

        with patch("orchestrator.settings", _base_settings("basic")):
            with patch(
                "generation.advanced_generation.SelfRAGGenerator.generate",
                new_callable=AsyncMock,
            ) as mock_self_rag:
                response = asyncio.run(
                    orch.query(QueryRequest(query="What is a Roth IRA?", verify_claims=False))
                )

        mock_generation.generate.assert_called_once()
        mock_self_rag.assert_not_called()
        assert response.answer == "basic answer"
        assert response.self_rag_stats is None

    def test_verify_claims_true_basic_strategy(self):
        """verify_claims=True + strategy=basic → self_rag path triggered by override."""
        from utils.models import QueryRequest

        mock_generation = AsyncMock()
        mock_generation.generate_followups = AsyncMock(return_value=[])

        orch = _build_orchestrator(mock_generation)

        with patch("orchestrator.settings", _base_settings("basic")):
            with patch(
                "generation.advanced_generation.SelfRAGGenerator.generate",
                new_callable=AsyncMock,
                return_value=_make_self_rag_result(),
            ) as mock_self_rag:
                response = asyncio.run(
                    orch.query(QueryRequest(query="What is a Roth IRA?", verify_claims=True))
                )

        mock_self_rag.assert_called_once()
        mock_generation.generate.assert_not_called()
        assert response.answer == "self-rag answer"
        assert response.self_rag_stats is not None
        assert response.self_rag_stats["verified_claims"] == ["claim 1"]

    def test_verify_claims_true_self_rag_strategy(self):
        """verify_claims=True + strategy=self_rag → self_rag path; override is a no-op."""
        from utils.models import QueryRequest

        mock_generation = AsyncMock()
        mock_generation.generate_followups = AsyncMock(return_value=[])

        orch = _build_orchestrator(mock_generation)

        with patch("orchestrator.settings", _base_settings("self_rag")):
            with patch(
                "generation.advanced_generation.SelfRAGGenerator.generate",
                new_callable=AsyncMock,
                return_value=_make_self_rag_result(),
            ) as mock_self_rag:
                response = asyncio.run(
                    orch.query(QueryRequest(query="What is a Roth IRA?", verify_claims=True))
                )

        mock_self_rag.assert_called_once()
        mock_generation.generate.assert_not_called()
        assert response.answer == "self-rag answer"
        assert response.self_rag_stats is not None

    def test_verify_claims_default_when_omitted(self):
        """QueryRequest constructed without verify_claims must default to False."""
        from utils.models import QueryRequest

        req = QueryRequest(query="What is a Roth IRA?")
        assert req.verify_claims is False
