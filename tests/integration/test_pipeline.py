"""
Integration tests — test the full RAG pipeline with a mock vector store.
These tests verify the orchestrator wires all layers correctly.

Run with: pytest tests/integration/ -v
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from rag_system.utils.models import (
    QueryType, RetrievalStrategy, Chunk, RetrievedChunk,
    RetrievalResult, RetrievalRequest, DocumentMetadata,
)
from rag_system.orchestrator import RAGOrchestrator
from rag_system.retrieval.strategies.retrieval_executor import RetrievalExecutor
from rag_system.retrieval.router.query_router import QueryRouter, RoutingDecision
from rag_system.generation.llm_service import GenerationService, GenerationResult
from rag_system.generation.prompts.prompt_builder import PromptBuilder, ConstructedPrompt
from rag_system.utils.models import QueryRequest, Citation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_chunk(text: str = "The quarterly results showed 15% growth year-over-year.") -> Chunk:
    return Chunk(
        content=text,
        doc_id=uuid4(),
        chunk_index=0,
        metadata={"source": "report.pdf", "title": "Q3 Report", "doc_type": "pdf"},
    )


def make_retrieved(text: str = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(text or "Relevant retrieved content for the query."),
        score=0.88,
        strategy_used=RetrievalStrategy.HYBRID,
        rank=0,
    )


def make_routing_decision(
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    query_type: QueryType = QueryType.FACTUAL,
) -> RoutingDecision:
    return RoutingDecision(
        query_type=query_type,
        primary_strategy=strategy,
        fallback_strategy=RetrievalStrategy.SEMANTIC,
        expanded_queries=["What were the Q3 revenue results?"],
    )


def make_retrieval_result(chunks=None) -> RetrievalResult:
    return RetrievalResult(
        request=RetrievalRequest(
            query="test query",
            query_type=QueryType.FACTUAL,
            strategy=RetrievalStrategy.HYBRID,
        ),
        chunks=chunks or [make_retrieved()],
        latency_ms=120.0,
    )


def make_generation_result(answer: str = "Q3 revenue grew 15% year-over-year.") -> GenerationResult:
    return GenerationResult(
        answer=answer,
        citations=[
            Citation(
                chunk_id=str(uuid4()),
                doc_id=str(uuid4()),
                source="report.pdf",
                title="Q3 Report",
                excerpt="The quarterly results showed 15% growth.",
                score=0.88,
            )
        ],
        query_type=QueryType.FACTUAL,
        strategy_used=RetrievalStrategy.HYBRID,
        model_used="gpt-4o",
        total_tokens=512,
        latency_ms=800.0,
        cached=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRAGOrchestrator:

    async def _build_orchestrator_with_mocks(
        self,
        routing_decision=None,
        retrieval_result=None,
        generation_result=None,
    ):
        """Build orchestrator with all external dependencies mocked."""
        mock_router = AsyncMock(spec=QueryRouter)
        mock_router.route.return_value = routing_decision or make_routing_decision()

        mock_executor = AsyncMock(spec=RetrievalExecutor)
        mock_executor.execute.return_value = retrieval_result or make_retrieval_result()

        mock_generation = AsyncMock(spec=GenerationService)
        mock_generation.generate.return_value = generation_result or make_generation_result()

        orch = RAGOrchestrator(
            router=mock_router,
            executor=mock_executor,
            generation=mock_generation,
        )
        # Mock reranker to pass through
        orch._reranker = AsyncMock()
        orch._reranker.rerank.side_effect = lambda query, chunks, top_k: chunks[:top_k]
        # Mock tracer
        orch._tracer = MagicMock()
        orch._tracer.start_trace = AsyncMock(return_value="test-trace-id")
        orch._tracer.log_routing = AsyncMock()
        orch._tracer.log_retrieval = AsyncMock()
        orch._tracer.log_reranking = AsyncMock()
        orch._tracer.log_generation = AsyncMock()
        orch._tracer.log_error = AsyncMock()
        orch._tracer.end_trace = AsyncMock()

        return orch, mock_router, mock_executor, mock_generation

    async def test_successful_query_pipeline(self):
        orch, router, executor, generation = await self._build_orchestrator_with_mocks()
        request = QueryRequest(query="What were the Q3 revenue results?")
        response = await orch.query(request)

        assert response.answer == "Q3 revenue grew 15% year-over-year."
        assert len(response.citations) == 1
        assert response.citations[0]["source"] == "report.pdf"
        assert response.query_type == "factual"
        assert response.strategy_used == "hybrid"
        assert response.total_tokens == 512
        assert not response.cached

    async def test_router_called_with_query(self):
        orch, router, executor, generation = await self._build_orchestrator_with_mocks()
        request = QueryRequest(query="Test query for routing?")
        await orch.query(request)
        router.route.assert_called_once()
        call_kwargs = router.route.call_args.kwargs
        assert call_kwargs["query"] == "Test query for routing?"

    async def test_empty_retrieval_returns_graceful_response(self):
        empty_result = make_retrieval_result(chunks=[])
        orch, _, _, _ = await self._build_orchestrator_with_mocks(retrieval_result=empty_result)
        request = QueryRequest(query="Query with no results.")
        response = await orch.query(request)

        assert "could not find" in response.answer.lower()
        assert response.citations == []

    async def test_strategy_override_passed_to_router(self):
        orch, router, _, _ = await self._build_orchestrator_with_mocks()
        request = QueryRequest(
            query="Override strategy test.",
            strategy_override=RetrievalStrategy.KEYWORD,
        )
        await orch.query(request)
        call_kwargs = router.route.call_args.kwargs
        assert call_kwargs["strategy_override"] == RetrievalStrategy.KEYWORD

    async def test_metadata_filter_passed_to_router(self):
        orch, router, _, _ = await self._build_orchestrator_with_mocks()
        request = QueryRequest(
            query="Filtered query.",
            metadata_filter={"doc_type": "pdf"},
        )
        await orch.query(request)
        call_kwargs = router.route.call_args.kwargs
        assert call_kwargs["metadata_filter"] == {"doc_type": "pdf"}

    async def test_cached_response_flagged(self):
        cached_result = make_generation_result()
        cached_result.cached = True
        orch, _, _, _ = await self._build_orchestrator_with_mocks(generation_result=cached_result)
        request = QueryRequest(query="Cached query?")
        response = await orch.query(request)
        assert response.cached is True

    async def test_retrieval_failure_raises_exception(self):
        orch, _, executor, _ = await self._build_orchestrator_with_mocks()
        executor.execute.side_effect = RuntimeError("Vector store connection failed")
        request = QueryRequest(query="This should fail.")
        with pytest.raises(RuntimeError, match="Vector store connection failed"):
            await orch.query(request)

    async def test_latency_measured_and_returned(self):
        orch, _, _, _ = await self._build_orchestrator_with_mocks()
        request = QueryRequest(query="Latency test query.")
        response = await orch.query(request)
        assert response.latency_ms > 0


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval executor tests (with mocked vector store)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRetrievalExecutor:

    def _make_executor(self, mock_chunks=None):
        mock_store = AsyncMock()
        mock_store.vector_search.return_value = mock_chunks or [make_retrieved()]
        mock_store.keyword_search.return_value = mock_chunks or [make_retrieved()]
        mock_store.hybrid_search.return_value = mock_chunks or [make_retrieved()]

        mock_embedder = AsyncMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        return RetrievalExecutor(vector_store=mock_store, embedder=mock_embedder)

    async def test_semantic_strategy_calls_vector_search(self):
        executor = self._make_executor()
        decision = make_routing_decision(RetrievalStrategy.SEMANTIC)
        result = await executor.execute(decision, top_k=5)
        executor._store.vector_search.assert_called_once()
        assert len(result.chunks) > 0

    async def test_keyword_strategy_calls_keyword_search(self):
        executor = self._make_executor()
        decision = make_routing_decision(RetrievalStrategy.KEYWORD)
        result = await executor.execute(decision, top_k=5)
        executor._store.keyword_search.assert_called_once()

    async def test_hybrid_strategy_calls_hybrid_search(self):
        executor = self._make_executor()
        decision = make_routing_decision(RetrievalStrategy.HYBRID)
        result = await executor.execute(decision, top_k=5)
        executor._store.hybrid_search.assert_called_once()

    async def test_fallback_triggered_on_primary_failure(self):
        executor = self._make_executor()
        executor._store.hybrid_search.side_effect = RuntimeError("Weaviate down")
        executor._store.vector_search.return_value = [make_retrieved()]

        decision = RoutingDecision(
            query_type=QueryType.FACTUAL,
            primary_strategy=RetrievalStrategy.HYBRID,
            fallback_strategy=RetrievalStrategy.SEMANTIC,
            expanded_queries=["Test fallback."],
        )
        result = await executor.execute(decision, top_k=5)
        # Fallback to semantic should have been called
        assert result.fallback_used is True

    async def test_multi_query_runs_parallel_retrieval(self):
        executor = self._make_executor()
        decision = RoutingDecision(
            query_type=QueryType.MULTI_HOP,
            primary_strategy=RetrievalStrategy.MULTI_QUERY,
            fallback_strategy=RetrievalStrategy.HYBRID,
            expanded_queries=["Query 1.", "Query 2.", "Query 3."],
        )
        result = await executor.execute(decision, top_k=10)
        # hybrid_search should be called once per expanded query
        assert executor._store.hybrid_search.call_count == 3
