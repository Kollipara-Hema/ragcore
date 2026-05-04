"""
Integration tests for the /metrics endpoint and orchestrator instrumentation.

These tests verify:
  1. /metrics returns valid Prometheus text exposition format.
  2. Custom metric names appear in the output.
  3. Stage histogram and token counter values appear after a query
     processed by a real RAGOrchestrator (with mocked sub-components).

Note: this module assumes the global tracer is the default NoOpTracer.
When run in isolation or after most other tests, this holds. In some
full-suite orderings, an earlier test may leave LangfuseTracer or another
tracer registered, and test_stage_histogram_and_token_counter_populated_after_query
will then surface a pre-existing tracer serialization bug as a 500 from
the orchestrator. Tracked separately; not a metrics issue.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_metrics_endpoint_status_and_content_type(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_endpoint_contains_custom_metric_names(client):
    body = client.get("/metrics").text
    assert "ragcore_stage_duration_seconds" in body
    assert "ragcore_generation_tokens_total" in body
    assert "ragcore_self_rag_claims_total" in body
    assert "ragcore_process_memory_bytes" in body
    assert "ragcore_vector_store_disk_bytes" in body


def test_metrics_endpoint_contains_http_red_metrics(client):
    # prometheus_fastapi_instrumentator registers this family
    body = client.get("/metrics").text
    assert "http_request_duration_seconds" in body


def _make_mock_orchestrator():
    """Real RAGOrchestrator with all sub-components mocked."""
    from orchestrator import RAGOrchestrator
    from retrieval.router.query_router import RoutingDecision
    from utils.models import (
        QueryType, RetrievalStrategy, GenerationResult, Citation,
        RetrievedChunk, Chunk, DocumentMetadata,
    )

    chunk = Chunk(
        content="test content",
        doc_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        metadata={"source": "test.txt", "title": "Test Doc"},
    )
    rc = RetrievedChunk(chunk=chunk, score=0.9, strategy_used=RetrievalStrategy.HYBRID)

    decision = RoutingDecision(
        query_type=QueryType.SEMANTIC,
        primary_strategy=RetrievalStrategy.HYBRID,
        fallback_strategy=RetrievalStrategy.SEMANTIC,
        expanded_queries=[],
    )

    citation = Citation(
        chunk_id=str(chunk.chunk_id),
        doc_id=str(chunk.doc_id),
        source="test.txt",
        title="Test Doc",
        excerpt="test content",
        score=0.9,
    )
    gen_result = GenerationResult(
        answer="test answer",
        citations=[citation],
        query_type=QueryType.SEMANTIC,
        strategy_used=RetrievalStrategy.HYBRID,
        model_used="test-model",
        total_tokens=50,
        latency_ms=80.0,
        input_tokens=10,
        output_tokens=40,
    )

    mock_router = MagicMock()
    mock_router.route = AsyncMock(return_value=decision)

    retrieval_result = MagicMock()
    retrieval_result.chunks = [rc]
    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value=retrieval_result)

    mock_gen = MagicMock()
    mock_gen.generate = AsyncMock(return_value=gen_result)
    mock_gen.generate_followups = AsyncMock(return_value=[])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[rc])

    orch = RAGOrchestrator(
        router=mock_router,
        executor=mock_executor,
        generation=mock_gen,
    )
    orch._reranker = mock_reranker
    return orch


def test_stage_histogram_and_token_counter_populated_after_query():
    """
    POST /query with a real RAGOrchestrator (mocked sub-components) →
    GET /metrics shows stage duration and token counters incremented.
    """
    import api.main as api_module

    mock_orch = _make_mock_orchestrator()
    original = api_module.orchestrator
    api_module.orchestrator = mock_orch
    try:
        c = TestClient(app)
        query_resp = c.post("/query", json={"query": "What is machine learning?"})
        assert query_resp.status_code == 200

        body = c.get("/metrics").text
        # Stage histogram should have at least one observation
        assert 'ragcore_stage_duration_seconds_count{' in body
        # Token counter should have been incremented
        assert 'ragcore_generation_tokens_total{' in body
    finally:
        api_module.orchestrator = original
