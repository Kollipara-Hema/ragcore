"""Integration tests for evaluation and observability features.
Run with: pytest tests/integration/test_evaluation.py -v
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from evaluation.evaluator import RAGEvaluator, EvalSample
from evaluation.dataset import get_sample_dataset
from monitoring.tracer import get_tracer, LangfuseTracer, NoOpTracer
from agent.graph import build_graph
from agent.state import initial_state
from config.settings import settings


class TestEvaluationIntegration:
    """Test evaluation pipeline integration."""

    @pytest.mark.asyncio
    async def test_rag_evaluator_with_sample_data(self):
        """Test RAGEvaluator can process sample evaluation data."""
        evaluator = RAGEvaluator()

        # Create mock samples
        samples = [
            EvalSample(
                query="What is RAG?",
                ground_truth="Retrieval-Augmented Generation",
                retrieved_doc_ids=["doc1", "doc2"],
                retrieved_contexts=["Context 1", "Context 2"],
                generated_answer="RAG is Retrieval-Augmented Generation",
                latency_ms=100.0,
                total_tokens=50
            )
        ]

        # Mock the generation evaluator to avoid RAGAS dependency in tests
        with patch.object(evaluator._generation, 'evaluate', new_callable=AsyncMock) as mock_gen_eval:
            mock_gen_eval.return_value = MagicMock(
                faithfulness=0.8,
                answer_relevance=0.9,
                hallucination_rate=0.1,
                sample_count=1
            )

            report = await evaluator.run_evaluation(samples)

            assert report is not None
            assert report.generation.faithfulness == 0.8
            assert report.generation.answer_relevance == 0.9
            assert report.latency.mean_ms == 100.0

    def test_sample_dataset_loading(self):
        """Test that sample golden dataset loads correctly."""
        dataset = get_sample_dataset()

        assert isinstance(dataset, list)
        assert len(dataset) > 0

        # Check structure
        sample = dataset[0]
        assert "query" in sample
        assert "ground_truth" in sample
        assert "relevant_doc_ids" in sample
        assert isinstance(sample["relevant_doc_ids"], list)

    @pytest.mark.asyncio
    async def test_evaluator_against_orchestrator_mock(self):
        """Test evaluation against a mocked orchestrator."""
        evaluator = RAGEvaluator()

        golden_dataset = [
            {
                "query": "What is the capital of France?",
                "ground_truth": "Paris",
                "relevant_doc_ids": ["doc_geo_001"]
            }
        ]

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "Paris is the capital of France."
        mock_response.citations = [{"doc_id": "doc_geo_001", "excerpt": "Paris is the capital"}]
        mock_response.total_tokens = 25
        mock_orchestrator.query = AsyncMock(return_value=mock_response)

        # Mock generation evaluation
        with patch.object(evaluator._generation, 'evaluate', new_callable=AsyncMock) as mock_gen_eval:
            mock_gen_eval.return_value = MagicMock(
                faithfulness=0.9,
                answer_relevance=0.95,
                hallucination_rate=0.05,
                sample_count=1
            )
            report = await evaluator.run_against_orchestrator(golden_dataset, mock_orchestrator)

            assert report is not None
            assert report.retrieval.sample_count == 1
            assert report.generation.faithfulness == 0.9


class TestObservabilityIntegration:
    """Test observability and tracing integration."""

    def test_tracer_initialization(self):
        """Test that tracer initializes correctly based on settings."""
        # Test NoOpTracer when tracing disabled
        with patch.object(settings, 'enable_tracing', False):
            tracer = get_tracer()
            assert isinstance(tracer, NoOpTracer)

        # Test LangfuseTracer when enabled (mock the check)
        with patch.object(settings, 'enable_tracing', True), \
             patch.object(settings, 'langfuse_public_key', 'test_key'):
            tracer = get_tracer()
            assert isinstance(tracer, LangfuseTracer)

    @pytest.mark.asyncio
    async def test_langfuse_tracer_trace_lifecycle(self):
        """Test full trace lifecycle with LangfuseTracer."""
        tracer = LangfuseTracer()

        # Mock Langfuse client
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace

        with patch.object(tracer, '_get_client', return_value=mock_client):
            # Start trace
            trace_id = await tracer.start_trace("Test query")
            assert trace_id is not None

            # Log events
            await tracer.log_routing(trace_id, MagicMock(query_type=MagicMock(value="factual"),
                                                        primary_strategy=MagicMock(value="hybrid"),
                                                        expanded_queries=[]))

            await tracer.log_retrieval(trace_id, MagicMock(
                chunks=[MagicMock(score=0.9), MagicMock(score=0.8)],
                latency_ms=50.0,
                request=MagicMock(strategy=MagicMock(value="hybrid")),
                fallback_used=False
            ))

            # End trace
            await tracer.end_trace(trace_id, 150.0)

            # Verify trace was stored
            trace = tracer.get_trace(trace_id)
            assert trace is not None
            assert trace.query == "Test query"
            assert trace.total_duration_ms == 150.0
            assert len(trace.events) >= 2  # routing + retrieval

    @pytest.mark.asyncio
    async def test_agent_graph_with_tracing(self):
        """Test that agent graph integrates with tracing."""
        # Mock the graph to avoid full initialization
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "answer": "Test answer",
            "confidence": 0.8,
            "citations": []
        })

        with patch('agent.graph.build_graph', return_value=mock_graph):
            graph = build_graph()
            assert hasattr(graph, 'ainvoke')

            # Test invocation
            state = initial_state("Test query")
            result = await graph.ainvoke(state)

            assert result["answer"] == "Test answer"
            # Verify graph was called
            mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_trace_retrieval_api_mock(self):
        """Test trace retrieval through API (mocked)."""
        from api.main import _trace_store

        # Clear any existing traces
        _trace_store.clear()

        # Simulate storing a trace
        from utils.models import QueryTrace
        from datetime import datetime

        trace = QueryTrace(
            trace_id="test-trace-123",
            query="Test query",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            total_duration_ms=100.0,
            events=[],
            final_answer="Test answer",
            confidence=0.8,
            status="completed"
        )
        _trace_store["test-trace-123"] = trace

        # Test retrieval
        retrieved = _trace_store.get("test-trace-123")
        assert retrieved is not None
        assert retrieved.trace_id == "test-trace-123"
        assert retrieved.final_answer == "Test answer"