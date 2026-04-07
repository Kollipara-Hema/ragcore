"""Tests for individual graph nodes (unit-level, no real services)."""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from agent.state import initial_state
from agent.nodes.evaluator import evaluator_node, should_retry
from agent.nodes.router import router_node
from agent.nodes.retriever import retriever_node
from agent.nodes.reranker import reranker_node
from agent.nodes.generator import generator_node


class TestEvaluatorNode:
    """Evaluator has no external dependencies — test it directly."""

    def test_good_answer_accepted(self):
        state = initial_state("What is X?")
        state.update({
            "answer": "X is a well-known concept described in many documents. " * 4,
            "citations": [{"source": "doc.pdf", "score": 0.9}],
            "query_type": "factual",
            "retry_count": 0,
        })
        result = asyncio.run(evaluator_node(state))
        assert result["confidence"] > 0.6
        assert result["needs_retry"] is False

    def test_empty_answer_triggers_retry(self):
        state = initial_state("What is X?")
        state.update({"answer": "", "citations": [], "retry_count": 0})
        result = asyncio.run(evaluator_node(state))
        assert result["confidence"] < 0.6
        assert result["needs_retry"] is True

    def test_max_retries_not_exceeded(self):
        state = initial_state("What is X?")
        state.update({"answer": "", "citations": [], "retry_count": 2})
        result = asyncio.run(evaluator_node(state))
        assert result["needs_retry"] is False

    def test_not_found_answer_low_score(self):
        state = initial_state("q")
        state.update({
            "answer": "I could not find relevant information.",
            "citations": [],
            "retry_count": 0,
        })
        result = asyncio.run(evaluator_node(state))
        assert result["confidence"] < 0.6


class TestRouterNode:
    """Test router node with mocked QueryRouter."""

    @patch("agent.nodes.router._get_router")
    def test_router_sets_strategy(self, mock_get_router):
        mock_router = AsyncMock()
        mock_decision = AsyncMock()
        mock_decision.query_type.value = "semantic"
        mock_decision.primary_strategy.value = "hybrid"
        mock_decision.fallback_strategy.value = "semantic"
        mock_decision.expanded_queries = ["What is X?"]
        mock_decision.reasoning = "test"
        mock_decision.metadata_filter = None
        mock_router.route.return_value = mock_decision
        mock_get_router.return_value = mock_router

        state = initial_state("What is X?")
        result = asyncio.run(router_node(state))

        assert result["metadata"]["strategy"] == "hybrid"
        assert result["query_type"] == "semantic"


class TestRetrieverNode:
    """Test retriever node with mocked RetrievalExecutor."""

    @patch("agent.nodes.retriever._get_executor")
    def test_retriever_calls_executor(self, mock_get_executor):
        mock_executor = AsyncMock()
        mock_result = AsyncMock()
        mock_chunk = AsyncMock()
        mock_chunk.chunk.content = "test content"
        mock_chunk.chunk.chunk_id = "123"
        mock_chunk.chunk.doc_id = "456"
        mock_chunk.chunk.metadata = {}
        mock_chunk.score = 0.8
        mock_chunk.rank = 1
        mock_chunk.strategy_used.value = "hybrid"
        mock_result.chunks = [mock_chunk]
        mock_result.fallback_used = False
        mock_executor.execute.return_value = mock_result
        mock_get_executor.return_value = mock_executor

        state = initial_state("What is X?")
        state.update({"primary_strategy": "hybrid"})
        result = asyncio.run(retriever_node(state))

        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieved_chunks"][0]["content"] == "test content"


class TestRerankerNode:
    """Test reranker node with mocked reranker."""

    @patch("agent.nodes.reranker._get_reranker")
    def test_reranker_reranks_chunks(self, mock_get_reranker):
        mock_reranker = AsyncMock()
        mock_reranked = [AsyncMock()]
        mock_reranked[0].chunk.content = "reranked content"
        mock_reranked[0].chunk.chunk_id = "12345678-1234-5678-9012-123456789012"
        mock_reranked[0].chunk.doc_id = "12345678-1234-5678-9012-123456789013"
        mock_reranked[0].chunk.metadata = {}
        mock_reranked[0].score = 0.9
        mock_reranked[0].rank = 1
        mock_reranker.rerank.return_value = mock_reranked
        mock_get_reranker.return_value = mock_reranker

        state = initial_state("What is X?")
        state.update({"retrieved_chunks": [{"content": "content", "chunk_id": "12345678-1234-5678-9012-123456789012", "doc_id": "12345678-1234-5678-9012-123456789013", "score": 0.8, "rank": 1, "metadata": {}}]})
        result = asyncio.run(reranker_node(state))

        assert len(result["reranked_chunks"]) == 1
        assert result["reranked_chunks"][0]["content"] == "reranked content"


class TestGeneratorNode:
    """Test generator node with mocked generation service."""

    @patch("agent.nodes.generator._get_generation")
    @patch("agent.nodes.generator._get_prompt_builder")
    def test_generator_calls_llm(self, mock_get_prompt_builder, mock_get_generation):
        mock_prompt_builder = AsyncMock()
        mock_prompt = AsyncMock()
        mock_prompt_builder.build.return_value = mock_prompt
        mock_get_prompt_builder.return_value = mock_prompt_builder

        mock_generation = AsyncMock()
        mock_result = AsyncMock()
        mock_result.answer = "Generated answer"
        mock_result.citations = []
        mock_result.model_used = "gpt-4o-mini"
        mock_result.total_tokens = 100
        mock_result.generation_latency_ms = 500
        mock_result.cached = False
        mock_generation.generate.return_value = mock_result
        mock_get_generation.return_value = mock_generation

        state = initial_state("What is X?")
        state.update({"reranked_chunks": [{"content": "content", "chunk_id": "12345678-1234-5678-9012-123456789012", "doc_id": "12345678-1234-5678-9012-123456789013", "score": 0.8, "rank": 1, "metadata": {}}], "query_type": "semantic", "primary_strategy": "hybrid"})
        result = asyncio.run(generator_node(state))

        assert result["answer"] == "Generated answer"
        assert result["model_used"] == "gpt-4o-mini"

    @patch("agent.nodes.generator._get_generation")
    @patch("agent.nodes.generator._get_prompt_builder")
    @patch("agent.nodes.generator._get_short_term_memory")
    def test_generator_uses_short_term_memory(self, mock_get_stm, mock_get_prompt_builder, mock_get_generation):
        # Mock short-term memory
        mock_stm = AsyncMock()
        mock_stm.get_context = MagicMock(return_value="Previous: Q1: What is AI? A1: AI is...")
        mock_stm.add = MagicMock()
        mock_get_stm.return_value = mock_stm

        # Mock prompt builder
        mock_pb = AsyncMock()
        mock_prompt = AsyncMock()
        mock_pb.build = MagicMock(return_value=mock_prompt)
        mock_get_prompt_builder.return_value = mock_pb

        # Mock generation
        mock_gen = AsyncMock()
        mock_result = AsyncMock()
        mock_result.answer = "Answer with context"
        mock_result.citations = []
        mock_result.model_used = "gpt-4o-mini"
        mock_result.total_tokens = 100
        mock_result.cached = False
        mock_gen.generate = AsyncMock(return_value=mock_result)
        mock_get_generation.return_value = mock_gen

        state = initial_state("What is Y?")
        state.update({"reranked_chunks": [{"content": "content", "chunk_id": "12345678-1234-5678-9012-123456789012", "doc_id": "12345678-1234-5678-9012-123456789013", "score": 0.8, "rank": 1, "metadata": {}}], "query_type": "semantic", "primary_strategy": "hybrid"})
        result = asyncio.run(generator_node(state))

        # Verify memory context was retrieved
        mock_stm.get_context.assert_called()
        # Verify memory was updated with new turn
        mock_stm.add.assert_called()
        assert result["answer"] == "Answer with context"


class TestShouldRetry:
    def test_routes_to_retriever_when_retry_needed(self):
        state = initial_state("q")
        state["needs_retry"] = True
        assert should_retry(state) == "retriever"

    def test_routes_to_end_when_no_retry(self):
        state = initial_state("q")
        state["needs_retry"] = False
        assert should_retry(state) == "end"
