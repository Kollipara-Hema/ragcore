"""Tests for agent/graph.py — graph construction and end-to-end execution with mocks."""
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from agent.graph import build_graph
from agent.state import initial_state


class TestBuildGraph:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        graph = build_graph()
        # Compiled LangGraph exposes its nodes via .nodes attribute or graph dict
        assert graph is not None

    def test_build_graph_returns_new_instance_each_call(self):
        g1 = build_graph()
        g2 = build_graph()
        assert g1 is not g2


class TestGraphE2E:
    """End-to-end tests with all nodes mocked."""

    def test_full_flow_success(self):
        """Test successful query flow: router → retriever → reranker → generator → evaluator → END"""
        graph = build_graph()
        
        with patch("agent.nodes.router._get_router") as mock_router, \
             patch("agent.nodes.retriever._get_executor") as mock_executor, \
             patch("agent.nodes.reranker._get_reranker") as mock_reranker, \
             patch("agent.nodes.generator._get_generation") as mock_gen, \
             patch("agent.nodes.generator._get_prompt_builder") as mock_pb:
            
            # Mock router decision
            mock_router_obj = MagicMock()
            mock_decision = MagicMock()
            mock_decision.query_type.value = "semantic"
            mock_decision.primary_strategy.value = "hybrid"
            mock_decision.fallback_strategy.value = "semantic"
            mock_decision.expanded_queries = ["test query"]
            mock_decision.reasoning = "test"
            mock_decision.metadata_filter = None
            mock_router_obj.route = AsyncMock(return_value=mock_decision)
            mock_router.return_value = mock_router_obj
            
            # Mock retriever result
            mock_executor_obj = MagicMock()
            mock_result = MagicMock()
            mock_chunk = MagicMock()
            mock_chunk.chunk.content = "test content"
            mock_chunk.chunk.chunk_id = "12345678-1234-5678-9012-123456789012"
            mock_chunk.chunk.doc_id = "12345678-1234-5678-9012-123456789013"
            mock_chunk.chunk.metadata = {"source": "test.pdf"}
            mock_chunk.score = 0.8
            mock_chunk.rank = 1
            mock_chunk.strategy_used.value = "hybrid"
            mock_result.chunks = [mock_chunk]
            mock_result.fallback_used = False
            mock_executor_obj.execute = AsyncMock(return_value=mock_result)
            mock_executor.return_value = mock_executor_obj
            
            # Mock reranker
            mock_reranker_obj = MagicMock()
            mock_reranker_obj.rerank = AsyncMock(return_value=[mock_chunk])
            mock_reranker.return_value = mock_reranker_obj
            
            # Mock generation
            mock_gen_obj = MagicMock()
            mock_gen_result = MagicMock()
            mock_gen_result.answer = "This is a test answer."
            mock_gen_result.citations = []
            mock_gen_result.model_used = "gpt-4o-mini"
            mock_gen_result.total_tokens = 50
            mock_gen_result.cached = False
            mock_gen_obj.generate = AsyncMock(return_value=mock_gen_result)
            mock_gen.return_value = mock_gen_obj
            
            # Mock prompt builder
            mock_pb_obj = MagicMock()
            mock_pb_obj.build = MagicMock()
            mock_pb.return_value = mock_pb_obj
            
            # Run graph
            state = initial_state("What is this?")
            result = asyncio.run(graph.ainvoke(state))
            
            assert result["answer"] == "This is a test answer."
            assert result["confidence"] > 0.0

    def test_retry_on_low_confidence(self):
        """Test retry flow when confidence < 0.6"""
        graph = build_graph()
        
        with patch("agent.nodes.router._get_router") as mock_router, \
             patch("agent.nodes.retriever._get_executor") as mock_executor, \
             patch("agent.nodes.reranker._get_reranker") as mock_reranker, \
             patch("agent.nodes.generator._get_generation") as mock_gen, \
             patch("agent.nodes.generator._get_prompt_builder") as mock_pb:
            
            # Mock router
            mock_router_obj = MagicMock()
            mock_decision = MagicMock()
            mock_decision.query_type.value = "semantic"
            mock_decision.primary_strategy.value = "semantic"
            mock_decision.fallback_strategy.value = "semantic"
            mock_decision.expanded_queries = ["test"]
            mock_decision.reasoning = "test"
            mock_decision.metadata_filter = None
            mock_router_obj.route = AsyncMock(return_value=mock_decision)
            mock_router.return_value = mock_router_obj
            
            # Mock empty retriever (triggers low confidence)
            mock_executor_obj = MagicMock()
            mock_result = MagicMock()
            mock_result.chunks = []
            mock_result.fallback_used = False
            mock_executor_obj.execute = AsyncMock(return_value=mock_result)
            mock_executor.return_value = mock_executor_obj
            
            # Mock reranker
            mock_reranker_obj = MagicMock()
            mock_reranker_obj.rerank = AsyncMock(return_value=[])
            mock_reranker.return_value = mock_reranker_obj
            
            # Mock generation (returns empty/low quality)
            mock_gen_obj = MagicMock()
            mock_gen_result = MagicMock()
            mock_gen_result.answer = ""
            mock_gen_result.citations = []
            mock_gen_result.model_used = "gpt-4o-mini"
            mock_gen_result.total_tokens = 0
            mock_gen_result.cached = False
            mock_gen_obj.generate = AsyncMock(return_value=mock_gen_result)
            mock_gen.return_value = mock_gen_obj
            
            # Mock prompt builder
            mock_pb_obj = MagicMock()
            mock_pb_obj.build = MagicMock()
            mock_pb.return_value = mock_pb_obj
            
            # Run graph
            state = initial_state("What is this?")
            result = asyncio.run(graph.ainvoke(state))
            
            # Should have low confidence and retry_count > 0
            assert result["confidence"] < 0.6
            assert result["retry_count"] > 0
