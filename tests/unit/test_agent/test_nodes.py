"""Tests for individual graph nodes (unit-level, no real services)."""
import asyncio
from unittest.mock import AsyncMock, patch

from agent.state import initial_state
from agent.nodes.evaluator import evaluator_node, should_retry


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
        assert result["answer_score"] > 0.4
        assert result["needs_retry"] is False

    def test_empty_answer_triggers_retry(self):
        state = initial_state("What is X?")
        state.update({"answer": "", "citations": [], "retry_count": 0})
        result = asyncio.run(evaluator_node(state))
        assert result["answer_score"] < 0.4
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
        assert result["answer_score"] < 0.4


class TestShouldRetry:
    def test_routes_to_retriever_when_retry_needed(self):
        state = initial_state("q")
        state["needs_retry"] = True
        assert should_retry(state) == "retriever"

    def test_routes_to_end_when_no_retry(self):
        state = initial_state("q")
        state["needs_retry"] = False
        assert should_retry(state) == "end"
