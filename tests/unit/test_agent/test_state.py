"""Tests for agent/state.py."""
from agent.state import AgentState, initial_state


class TestInitialState:
    def test_required_fields_set(self):
        state = initial_state("What is X?")
        assert state["query"] == "What is X?"
        assert state["top_k"] == 20
        assert state["retry_count"] == 0
        assert state["needs_retry"] is False

    def test_expanded_queries_defaults_to_query(self):
        state = initial_state("hello")
        assert state["expanded_queries"] == ["hello"]

    def test_custom_top_k(self):
        state = initial_state("q", top_k=10)
        assert state["top_k"] == 10

    def test_metadata_filter_none_by_default(self):
        state = initial_state("q")
        assert state["metadata_filter"] is None

    def test_error_none_by_default(self):
        state = initial_state("q")
        assert state["error"] is None
