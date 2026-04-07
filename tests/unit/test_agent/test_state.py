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

    def test_messages_empty_by_default(self):
        state = initial_state("q")
        assert state["messages"] == []

    def test_tool_calls_empty_by_default(self):
        state = initial_state("q")
        assert state["tool_calls"] == []

    def test_confidence_zero_by_default(self):
        state = initial_state("q")
        assert state["confidence"] == 0.0

    def test_final_answer_empty_by_default(self):
        state = initial_state("q")
        assert state["final_answer"] == ""

    def test_metadata_empty_dict_by_default(self):
        state = initial_state("q")
        assert state["metadata"] == {}


class TestAgentStateFields:
    def test_messages_accepts_list_of_dicts(self):
        state = initial_state("q")
        state["messages"] = [{"role": "user", "content": "hello"}]
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"

    def test_tool_calls_accepts_list_of_dicts(self):
        state = initial_state("q")
        state["tool_calls"] = [{"tool": "search_docs", "args": {"query": "x"}}]
        assert state["tool_calls"][0]["tool"] == "search_docs"

    def test_confidence_accepts_float(self):
        state = initial_state("q")
        state["confidence"] = 0.85
        assert state["confidence"] == 0.85

    def test_final_answer_accepts_string(self):
        state = initial_state("q")
        state["final_answer"] = "The answer is 42."
        assert state["final_answer"] == "The answer is 42."

    def test_metadata_accepts_arbitrary_keys(self):
        state = initial_state("q")
        state["metadata"] = {"session_id": "abc", "user_id": 7}
        assert state["metadata"]["session_id"] == "abc"

    def test_retrieved_chunks_empty_by_default(self):
        state = initial_state("q")
        assert state["retrieved_chunks"] == []
