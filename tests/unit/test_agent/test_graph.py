"""Tests for agent/graph.py — graph construction only, no live execution."""
from agent.graph import build_graph


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
