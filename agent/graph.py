"""
Agent graph — wires all nodes into a LangGraph StateGraph.

Flow:
    router → retriever → reranker → generator → evaluator
                ↑                                    |
                └──────── retry (if score low) ──────┘

Usage:
    from agent.graph import build_graph
    graph = build_graph()
    result = await graph.ainvoke({"query": "What is X?"})
"""
from __future__ import annotations
import time
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.router import router_node
from agent.nodes.retriever import retriever_node
from agent.nodes.reranker import reranker_node
from agent.nodes.generator import generator_node
from agent.nodes.evaluator import evaluator_node, should_retry
from monitoring.tracer import get_tracer


class TracedGraph:
    """
    Wrapper around compiled StateGraph that adds tracing to node executions.
    """

    def __init__(self, graph: StateGraph):
        self.graph = graph
        self._tracer = get_tracer()

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the graph with tracing around each node execution.
        """
        query = state.get("query", "")
        trace_id = await self._tracer.start_trace(query)

        try:
            # Start timing the entire graph execution
            start_time = time.monotonic()

            # We need to intercept node calls, but LangGraph doesn't expose hooks easily.
            # For now, we'll add tracing at the graph level and rely on individual nodes
            # to log their own details if needed.
            result = await self.graph.ainvoke(state)

            # Calculate total latency
            total_latency = (time.monotonic() - start_time) * 1000
            await self._tracer.end_trace(trace_id, total_latency)

            return result

        except Exception as e:
            await self._tracer.log_error(trace_id, str(e))
            raise


def build_graph() -> TracedGraph:
    """
    Construct and compile the agent StateGraph.

    Returns a compiled graph that can be invoked with:
        result = await graph.ainvoke(initial_state)
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)

    # Linear edges
    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", "evaluator")

    # Conditional edge: retry retrieval or finish
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "retriever": "retriever",
            "end": END,
        },
    )

    return TracedGraph(graph.compile())
