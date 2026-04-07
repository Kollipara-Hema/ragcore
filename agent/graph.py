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

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.router import router_node
from agent.nodes.retriever import retriever_node
from agent.nodes.reranker import reranker_node
from agent.nodes.generator import generator_node
from agent.nodes.evaluator import evaluator_node, should_retry


def build_graph() -> StateGraph:
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

    return graph.compile()
