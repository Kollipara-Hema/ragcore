"""
Agent package — LangGraph-based agentic RAG workflow.

Entry point:
    from agent.graph import build_graph
    graph = build_graph()
    result = await graph.ainvoke({"query": "What is X?"})
"""

__all__ = ["build_graph"]
