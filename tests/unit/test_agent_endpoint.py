"""
Regression tests for the agent_query endpoint handler.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.main import agent_query
from utils.models import AgentQueryRequest


@pytest.mark.asyncio
async def test_agent_query_invokes_graph_exactly_once():
    """
    Regression: prior implementation called graph.astream_events() followed by
    graph.ainvoke() on the same state, executing the graph twice per request.

    Assert ainvoke is called exactly once and astream_events is never called.
    """
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "answer": "Test answer",
        "confidence": 0.85,
        "citations": [],
        "retry_count": 0,
        "model_used": "test-model",
    })
    mock_graph.astream_events = MagicMock()  # sync mock — any call is wrong

    request = AgentQueryRequest(
        query="What are the main findings?",
        trace_enabled=False,
    )

    with patch("api.main._get_agent_graph", return_value=mock_graph):
        response = await agent_query(request)

    mock_graph.ainvoke.assert_called_once()
    mock_graph.astream_events.assert_not_called()
    assert response.answer == "Test answer"
    assert response.confidence == 0.85
