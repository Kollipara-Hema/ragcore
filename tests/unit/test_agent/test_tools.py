"""Tests for agent/tools — all external calls are mocked."""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestSearchDocs:
    def test_returns_list(self):
        from agent.tools.search_docs import search_docs
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_result)
        with patch("agent.tools.search_docs._get_executor", return_value=mock_executor):
            result = asyncio.run(search_docs.arun({"query": "test query", "top_k": 5}))
        assert isinstance(result, list)

    def test_invalid_strategy_falls_back_to_semantic(self):
        from agent.tools.search_docs import search_docs
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_result)
        with patch("agent.tools.search_docs._get_executor", return_value=mock_executor):
            result = asyncio.run(search_docs.arun({"query": "q", "strategy": "not_real"}))
        assert isinstance(result, list)


class TestSummarizeDoc:
    def test_empty_result_when_no_chunks(self):
        from agent.tools.summarize_doc import summarize_doc
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_result)
        with patch("agent.tools.summarize_doc._get_executor", return_value=mock_executor):
            result = asyncio.run(summarize_doc.arun({"doc_id": "fake-doc-id"}))
        assert result["chunk_count"] == 0
        assert result["combined_text"] == ""


class TestCompareDocs:
    def test_returns_comparison_dict(self):
        from agent.tools.compare_docs import compare_docs
        empty_summary = {"doc_id": "x", "chunk_count": 0, "combined_text": "", "sources": []}
        with patch("agent.tools.compare_docs.summarize_doc", new_callable=AsyncMock, return_value=empty_summary):
            result = asyncio.run(compare_docs.arun({"doc_id_a": "id-a", "doc_id_b": "id-b"}))
        assert "doc_a" in result
        assert "doc_b" in result
        assert "comparison_ready" in result


class TestGetMetadata:
    def test_returns_none_on_error(self):
        from agent.tools.get_metadata import get_metadata
        with patch("agent.tools.get_metadata.get_vector_store", side_effect=Exception("no store")):
            result = asyncio.run(get_metadata.arun({"doc_id": "some-doc-id"}))
        assert result is None
