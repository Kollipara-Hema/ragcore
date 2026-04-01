"""
Unit tests for core RAG components.
Run with: pytest tests/ -v
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from rag_system.utils.models import (
    QueryType, RetrievalStrategy, Chunk, RetrievedChunk,
    DocumentMetadata, Document,
)
from rag_system.retrieval.router.query_router import HeuristicRouter, RoutingDecision
from rag_system.ingestion.chunkers.chunkers import FixedSizeChunker, SemanticChunker
from rag_system.evaluation.evaluator import (
    RetrievalEvaluator, EvalSample, LatencyEvaluator, CostEvaluator,
)
from rag_system.generation.prompts.prompt_builder import PromptBuilder
from rag_system.vectorstore.vector_store import BaseVectorStore


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_document(text: str = "Sample document content for testing purposes.") -> Document:
    meta = DocumentMetadata(source="test.txt", doc_type="txt", title="Test Doc")
    return Document(content=text, metadata=meta)


def make_chunk(text: str = "Test chunk content.", doc_id=None) -> Chunk:
    return Chunk(
        content=text,
        doc_id=doc_id or uuid4(),
        chunk_index=0,
        metadata={"source": "test.txt", "title": "Test Doc", "doc_type": "txt"},
    )


def make_retrieved(chunk: Chunk = None, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk or make_chunk(),
        score=score,
        strategy_used=RetrievalStrategy.HYBRID,
        rank=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic router tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicRouter:
    def setup_method(self):
        self.router = HeuristicRouter()

    def test_lookup_query(self):
        result = self.router.classify("What is the definition of retrieval augmented generation?")
        assert result == QueryType.LOOKUP

    def test_analytical_query(self):
        result = self.router.classify("Compare the advantages and disadvantages of vector search versus BM25.")
        assert result == QueryType.ANALYTICAL

    def test_multi_hop_query(self):
        result = self.router.classify("How does the policy affect the process, which then determines outcomes?")
        assert result == QueryType.MULTI_HOP

    def test_unknown_query_returns_none(self):
        result = self.router.classify("Tell me about the quarterly results.")
        assert result is None  # uncertain — should defer to LLM

    def test_metadata_hints_extraction(self):
        hints = self.router.extract_metadata_hints("Find the report written by John Smith in the pdf.")
        assert "doc_type" in hints
        assert "author" in hints


# ─────────────────────────────────────────────────────────────────────────────
# Chunker tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def setup_method(self):
        self.chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)

    def test_single_chunk_short_doc(self):
        doc = make_document("Short document.")
        chunks = self.chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short document."

    def test_multiple_chunks_long_doc(self):
        long_text = "word " * 200  # 1000 chars
        doc = make_document(long_text)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunks_preserve_doc_id(self):
        doc = make_document("Some content here for testing.")
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.doc_id == doc.doc_id

    def test_chunk_metadata_populated(self):
        doc = make_document("Content for metadata test.")
        chunks = self.chunker.chunk(doc)
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["doc_type"] == "txt"

    def test_overlap_creates_repeated_content(self):
        text = "A B C D E F G H I J K L M N O P Q R S T " * 5
        doc = make_document(text)
        chunks = self.chunker.chunk(doc)
        # Second chunk should overlap with end of first
        if len(chunks) > 1:
            first_end = chunks[0].content[-10:]
            second_start = chunks[1].content[:50]
            # They should share some words due to overlap
            assert any(word in second_start for word in first_end.split() if word)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder(max_context_tokens=2000)

    def test_builds_messages_list(self):
        chunks = [make_retrieved() for _ in range(3)]
        result = self.builder.build("What is X?", chunks, QueryType.FACTUAL)
        assert isinstance(result.messages, list)
        assert len(result.messages) >= 2
        assert result.messages[0]["role"] == "system"

    def test_citations_generated(self):
        chunks = [make_retrieved() for _ in range(3)]
        result = self.builder.build("What is X?", chunks, QueryType.FACTUAL)
        assert len(result.citations) == 3

    def test_token_budget_respected(self):
        # Create many large chunks
        large_chunks = [
            make_retrieved(make_chunk("word " * 300))  # ~300 tokens each
            for _ in range(20)
        ]
        result = self.builder.build("Query?", large_chunks, QueryType.SEMANTIC)
        # Should not use all 20 chunks
        assert result.chunks_used < 20

    def test_different_templates_per_query_type(self):
        chunks = [make_retrieved()]
        analytical_prompt = self.builder.build("Compare X and Y", chunks, QueryType.ANALYTICAL)
        factual_prompt = self.builder.build("What is X?", chunks, QueryType.FACTUAL)
        # Different query types get different user message instructions
        assert analytical_prompt.messages[1]["content"] != factual_prompt.messages[1]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval evaluation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalEvaluator:
    def setup_method(self):
        self.evaluator = RetrievalEvaluator()

    def test_perfect_retrieval(self):
        samples = [
            EvalSample(
                query="Q",
                ground_truth="A",
                relevant_doc_ids=["doc1"],
                retrieved_doc_ids=["doc1", "doc2", "doc3"],
            )
        ]
        metrics = self.evaluator.evaluate(samples)
        assert metrics.hit_rate == 1.0
        assert metrics.mrr == 1.0  # found at rank 1

    def test_no_relevant_docs_retrieved(self):
        samples = [
            EvalSample(
                query="Q",
                ground_truth="A",
                relevant_doc_ids=["doc1"],
                retrieved_doc_ids=["doc2", "doc3", "doc4"],
            )
        ]
        metrics = self.evaluator.evaluate(samples)
        assert metrics.hit_rate == 0.0
        assert metrics.mrr == 0.0

    def test_mrr_with_second_rank_hit(self):
        samples = [
            EvalSample(
                query="Q",
                ground_truth="A",
                relevant_doc_ids=["doc2"],
                retrieved_doc_ids=["doc1", "doc2", "doc3"],
            )
        ]
        metrics = self.evaluator.evaluate(samples)
        assert metrics.mrr == pytest.approx(0.5)  # 1/2

    def test_empty_samples(self):
        metrics = self.evaluator.evaluate([])
        assert metrics.hit_rate == 0.0
        assert metrics.sample_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Latency evaluation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLatencyEvaluator:
    def setup_method(self):
        self.evaluator = LatencyEvaluator()

    def test_percentiles(self):
        latencies = list(range(1, 101))  # 1ms to 100ms
        samples = [
            EvalSample(query="Q", ground_truth="A", relevant_doc_ids=[], latency_ms=float(l))
            for l in latencies
        ]
        metrics = self.evaluator.evaluate(samples)
        assert metrics.p50_ms == pytest.approx(50.0, abs=2)
        assert metrics.p95_ms == pytest.approx(95.0, abs=2)
        assert metrics.p99_ms == pytest.approx(99.0, abs=2)

    def test_empty_samples(self):
        metrics = self.evaluator.evaluate([])
        assert metrics.p50_ms == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Vector store fusion tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRRFFusion:
    """Test the RRF fusion logic in BaseVectorStore."""

    def _make_store(self) -> BaseVectorStore:
        """Create a minimal concrete subclass for testing."""
        class TestStore(BaseVectorStore):
            async def upsert(self, chunks): return 0
            async def vector_search(self, *a, **kw): return []
            async def keyword_search(self, *a, **kw): return []
            async def delete_document(self, *a): return 0

        return TestStore()

    def test_rrf_deduplicates(self):
        store = self._make_store()
        chunk = make_chunk("Shared chunk content")
        rc = make_retrieved(chunk, score=0.9)

        # Same chunk appears in both result sets
        vector_results = [rc]
        keyword_results = [rc]
        fused = store._rrf_fusion(vector_results, keyword_results, alpha=0.7)

        # Should only appear once
        chunk_ids = [str(r.chunk.chunk_id) for r in fused]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_rrf_higher_alpha_favors_vector(self):
        store = self._make_store()
        vector_chunk = make_chunk("Vector result")
        keyword_chunk = make_chunk("Keyword result")

        vector_results = [make_retrieved(vector_chunk, 0.9)]
        keyword_results = [make_retrieved(keyword_chunk, 0.9)]

        # alpha=0.9 strongly favors vector
        fused = store._rrf_fusion(vector_results, keyword_results, alpha=0.9)
        assert str(fused[0].chunk.chunk_id) == str(vector_chunk.chunk_id)

    def test_rrf_alpha_zero_favors_keyword(self):
        store = self._make_store()
        vector_chunk = make_chunk("Vector result")
        keyword_chunk = make_chunk("Keyword result")

        vector_results = [make_retrieved(vector_chunk, 0.9)]
        keyword_results = [make_retrieved(keyword_chunk, 0.9)]

        # alpha=0 strongly favors keyword
        fused = store._rrf_fusion(vector_results, keyword_results, alpha=0.0)
        assert str(fused[0].chunk.chunk_id) == str(keyword_chunk.chunk_id)
