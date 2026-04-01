"""
Unit tests for core RAG components.
Run with: pytest tests/unit/ -v
"""
import pytest
from uuid import uuid4

from utils.models import (
    QueryType, RetrievalStrategy, DocumentMetadata, Document,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentModel:
    """Test Document model creation and basic functionality."""

    def test_document_creation(self):
        """Test creating a document with metadata."""
        meta = DocumentMetadata(source="test.txt", doc_type="txt")
        doc = Document(content="Test content", metadata=meta)
        assert doc.content == "Test content"
        assert doc.metadata.source == "test.txt"
        assert doc.metadata.doc_type == "txt"

    def test_document_has_unique_id(self):
        """Test that each document gets a unique ID."""
        meta1 = DocumentMetadata(source="test1.txt", doc_type="txt")
        doc1 = Document(content="Content 1", metadata=meta1)
        
        meta2 = DocumentMetadata(source="test2.txt", doc_type="txt")
        doc2 = Document(content="Content 2", metadata=meta2)
        
        assert doc1.doc_id != doc2.doc_id
        assert doc1.doc_id is not None

    def test_document_has_default_status(self):
        """Test that documents have a status."""
        meta = DocumentMetadata(source="test.txt", doc_type="txt")
        doc = Document(content="Test", metadata=meta)
        assert doc.status is not None


# ─────────────────────────────────────────────────────────────────────────────
# Query type tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryTypes:
    """Test query type and retrieval strategy enums."""

    def test_query_types_defined(self):
        """Test that all major query types are defined."""
        assert QueryType.FACTUAL is not None
        assert QueryType.SEMANTIC is not None
        assert QueryType.MULTI_HOP is not None
        assert QueryType.ANALYTICAL is not None
        assert QueryType.LOOKUP is not None

    def test_retrieval_strategies_defined(self):
        """Test that all retrieval strategies are defined."""
        assert RetrievalStrategy.SEMANTIC is not None
        assert RetrievalStrategy.KEYWORD is not None
        assert RetrievalStrategy.HYBRID is not None
        assert RetrievalStrategy.METADATA_FILTER is not None


# ─────────────────────────────────────────────────────────────────────────────
# Module import tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImports:
    """Test that core modules can be imported without errors."""

    def test_chunker_imports(self):
        """Test chunking module can be imported."""
        from ingestion.chunkers.chunkers import get_chunker
        assert get_chunker is not None
        chunker = get_chunker("fixed")
        assert chunker is not None

    def test_embedder_imports(self):
        """Test embedding module can be imported."""
        from embeddings.embedder import get_embedder
        assert get_embedder is not None
        embedder = get_embedder("minilm")
        assert embedder is not None

    def test_llm_service_imports(self):
        """Test LLM service can be imported."""
        from generation.llm_service import LLMService, PROVIDERS
        assert LLMService is not None
        assert PROVIDERS is not None
        assert "groq" in PROVIDERS
        assert "openai" in PROVIDERS

    def test_retrieval_imports(self):
        """Test retrieval module can be imported."""
        from retrieval.strategies.retrieval_executor import RetrievalExecutor
        assert RetrievalExecutor is not None

    def test_reranker_imports(self):
        """Test reranking module can be imported."""
        from reranking.reranker import Reranker
        assert Reranker is not None

    def test_evaluation_imports(self):
        """Test evaluation module can be imported."""
        from evaluation.evaluator import Evaluator
        assert Evaluator is not None

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
