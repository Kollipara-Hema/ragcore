"""
Unit tests for core RAG components.
Run with: pytest tests/unit/ -v
"""

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



class TestEvaluator:
    """Test Evaluator with golden dataset."""

    def setup_method(self, method):
        from evaluation.evaluator import Evaluator
        self.evaluator = Evaluator()

    def test_empty_samples(self):
        import tempfile, os
        # Write a CSV with only a header (zero data rows)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("query,answer\n")
            tmp_path = f.name
        try:
            metrics = self.evaluator.evaluate(tmp_path)
            assert metrics["sample_count"] == 0
        finally:
            os.unlink(tmp_path)



