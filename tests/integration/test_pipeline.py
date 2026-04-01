"""Integration tests — test the full RAG pipeline with mocks.
Run with: pytest tests/integration/ -v
"""
from utils.models import (
    DocumentMetadata, Document,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Test that core pipeline components work together."""

    def test_document_to_chunks_to_embeddings_flow(self):
        """Test flow: Document → Chunks → Embeddings."""
        from ingestion.chunkers.chunkers import get_chunker
        from embeddings.embedder import get_embedder

        # Create document
        meta = DocumentMetadata(source="test.pdf", doc_type="pdf")
        doc = Document(
            content="This is a test document. It has multiple sentences. And paragraphs.",
            metadata=meta
        )

        # Chunk it
        chunker = get_chunker("fixed", chunk_size=50)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        assert all(isinstance(c, dict) for c in chunks)

        # Embed it
        embedder = get_embedder("minilm")
        result = embedder.embed_texts([c["text"] for c in chunks])
        assert "embeddings" in result
        assert len(result["embeddings"]) == len(chunks)

    def test_retrieval_executor_initialization(self):
        """Test that retrieval executor can be initialized and used."""
        from retrieval.strategies.retrieval_executor import RetrievalExecutor

        executor = RetrievalExecutor()
        assert executor is not None
        assert executor.embedder is not None
        assert executor.vector_store is not None

    def test_llm_service_providers_available(self):
        """Test that all LLM providers are available."""
        from generation.llm_service import LLMService, PROVIDERS

        for provider_name in ["groq", "openai", "anthropic", "ollama", "demo"]:
            assert provider_name in PROVIDERS
            service = LLMService(provider=provider_name)
            assert service is not None

    def test_demo_mode_generation_works(self):
        """Test that demo mode generation works without API keys."""
        from generation.llm_service import LLMService

        service = LLMService(provider="demo")
        result = service.generate(
            query="What is this?",
            chunks=[
                {"text_preview": "Some text", "source_doc": "doc.pdf", "page": 1}
            ]
        )

        assert result is not None
        assert "answer" in result
        assert "provider" in result
        assert "DEMO" in result["provider"]

    def test_reranker_initialization_and_use(self):
        """Test that reranker can be initialized and used."""
        from reranking.reranker import Reranker

        reranker = Reranker()
        assert reranker is not None

        chunks = [
            {"rank": 1, "score": 0.8, "text_preview": "First result"},
            {"rank": 2, "score": 0.7, "text_preview": "Second result"},
        ]

        result = reranker.rerank("test query", chunks, top_k=2)
        assert "before" in result
        assert "after" in result
