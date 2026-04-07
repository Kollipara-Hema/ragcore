"""Integration tests — test the full RAG pipeline with mocks.
Run with: pytest tests/integration/ -v
"""
from utils.models import (
    DocumentMetadata, Document, RetrievalStrategy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Test that core pipeline components work together."""

    def test_document_to_chunks_to_embeddings_flow(self):
        """Test flow: Document → Chunks → Embeddings."""
        import asyncio
        from unittest.mock import AsyncMock, patch
        from ingestion.chunkers.chunkers import get_chunker
        from embeddings.embedder import get_embedder, BGEEmbedder
        from utils.models import Chunk

        # Create document
        meta = DocumentMetadata(source="test.pdf", doc_type="pdf")
        doc = Document(
            content="This is a test document. It has multiple sentences. And paragraphs.",
            metadata=meta
        )

        # Chunk it (get_chunker takes only strategy, no chunk_size kwarg)
        chunker = get_chunker("fixed")
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

        # Embed it — mock the model to avoid downloading large weights
        fake_embeddings = [[0.1] * 384] * len(chunks)
        with patch.object(BGEEmbedder, "embed_texts", new_callable=AsyncMock, return_value=fake_embeddings):
            embedder = get_embedder("minilm", cached=False)
            embeddings = asyncio.run(embedder.embed_texts([c.content for c in chunks]))
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(chunks)

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
        import asyncio
        from reranking.reranker import NoOpReranker
        from utils.models import Chunk, RetrievedChunk
        from utils.models import DocumentMetadata, Document

        # Use NoOpReranker to avoid downloading cross-encoder model weights
        reranker = NoOpReranker()
        assert reranker is not None

        meta = DocumentMetadata(source="test.pdf", doc_type="pdf")
        doc = Document(content="Test", metadata=meta)
        chunk1 = Chunk(content="First result", doc_id=doc.doc_id)
        chunk2 = Chunk(content="Second result", doc_id=doc.doc_id)
        retrieved = [
            RetrievedChunk(chunk=chunk1, score=0.8, strategy_used=RetrievalStrategy.HYBRID),
            RetrievedChunk(chunk=chunk2, score=0.7, strategy_used=RetrievalStrategy.HYBRID),
        ]

        # rerank is async and returns List[RetrievedChunk]
        result = asyncio.run(reranker.rerank("test query", retrieved, top_k=2))
        assert isinstance(result, list)
        assert len(result) == 2
