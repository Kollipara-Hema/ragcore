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
        assert executor._embedder is not None
        assert executor._store is not None

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


class TestVectorStoreSearch:
    """Test FAISS keyword, semantic, and hybrid search."""

    def _make_store_with_chunks(self):
        """Create an in-memory FAISSVectorStore with three pre-indexed chunks.
        Three chunks ensure BM25 IDF > 0 (avoids log(1.5/1.5)=0 with only 2 docs).
        """
        import asyncio
        import tempfile
        import os
        from uuid import uuid4
        from vectorstore.vector_store import FAISSVectorStore
        from utils.models import Chunk

        idx_file = os.path.join(tempfile.mkdtemp(), "test.idx")
        meta_file = os.path.join(tempfile.mkdtemp(), "test.pkl")
        store = FAISSVectorStore(index_file=idx_file, metadata_file=meta_file)

        doc_id = uuid4()
        chunk_a = Chunk(content="the cat sat on the mat", doc_id=doc_id, embedding=[0.1] * 8)
        chunk_b = Chunk(content="machine learning and neural networks", doc_id=doc_id, embedding=[0.9] * 8)
        chunk_c = Chunk(content="deep learning transforms computer vision", doc_id=doc_id, embedding=[0.5] * 8)
        asyncio.run(store.upsert([chunk_a, chunk_b, chunk_c]))
        return store

    def test_keyword_search_returns_results(self):
        import asyncio
        store = self._make_store_with_chunks()
        results = asyncio.run(store.keyword_search("cat mat", top_k=5))
        assert isinstance(results, list)
        assert len(results) > 0

    def test_keyword_search_ranks_relevant_chunk_first(self):
        import asyncio
        from utils.models import RetrievalStrategy
        store = self._make_store_with_chunks()
        results = asyncio.run(store.keyword_search("cat mat", top_k=5))
        assert results[0].chunk.content == "the cat sat on the mat"
        assert all(r.strategy_used == RetrievalStrategy.KEYWORD for r in results)

    def test_vector_search_returns_results(self):
        import asyncio
        store = self._make_store_with_chunks()
        results = asyncio.run(store.vector_search([0.1] * 8, top_k=5))
        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_search_returns_results(self):
        import asyncio
        from utils.models import RetrievalStrategy
        store = self._make_store_with_chunks()
        results = asyncio.run(store.hybrid_search("cat mat", [0.1] * 8, top_k=5, alpha=0.7))
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(r.strategy_used == RetrievalStrategy.HYBRID for r in results)

    def test_keyword_search_empty_when_no_data(self):
        import asyncio
        import tempfile
        import os
        from vectorstore.vector_store import FAISSVectorStore
        idx_file = os.path.join(tempfile.mkdtemp(), "empty.idx")
        meta_file = os.path.join(tempfile.mkdtemp(), "empty.pkl")
        store = FAISSVectorStore(index_file=idx_file, metadata_file=meta_file)
        results = asyncio.run(store.keyword_search("anything"))
        assert results == []


class TestStrategyMap:
    """Verify STRATEGY_MAP routes each query type to the correct strategy."""

    def test_factual_uses_hybrid(self):
        from retrieval.router.query_router import STRATEGY_MAP
        from utils.models import QueryType, RetrievalStrategy
        primary, _ = STRATEGY_MAP[QueryType.FACTUAL]
        assert primary == RetrievalStrategy.HYBRID

    def test_lookup_uses_keyword(self):
        from retrieval.router.query_router import STRATEGY_MAP
        from utils.models import QueryType, RetrievalStrategy
        primary, _ = STRATEGY_MAP[QueryType.LOOKUP]
        assert primary == RetrievalStrategy.KEYWORD

    def test_semantic_uses_semantic(self):
        from retrieval.router.query_router import STRATEGY_MAP
        from utils.models import QueryType, RetrievalStrategy
        primary, _ = STRATEGY_MAP[QueryType.SEMANTIC]
        assert primary == RetrievalStrategy.SEMANTIC

    def test_multi_hop_uses_multi_query(self):
        from retrieval.router.query_router import STRATEGY_MAP
        from utils.models import QueryType, RetrievalStrategy
        primary, _ = STRATEGY_MAP[QueryType.MULTI_HOP]
        assert primary == RetrievalStrategy.MULTI_QUERY

    def test_analytical_uses_hybrid(self):
        from retrieval.router.query_router import STRATEGY_MAP
        from utils.models import QueryType, RetrievalStrategy
        primary, _ = STRATEGY_MAP[QueryType.ANALYTICAL]
        assert primary == RetrievalStrategy.HYBRID

    def test_comparative_uses_multi_query(self):
        from retrieval.router.query_router import STRATEGY_MAP
        from utils.models import QueryType, RetrievalStrategy
        primary, _ = STRATEGY_MAP[QueryType.COMPARATIVE]
        assert primary == RetrievalStrategy.MULTI_QUERY
