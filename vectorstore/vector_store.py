"""
Vector store adapters — unified interface for Weaviate, Pinecone, Qdrant, and Chroma.
Supports:
  - Upsert (index) chunks
  - Vector similarity search
  - Hybrid search (vector + BM25)
  - Metadata filtering
  - Deletion
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

from config.settings import settings, VectorStoreProvider
from utils.models import Chunk, RetrievedChunk, RetrievalStrategy

logger = logging.getLogger(__name__)
COLLECTION_NAME = "document_chunks"


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[Chunk]) -> int:
        """Index chunks. Returns count of successfully indexed chunks."""
        ...

    @abstractmethod
    async def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        ...

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 20,
        alpha: float = 0.7,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Default hybrid: run both vector and keyword, then RRF fusion.
        Override in adapters that have native hybrid (Weaviate, Qdrant).
        """
        vector_results = await self.vector_search(query_embedding, top_k, metadata_filter)
        keyword_results = await self.keyword_search(query, top_k, metadata_filter)
        return self._rrf_fusion(vector_results, keyword_results, alpha)

    def _rrf_fusion(
        self,
        vector_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
        alpha: float,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion with alpha weighting."""
        k = 60  # RRF constant
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for rank, rc in enumerate(vector_results):
            cid = str(rc.chunk.chunk_id)
            scores[cid] = scores.get(cid, 0) + alpha * (1 / (k + rank + 1))
            chunks[cid] = rc

        for rank, rc in enumerate(keyword_results):
            cid = str(rc.chunk.chunk_id)
            scores[cid] = scores.get(cid, 0) + (1 - alpha) * (1 / (k + rank + 1))
            if cid not in chunks:
                chunks[cid] = rc

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for rank, cid in enumerate(sorted_ids):
            rc = chunks[cid]
            rc.score = scores[cid]
            rc.rank = rank
            rc.strategy_used = RetrievalStrategy.HYBRID
            result.append(rc)
        return result

    @abstractmethod
    async def delete_document(self, doc_id: UUID) -> int:
        """Delete all chunks for a given document. Returns count deleted."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Weaviate adapter (native hybrid support)
# ─────────────────────────────────────────────────────────────────────────────

class WeaviateVectorStore(BaseVectorStore):
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import weaviate
            self._client = weaviate.connect_to_local(
                host=settings.weaviate_url.replace("http://", "").split(":")[0],
                port=int(settings.weaviate_url.split(":")[-1]),
            )
            self._ensure_schema()
        return self._client

    def _ensure_schema(self):
        client = self._client
        if not client.collections.exists(COLLECTION_NAME):
            from weaviate.classes.config import Configure, Property, DataType
            client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),  # we supply vectors
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="parent_chunk_id", data_type=DataType.TEXT),
                ],
            )
            logger.info("Created Weaviate collection: %s", COLLECTION_NAME)

    async def upsert(self, chunks: list[Chunk]) -> int:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_upsert, chunks)

    def _sync_upsert(self, chunks: list[Chunk]) -> int:
        client = self._get_client()
        collection = client.collections.get(COLLECTION_NAME)
        count = 0
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                props = {
                    "content": chunk.content,
                    "doc_id": str(chunk.doc_id),
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.metadata.get("source", ""),
                    "doc_type": chunk.metadata.get("doc_type", ""),
                    "title": chunk.metadata.get("title", ""),
                    "parent_chunk_id": str(chunk.parent_chunk_id) if chunk.parent_chunk_id else "",
                }
                batch.add_object(
                    properties=props,
                    vector=chunk.embedding,
                    uuid=str(chunk.chunk_id),
                )
                count += 1
        return count

    async def vector_search(self, query_embedding, top_k=20, metadata_filter=None):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_vector_search, query_embedding, top_k, metadata_filter)

    def _sync_vector_search(self, query_embedding, top_k, metadata_filter):
        from weaviate.classes.query import MetadataQuery, Filter
        client = self._get_client()
        collection = client.collections.get(COLLECTION_NAME)

        filters = self._build_filter(metadata_filter)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(certainty=True),
            filters=filters,
        )
        return [self._to_retrieved(obj, RetrievalStrategy.SEMANTIC) for obj in results.objects]

    async def keyword_search(self, query, top_k=20, metadata_filter=None):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_keyword_search, query, top_k, metadata_filter)

    def _sync_keyword_search(self, query, top_k, metadata_filter):
        from weaviate.classes.query import MetadataQuery
        client = self._get_client()
        collection = client.collections.get(COLLECTION_NAME)

        filters = self._build_filter(metadata_filter)
        results = collection.query.bm25(
            query=query,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
            filters=filters,
        )
        return [self._to_retrieved(obj, RetrievalStrategy.KEYWORD) for obj in results.objects]

    async def hybrid_search(self, query, query_embedding, top_k=20, alpha=0.7, metadata_filter=None):
        """Weaviate native hybrid — better than RRF post-processing."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_hybrid, query, query_embedding, top_k, alpha, metadata_filter)

    def _sync_hybrid(self, query, query_embedding, top_k, alpha, metadata_filter):
        from weaviate.classes.query import MetadataQuery
        client = self._get_client()
        collection = client.collections.get(COLLECTION_NAME)

        filters = self._build_filter(metadata_filter)
        results = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
            filters=filters,
        )
        return [self._to_retrieved(obj, RetrievalStrategy.HYBRID) for obj in results.objects]

    async def delete_document(self, doc_id: UUID) -> int:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_delete, str(doc_id))

    def _sync_delete(self, doc_id_str: str) -> int:
        from weaviate.classes.query import Filter
        client = self._get_client()
        collection = client.collections.get(COLLECTION_NAME)
        result = collection.data.delete_many(
            where=Filter.by_property("doc_id").equal(doc_id_str)
        )
        return result.successful

    def _build_filter(self, metadata_filter: Optional[dict]):
        if not metadata_filter:
            return None
        from weaviate.classes.query import Filter
        filters = None
        for key, value in metadata_filter.items():
            f = Filter.by_property(key).equal(value)
            filters = f if filters is None else filters & f
        return filters

    def _to_retrieved(self, obj, strategy: RetrievalStrategy) -> RetrievedChunk:
        from utils.models import Chunk, DocumentMetadata
        from uuid import UUID
        props = obj.properties
        chunk = Chunk(
            content=props.get("content", ""),
            doc_id=UUID(props.get("doc_id", str(UUID(int=0)))),
            chunk_id=obj.uuid,
            chunk_index=props.get("chunk_index", 0),
            metadata={
                "source": props.get("source", ""),
                "doc_type": props.get("doc_type", ""),
                "title": props.get("title", ""),
            },
        )
        score = 0.0
        if obj.metadata:
            score = obj.metadata.certainty or obj.metadata.score or 0.0
        return RetrievedChunk(chunk=chunk, score=float(score), strategy_used=strategy)


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone adapter
# ─────────────────────────────────────────────────────────────────────────────

class PineconeVectorStore(BaseVectorStore):
    def __init__(self):
        self._index = None

    def _get_index(self):
        if self._index is None:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            self._index = pc.Index(settings.pinecone_index_name)
        return self._index

    async def upsert(self, chunks: list[Chunk]) -> int:
        index = self._get_index()
        vectors = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            vectors.append({
                "id": str(chunk.chunk_id),
                "values": chunk.embedding,
                "metadata": {
                    "content": chunk.content[:1000],  # Pinecone metadata limit
                    "doc_id": str(chunk.doc_id),
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.metadata.get("source", ""),
                    "title": chunk.metadata.get("title", ""),
                },
            })
        if vectors:
            index.upsert(vectors=vectors, batch_size=100)
        return len(vectors)

    async def vector_search(self, query_embedding, top_k=20, metadata_filter=None):
        index = self._get_index()
        filter_dict = metadata_filter or {}
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict or None,
        )
        return [self._to_retrieved(match, RetrievalStrategy.SEMANTIC) for match in results.matches]

    async def keyword_search(self, query, top_k=20, metadata_filter=None):
        # Pinecone does not have native BM25; this is a placeholder.
        # In production, pair with Elasticsearch or Typesense for keyword search.
        logger.warning("Pinecone: keyword search not natively supported; returning empty.")
        return []

    async def delete_document(self, doc_id: UUID) -> int:
        index = self._get_index()
        results = index.query(
            vector=[0.0] * settings.embedding_dimension,
            top_k=10000,
            filter={"doc_id": str(doc_id)},
            include_metadata=False,
        )
        ids = [m.id for m in results.matches]
        if ids:
            index.delete(ids=ids)
        return len(ids)

    def _to_retrieved(self, match, strategy) -> RetrievedChunk:
        from utils.models import Chunk, DocumentMetadata
        from uuid import UUID
        meta = match.metadata or {}
        chunk = Chunk(
            content=meta.get("content", ""),
            doc_id=UUID(meta.get("doc_id", str(UUID(int=0)))),
            chunk_index=meta.get("chunk_index", 0),
            metadata=meta,
        )
        return RetrievedChunk(chunk=chunk, score=float(match.score), strategy_used=strategy)


# ─────────────────────────────────────────────────────────────────────────────
# Chroma (dev/test only)
# ─────────────────────────────────────────────────────────────────────────────

class ChromaVectorStore(BaseVectorStore):
    def __init__(self):
        import chromadb
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    async def upsert(self, chunks: list[Chunk]) -> int:
        ids, embeddings, documents, metadatas = [], [], [], []
        for c in chunks:
            if c.embedding is None:
                continue
            ids.append(str(c.chunk_id))
            embeddings.append(c.embedding)
            documents.append(c.content)
            metadatas.append({
                "doc_id": str(c.doc_id),
                "chunk_index": c.chunk_index,
                "source": c.metadata.get("source", ""),
                "title": c.metadata.get("title", ""),
            })
        if ids:
            self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(ids)

    async def vector_search(self, query_embedding, top_k=20, metadata_filter=None):
        where = metadata_filter or None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        chunks = []
        for i, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        )):
            from utils.models import Chunk
            from uuid import UUID
            chunk = Chunk(
                content=doc,
                doc_id=UUID(meta.get("doc_id", str(UUID(int=0)))),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
            score = 1 - dist  # Chroma returns L2 distance
            chunks.append(RetrievedChunk(chunk=chunk, score=score, strategy_used=RetrievalStrategy.SEMANTIC, rank=i))
        return chunks

    async def keyword_search(self, query, top_k=20, metadata_filter=None):
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        return []  # Chroma BM25 is limited; treat as empty for now

    async def delete_document(self, doc_id: UUID) -> int:
        self._collection.delete(where={"doc_id": str(doc_id)})
        return 0  # Chroma doesn't return count


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store(provider: VectorStoreProvider = None) -> BaseVectorStore:
    provider = provider or settings.vector_store_provider
    mapping = {
        VectorStoreProvider.WEAVIATE: WeaviateVectorStore,
        VectorStoreProvider.PINECONE: PineconeVectorStore,
        VectorStoreProvider.CHROMA: ChromaVectorStore,
    }
    cls = mapping.get(provider)
    if not cls:
        raise ValueError(f"Unknown vector store: {provider}")
    return cls()
