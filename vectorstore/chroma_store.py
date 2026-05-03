"""
ChromaDB vector store using BM25Index for hybrid retrieval.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import List, Optional
from uuid import UUID

from config.settings import settings
from utils.models import Chunk, RetrievedChunk, RetrievalStrategy
from vectorstore.bm25_index import BM25Index

logger = logging.getLogger(__name__)


def _coerce_filter(metadata_filter: Optional[dict]) -> Optional[dict]:
    """Coerce UUID values to strings for Chroma metadata comparison."""
    if not metadata_filter:
        return None
    return {k: str(v) if isinstance(v, UUID) else v for k, v in metadata_filter.items()}


class ChromaVectorStore:
    """Implements the BaseVectorStore interface backed by ChromaDB + BM25Index."""

    def __init__(self, persist_dir: str = None) -> None:
        # chromadb 0.4.24 uses numpy 1.x aliases (np.float_, np.int_, np.NaN, etc.)
        # that were removed in NumPy 2.0.  The Docker image pins numpy==1.26.4 so
        # this shim is a no-op in production; local environments running NumPy 2.x
        # need it to import chromadb without AttributeError.  Bumping chromadb to
        # 0.5.x (which targets NumPy 2.0) removes the need for this shim entirely.
        import numpy as _np
        for _attr, _alias in (
            ("float_", "float64"), ("int_", "intp"),
            ("complex_", "complex128"), ("uint", "uintp"),
            ("NaN", "nan"),
        ):
            if not hasattr(_np, _attr):
                setattr(_np, _attr, getattr(_np, _alias))
        import chromadb

        self._persist_dir = persist_dir or settings.chroma_persist_dir
        os.makedirs(self._persist_dir, exist_ok=True)
        self._bm25_state_path = os.path.join(self._persist_dir, "bm25_state.pkl")
        self._corpus_ids: List[str] = []
        self._bm25_index: BM25Index = BM25Index()

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._load_bm25_state()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _load_bm25_state(self) -> None:
        if not os.path.exists(self._bm25_state_path):
            return
        try:
            with open(self._bm25_state_path, "rb") as f:
                state = pickle.load(f)
            self._corpus_ids = state["corpus_ids"]
            self._bm25_index.build(state["corpus"])
        except Exception:
            logger.warning("bm25_state.pkl unreadable; rebuilding BM25 state from Chroma")
            self._rebuild_from_chroma()
            self._save_bm25_state()

    def _save_bm25_state(self) -> None:
        with open(self._bm25_state_path, "wb") as f:
            pickle.dump(
                {"corpus": self._bm25_index._corpus, "corpus_ids": self._corpus_ids},
                f,
            )

    def _rebuild_from_chroma(self) -> None:
        """Fetch all documents from Chroma (paginated), sort by chunk_id, rebuild BM25."""
        all_ids: List[str] = []
        all_documents: List[str] = []
        page_size = 1000
        offset = 0

        while True:
            page = self._collection.get(
                limit=page_size,
                offset=offset,
                include=["documents"],
            )
            page_ids = page["ids"]
            page_docs = page["documents"]
            if not page_ids:
                break
            all_ids.extend(page_ids)
            all_documents.extend(page_docs)
            if len(page_ids) < page_size:
                break
            offset += page_size

        # Lexicographic sort on chunk_id string produces deterministic ordering
        # that survives restarts and is reconstructible from Chroma alone.
        paired = sorted(zip(all_ids, all_documents), key=lambda x: x[0])
        self._corpus_ids = [p[0] for p in paired]
        self._bm25_index.build([p[1] for p in paired])

    # ── Public interface ──────────────────────────────────────────────────────

    async def upsert(self, chunks: List[Chunk]) -> int:
        chunks_with_emb = [c for c in chunks if c.embedding is not None]
        if not chunks_with_emb:
            return 0

        ids = [str(c.chunk_id) for c in chunks_with_emb]
        embeddings = [c.embedding for c in chunks_with_emb]
        documents = [c.content for c in chunks_with_emb]

        metadatas = []
        for c in chunks_with_emb:
            meta: dict = {
                "doc_id": str(c.doc_id),
                "chunk_id": str(c.chunk_id),
                "chunk_index": c.chunk_index,
            }
            for k, v in c.metadata.items():
                if isinstance(v, UUID):
                    meta[k] = str(v)
                elif isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._corpus_ids.extend(ids)
        self._bm25_index.upsert([c.content for c in chunks_with_emb])
        self._save_bm25_state()
        return len(chunks_with_emb)

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        metadata_filter: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        count = self._collection.count()
        if count == 0:
            return []

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        where = _coerce_filter(metadata_filter)
        if where is not None:
            query_kwargs["where"] = where

        result = self._collection.query(**query_kwargs)

        results = []
        for chunk_id, distance, document, meta in zip(
            result["ids"][0],
            result["distances"][0],
            result["documents"][0],
            result["metadatas"][0],
        ):
            # Chroma cosine space returns distance in [0, 2]; convert to similarity in [0, 1].
            score = max(0.0, 1.0 - distance)
            chunk = Chunk(
                content=document,
                doc_id=UUID(meta["doc_id"]),
                chunk_id=UUID(meta["chunk_id"]),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
            results.append(RetrievedChunk(chunk=chunk, score=score, strategy_used=RetrievalStrategy.SEMANTIC))

        return results

    async def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        if not self._corpus_ids:
            return []

        assert len(self._corpus_ids) == len(self._bm25_index._corpus), (
            f"corpus_ids length {len(self._corpus_ids)} != BM25 corpus length {len(self._bm25_index._corpus)}"
        )

        raw_candidates = self._bm25_index.query(query, top_k * 2)
        if not raw_candidates:
            return []

        candidate_ids_ordered = [self._corpus_ids[idx] for idx, _ in raw_candidates]
        candidate_scores = {self._corpus_ids[idx]: score for idx, score in raw_candidates}

        result = self._collection.get(
            ids=candidate_ids_ordered,
            include=["documents", "metadatas"],
        )

        # Response order from collection.get is not guaranteed to match the request order.
        lookup = {
            cid: (doc, meta)
            for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        }

        normalized_filter = _coerce_filter(metadata_filter)
        results = []
        for chunk_id in candidate_ids_ordered:
            if chunk_id not in lookup:
                continue
            document, meta = lookup[chunk_id]
            if normalized_filter and not all(meta.get(k) == v for k, v in normalized_filter.items()):
                continue
            chunk = Chunk(
                content=document,
                doc_id=UUID(meta["doc_id"]),
                chunk_id=UUID(meta["chunk_id"]),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
            results.append(
                RetrievedChunk(chunk=chunk, score=candidate_scores[chunk_id], strategy_used=RetrievalStrategy.KEYWORD)
            )
            if len(results) >= top_k:
                break

        return results

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 20,
        alpha: float = 0.7,
        metadata_filter: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        vector_results = await self.vector_search(query_embedding, top_k * 2, metadata_filter)
        keyword_results = await self.keyword_search(query, top_k * 2, metadata_filter)

        vector_scores = {str(rc.chunk.chunk_id): rc.score for rc in vector_results}
        keyword_scores_raw = {str(rc.chunk.chunk_id): rc.score for rc in keyword_results}

        max_kw = max(keyword_scores_raw.values(), default=1.0)
        keyword_scores = {k: v / max_kw for k, v in keyword_scores_raw.items()}

        chunk_map = {str(rc.chunk.chunk_id): rc for rc in vector_results + keyword_results}
        all_ids = set(vector_scores) | set(keyword_scores)

        fused = {
            cid: alpha * vector_scores.get(cid, 0.0) + (1 - alpha) * keyword_scores.get(cid, 0.0)
            for cid in all_ids
        }

        sorted_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:top_k]
        results = []
        for rank, cid in enumerate(sorted_ids):
            rc = chunk_map[cid]
            rc.score = fused[cid]
            rc.rank = rank
            rc.strategy_used = RetrievalStrategy.HYBRID
            results.append(rc)

        return results

    async def delete_document(self, doc_id: UUID) -> int:
        before_count = self._collection.count()
        self._collection.delete(where={"doc_id": str(doc_id)})
        after_count = self._collection.count()
        removed = before_count - after_count
        self._rebuild_from_chroma()
        self._save_bm25_state()
        return removed
