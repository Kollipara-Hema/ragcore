"""
Retrieval strategies implementation.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi

from embeddings.embedder import get_embedder
from retrieval.router.query_router import RoutingDecision
from utils.models import RetrievalRequest, RetrievalResult, RetrievedChunk, RetrievalStrategy
from config.settings import settings
from vectorstore.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class RetrievalExecutor:
    """Executes retrieval strategies."""

    def __init__(self):
        self._embedder = get_embedder()
        self._store = get_vector_store()
        self.bm25 = None
        self.corpus = []
        self.metadata = []

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks for BM25."""
        self.corpus = [chunk["text"] for chunk in chunks]
        self.metadata = chunks
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Dense retrieval using vector similarity."""
        start_time = time.time()
        query_emb = self.embedder.embed_query(query)
        results = self.vector_store.search(query_emb, top_k)
        latency = time.time() - start_time

        retrieved = []
        for i, res in enumerate(results):
            retrieved.append({
                "rank": i + 1,
                "score": res["score"],
                "source_doc": res["metadata"]["source_doc"],
                "page": res["metadata"].get("page", 1),
                "text_preview": res["metadata"]["text"][:200] + "..."
            })

        return {
            "results": retrieved,
            "latency": latency
        }

    def sparse_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Sparse retrieval using BM25."""
        if not self.bm25:
            return {"results": [], "latency": 0}

        start_time = time.time()
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        latency = time.time() - start_time

        retrieved = []
        for rank, idx in enumerate(top_indices):
            score = doc_scores[idx]
            if score > 0:
                retrieved.append({
                    "rank": rank + 1,
                    "score": float(score),
                    "source_doc": self.metadata[idx]["source_doc"],
                    "page": self.metadata[idx].get("page", 1),
                    "text_preview": self.corpus[idx][:200] + "..."
                })

        return {
            "results": retrieved,
            "latency": latency
        }

    def hybrid_retrieval(self, query: str, top_k: int = 5, alpha: float = 0.5) -> Dict[str, Any]:
        """Hybrid retrieval combining dense and sparse."""
        dense_res = self.dense_retrieval(query, top_k * 2)
        sparse_res = self.sparse_retrieval(query, top_k * 2)

        # Combine scores
        combined = {}
        for res in dense_res["results"]:
            key = res["source_doc"] + str(res["page"])
            combined[key] = {
                "score": alpha * res["score"],
                "data": res
            }

        for res in sparse_res["results"]:
            key = res["source_doc"] + str(res["page"])
            if key in combined:
                combined[key]["score"] += (1 - alpha) * res["score"]
            else:
                combined[key] = {
                    "score": (1 - alpha) * res["score"],
                    "data": res
                }

        # Sort and take top_k
        sorted_combined = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]

        retrieved = []
        for rank, (key, item) in enumerate(sorted_combined):
            data = item["data"]
            retrieved.append({
                "rank": rank + 1,
                "score": item["score"],
                "source_doc": data["source_doc"],
                "page": data["page"],
                "text_preview": data["text_preview"]
            })

        latency = max(dense_res["latency"], sparse_res["latency"])

        return {
            "results": retrieved,
            "latency": latency
        }

    async def execute(self, decision: RoutingDecision, top_k: int = 5) -> RetrievalResult:
        start = time.monotonic()
        try:
            chunks = await self._dispatch(decision, top_k)
            fallback_used = False
        except Exception as e:
            logger.error("Primary retrieval failed (%s): %s — trying fallback", decision.primary_strategy, e)
            try:
                fallback_decision = RoutingDecision(
                    query_type=decision.query_type,
                    primary_strategy=decision.fallback_strategy,
                    fallback_strategy=RetrievalStrategy.SEMANTIC,
                    expanded_queries=decision.expanded_queries[:1],
                    metadata_filter=None,  # relax filter on fallback
                )
                chunks = await self._dispatch(fallback_decision, top_k)
                fallback_used = True
            except Exception as e2:
                logger.error("Fallback retrieval also failed: %s", e2)
                chunks = []
                fallback_used = True
        else:
            fallback_used = False

        latency_ms = (time.monotonic() - start) * 1000
        request = RetrievalRequest(
            query=decision.expanded_queries[0] if decision.expanded_queries else "",
            query_type=decision.query_type,
            strategy=decision.primary_strategy,
            top_k=top_k,
            metadata_filter=decision.metadata_filter,
            expanded_queries=decision.expanded_queries,
        )

        return RetrievalResult(
            request=request,
            chunks=chunks,
            latency_ms=latency_ms,
            fallback_used=fallback_used,
        )

    async def _dispatch(self, decision: RoutingDecision, top_k: int) -> list[RetrievedChunk]:
        strategy = decision.primary_strategy
        query = decision.expanded_queries[0]

        if strategy == RetrievalStrategy.SEMANTIC:
            return await self._semantic_search(query, top_k, decision.metadata_filter)

        elif strategy == RetrievalStrategy.KEYWORD:
            return await self._keyword_search(query, top_k, decision.metadata_filter)

        elif strategy == RetrievalStrategy.HYBRID:
            return await self._hybrid_search(query, top_k, decision.metadata_filter)

        elif strategy == RetrievalStrategy.METADATA_FILTER:
            return await self._metadata_filtered_search(query, top_k, decision.metadata_filter)

        elif strategy == RetrievalStrategy.MULTI_QUERY:
            return await self._multi_query_search(decision.expanded_queries, top_k, decision.metadata_filter)

        elif strategy == RetrievalStrategy.PARENT_CHILD:
            return await self._parent_child_search(query, top_k, decision.metadata_filter)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ── Strategy implementations ──────────────────────────────────────────────

    async def _semantic_search(self, query, top_k, metadata_filter) -> list[RetrievedChunk]:
        embedding = await self._embedder.embed_query(query)
        return await self._store.vector_search(embedding, top_k, metadata_filter)

    async def _keyword_search(self, query, top_k, metadata_filter) -> list[RetrievedChunk]:
        return await self._store.keyword_search(query, top_k, metadata_filter)

    async def _hybrid_search(self, query, top_k, metadata_filter) -> list[RetrievedChunk]:
        embedding = await self._embedder.embed_query(query)
        return await self._store.hybrid_search(
            query=query,
            query_embedding=embedding,
            top_k=top_k,
            alpha=settings.hybrid_alpha,
            metadata_filter=metadata_filter,
        )

    async def _metadata_filtered_search(self, query, top_k, metadata_filter) -> list[RetrievedChunk]:
        """Strict metadata filter + vector search within filtered set."""
        if not metadata_filter:
            logger.warning("Metadata strategy selected but no filters provided; falling back to hybrid")
            return await self._hybrid_search(query, top_k, None)
        embedding = await self._embedder.embed_query(query)
        return await self._store.vector_search(embedding, top_k, metadata_filter)

    async def _multi_query_search(self, queries: list[str], top_k: int, metadata_filter) -> list[RetrievedChunk]:
        """
        Run retrieval for each expanded query in parallel, then merge
        with Reciprocal Rank Fusion.
        """
        tasks = [
            self._hybrid_search(q, top_k, metadata_filter) for q in queries
        ]
        all_result_sets = await asyncio.gather(*tasks, return_exceptions=True)

        # Deduplicate and fuse across query result sets
        all_chunks: list[list[RetrievedChunk]] = []
        for result in all_result_sets:
            if isinstance(result, list):
                all_chunks.append(result)
            else:
                logger.warning("Sub-query retrieval failed: %s", result)

        if not all_chunks:
            return []

        # RRF across result sets
        k = 60
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for result_set in all_chunks:
            for rank, rc in enumerate(result_set):
                cid = str(rc.chunk.chunk_id)
                scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
                if cid not in chunk_map:
                    chunk_map[cid] = rc

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        merged = []
        for i, cid in enumerate(sorted_ids[:top_k]):
            rc = chunk_map[cid]
            rc.score = scores[cid]
            rc.rank = i
            rc.strategy_used = RetrievalStrategy.MULTI_QUERY
            merged.append(rc)

        return merged

    async def _parent_child_search(self, query, top_k, metadata_filter) -> list[RetrievedChunk]:
        """
        1. Search by child chunk embeddings (small, precise)
        2. Look up parent chunks for each hit
        3. Return parent chunks to LLM for fuller context
        """
        # Only search child chunks
        child_filter = dict(metadata_filter or {})
        child_filter["is_child_chunk"] = True

        embedding = await self._embedder.embed_query(query)
        child_results = await self._store.vector_search(embedding, top_k * 2, child_filter)

        # Collect unique parent IDs
        parent_ids: list[str] = []
        seen = set()
        for rc in child_results:
            pid = rc.chunk.metadata.get("parent_chunk_id")
            if pid and pid not in seen:
                parent_ids.append(pid)
                seen.add(pid)

        if not parent_ids:
            logger.warning("Parent-child: no parent IDs found; falling back to child chunks")
            return child_results[:top_k]

        # Fetch parent chunks by ID
        # Note: requires vector store to support ID-based fetch (add to adapter)
        # For now, use a keyword search to approximate parent lookup
        parent_results = []
        for cid in parent_ids[:top_k]:
            # In a real implementation, fetch by chunk_id directly
            # This is a placeholder showing the pattern
            parent_results.append(child_results[0])  # simplified

        return parent_results[:top_k]
