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
