"""
Retrieval strategies implementation.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import List

import structlog

import numpy as np

from typing import Optional

from embeddings.embedder import get_embedder
from retrieval.router.query_router import RoutingDecision
from utils.models import RetrievalRequest, RetrievalResult, RetrievedChunk, RetrievalStrategy
from config.settings import settings
from vectorstore.vector_store import BaseVectorStore, get_corpus, get_vector_store

logger = structlog.get_logger(__name__)


class RetrievalExecutor:
    """Executes retrieval strategies.

    Stateless w.r.t. the vector store — the store is looked up per-request via
    the corpus parameter on execute(). A single RetrievalExecutor instance can
    safely serve concurrent requests targeting different corpora; the store
    reference is threaded through dispatch as a parameter rather than mutated
    on the instance, so no cross-request contamination is possible.
    """

    def __init__(self):
        self._embedder = get_embedder()

    async def execute(
        self,
        decision: RoutingDecision,
        top_k: int = 5,
        corpus: str = "default",
        *,
        store_override: Optional[BaseVectorStore] = None,
    ) -> RetrievalResult:
        # Store resolution order:
        #   1. store_override (set by the API handler when the request carries a
        #      valid X-Session-Id). When present, `corpus` is unused — the
        #      override wins. This is what keeps session routing structurally
        #      separate from the public corpus string: a session token CANNOT
        #      reach get_corpus() through this path.
        #   2. Public registry, keyed by `corpus`. Resolved BEFORE the try/except
        #      so an unknown corpus surfaces as a clean KeyError rather than
        #      getting swallowed by the fallback path and producing a misleading
        #      "no results" response.
        if store_override is not None:
            store = store_override
        else:
            store = get_vector_store() if corpus == "default" else get_corpus(corpus)

        start = time.monotonic()
        try:
            chunks = await self._dispatch(decision, top_k, store)
            fallback_used = False
        except Exception as e:
            logger.error(
                "retrieval_failed",
                strategy=str(decision.primary_strategy),
                error=str(e),
                error_type=type(e).__name__,
                fallback=True,
            )
            try:
                fallback_decision = RoutingDecision(
                    query_type=decision.query_type,
                    primary_strategy=decision.fallback_strategy,
                    fallback_strategy=RetrievalStrategy.SEMANTIC,
                    expanded_queries=decision.expanded_queries[:1],
                    metadata_filter=None,  # relax filter on fallback
                )
                chunks = await self._dispatch(fallback_decision, top_k, store)
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

    async def _dispatch(
        self, decision: RoutingDecision, top_k: int, store: BaseVectorStore,
    ) -> list[RetrievedChunk]:
        strategy = decision.primary_strategy
        query = decision.expanded_queries[0]

        if strategy == RetrievalStrategy.SEMANTIC:
            return await self._semantic_search(query, top_k, decision.metadata_filter, store)

        elif strategy == RetrievalStrategy.KEYWORD:
            return await self._keyword_search(query, top_k, decision.metadata_filter, store)

        elif strategy == RetrievalStrategy.HYBRID:
            return await self._hybrid_search(query, top_k, decision.metadata_filter, store)

        elif strategy == RetrievalStrategy.METADATA_FILTER:
            return await self._metadata_filtered_search(query, top_k, decision.metadata_filter, store)

        elif strategy == RetrievalStrategy.MULTI_QUERY:
            return await self._multi_query_search(decision.expanded_queries, top_k, decision.metadata_filter, store)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ── Strategy implementations ──────────────────────────────────────────────

    async def _semantic_search(self, query, top_k, metadata_filter, store) -> list[RetrievedChunk]:
        embedding = await self._embedder.embed_query(query)
        return await store.vector_search(embedding, top_k, metadata_filter)

    async def _keyword_search(self, query, top_k, metadata_filter, store) -> list[RetrievedChunk]:
        return await store.keyword_search(query, top_k, metadata_filter)

    async def _hybrid_search(self, query, top_k, metadata_filter, store) -> list[RetrievedChunk]:
        embedding = await self._embedder.embed_query(query)
        return await store.hybrid_search(
            query=query,
            query_embedding=embedding,
            top_k=top_k,
            alpha=settings.hybrid_alpha,
            metadata_filter=metadata_filter,
        )

    async def _metadata_filtered_search(self, query, top_k, metadata_filter, store) -> list[RetrievedChunk]:
        """Strict metadata filter + vector search within filtered set."""
        if not metadata_filter:
            logger.warning("Metadata strategy selected but no filters provided; falling back to hybrid")
            return await self._hybrid_search(query, top_k, None, store)
        embedding = await self._embedder.embed_query(query)
        return await store.vector_search(embedding, top_k, metadata_filter)

    async def _multi_query_search(self, queries: list[str], top_k: int, metadata_filter, store) -> list[RetrievedChunk]:
        """
        Run retrieval for each expanded query in parallel, then merge
        with Reciprocal Rank Fusion.
        """
        tasks = [
            self._hybrid_search(q, top_k, metadata_filter, store) for q in queries
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

