"""
=============================================================================
orchestrator.py — The Main Brain of the RAG System
=============================================================================
PURPOSE:
    This is the single entry point for ALL query processing.
    It connects every other module together in the right order.

FLOW (what happens when a user asks a question):
    1. QueryRouter  → Understands WHAT the question is asking
    2. RetrievalExecutor → FINDS relevant document chunks
    3. Reranker     → RANKS chunks from most to least relevant
    4. PromptBuilder → PACKS chunks into a prompt for the LLM
    5. GenerationService → GENERATES the answer using the LLM
    6. Returns answer + citations back to the user

HOW TO USE:
    from rag_system.orchestrator import RAGOrchestrator
    orch = RAGOrchestrator()
    result = await orch.query(QueryRequest(query="What is X?"))
=============================================================================
"""

# --- Standard library ---
from __future__ import annotations   # Allows forward type references
import logging                        # For writing log messages
import re                             # For cite-tag parsing
import time                           # For measuring how long things take
from typing import AsyncIterator, Optional  # Type hints

import structlog

# --- Internal modules ---
from config.settings import settings   # All configuration from .env file

# Domain models — shared data structures used across the system
from utils.models import (
    QueryRequest,          # Incoming query from the user (via API)
    QueryResponse,         # Final response sent back to the user
    QueryType,             # Enum: FACTUAL, SEMANTIC, MULTI_HOP, etc.
    RetrievalStrategy,     # Enum: SEMANTIC, HYBRID, KEYWORD, etc.
)

# The five pipeline modules (one per step)
from retrieval.router.query_router import QueryRouter, RoutingDecision  # Step 1
from retrieval.strategies.retrieval_executor import RetrievalExecutor  # Step 2
from reranking.reranker import get_reranker                    # Step 3
from generation.prompts.prompt_builder import PromptBuilder    # Step 4
from generation.llm_service import GenerationService, get_generation_service  # Step 5
from monitoring.tracer import get_tracer                       # Observability

# Set up logger for this file — messages appear in console as "orchestrator: ..."
logger = structlog.get_logger(__name__)

_TRAILING_MARKER_RE = re.compile(r'<cite source="([^"]*)">')
# Matches inline [Source N] notation (with surrounding whitespace) emitted by
# GPT-4o-mini / Llama alongside the structured <cite> marker.
_SOURCE_BRACKET_RE = re.compile(r'\s*\[Source \d+\]\s*')


def _extract_attributed_spans(answer: str) -> tuple[str, list[dict]]:
    """
    Extract trailing citation markers emitted by Llama and attribute them
    to the preceding clause.

    Llama emits <cite source="N"> as a trailing marker (similar to its
    native [Source N] pattern) rather than wrapping spans. This parser
    detects each marker, attributes the preceding clause (text from the
    last sentence boundary ". ", "! ", or "? " before the marker) to the
    named source, and strips all markers from the cleaned answer.

    Heuristics documented:
    - Clause boundary: last ". ", "! ", or "? " before the marker, or
      start of text since the previous marker if no boundary found.
    - Empty clauses (marker at start of text, or two consecutive markers
      with nothing between them) are skipped; no span recorded.
    - Multi-source single sentence handled: second marker sees zero text
      between markers → skipped.
    - source_num must be a valid integer; non-integer group → span skipped.
    - [Source N] inline brackets are stripped from both cleaned_answer and
      span.text (models emit both notations as a pair; brackets are redundant
      in the structured output).

    Returns:
        (cleaned_answer, spans)
        cleaned_answer: original text with all <cite source="N"> markers and
                        [Source N] brackets removed.
        spans: list of {"source": int, "start": int, "end": int, "text": str}
               Offsets reference positions in cleaned_answer.

    Graceful fallback: any exception returns (answer, []).
    """
    spans: list[dict] = []
    cleaned_parts: list[str] = []
    last_consumed = 0   # position in original answer
    cleaned_pos = 0     # position being built in cleaned answer

    try:
        for match in _TRAILING_MARKER_RE.finditer(answer):
            # Text between last marker end and this marker start.
            # Strip [Source N] brackets so cleaned_answer omits them.
            between = answer[last_consumed : match.start()]
            between_clean = _SOURCE_BRACKET_RE.sub(' ', between)
            cleaned_parts.append(between_clean)
            cleaned_pos += len(between_clean)

            # Locate clause boundary within the cleaned segment
            boundary = max(
                between_clean.rfind(". "),
                between_clean.rfind("! "),
                between_clean.rfind("? "),
            )
            if boundary >= 0:
                raw_clause = between_clean[boundary + 2:]  # skip ". " / "! " / "? "
            else:
                raw_clause = between_clean                  # whole segment since last marker

            # Strip trailing whitespace and track how many chars were removed
            # so span_end can be adjusted to maintain the offset invariant.
            rstrip_count = len(raw_clause) - len(raw_clause.rstrip())
            clause_text = raw_clause.strip()

            if not clause_text:
                last_consumed = match.end()
                continue

            try:
                source_num = int(match.group(1))
            except ValueError:
                last_consumed = match.end()
                continue

            span_end = cleaned_pos - rstrip_count
            span_start = max(0, span_end - len(clause_text))
            spans.append({
                "source": source_num,
                "start": span_start,
                "end": span_end,
                "text": clause_text,
            })

            last_consumed = match.end()

        # Remainder after last marker — also strip [Source N] brackets
        cleaned_parts.append(_SOURCE_BRACKET_RE.sub(' ', answer[last_consumed:]))
        return "".join(cleaned_parts), spans

    except Exception as e:
        logger.warning("Attribution span parsing failed: %s", e)
        return answer, []


class RAGOrchestrator:
    """
    Orchestrates the full RAG pipeline from query to answer.

    All five pipeline steps are injected as dependencies.
    This makes it easy to:
      - Swap any step with a different implementation
      - Mock steps in tests without touching real services
      - Configure each step independently
    """

    def __init__(
        self,
        router: Optional[QueryRouter] = None,
        executor: Optional[RetrievalExecutor] = None,
        generation: Optional[GenerationService] = None,
        max_context_tokens: int = 6000,
    ):
        """
        Create the orchestrator. All parameters are optional —
        if not provided, each module creates itself using .env settings.

        Args:
            router: Query classification and strategy selection module
            executor: Document retrieval module
            generation: LLM answer generation module
            max_context_tokens: Maximum tokens to use for retrieved context
                                 (leave room for the LLM's answer)
        """
        # Step 1: Query Router — decides how to retrieve (hybrid, keyword, etc.)
        self._router = router or QueryRouter()

        # Step 2: Retrieval Executor — actually searches the vector store
        self._executor = executor or RetrievalExecutor()

        # Step 3: Reranker — re-scores chunks by relevance to the specific query
        # get_reranker() returns CrossEncoderReranker or NoOpReranker based on settings
        self._reranker = get_reranker()

        # Step 4: Prompt Builder — packs retrieved chunks into an LLM prompt
        # max_context_tokens limits how much document text goes into the prompt
        self._prompt_builder = PromptBuilder(max_context_tokens=max_context_tokens)

        # Step 5: Generation Service — calls the LLM and returns the answer
        self._generation = generation or get_generation_service()

        # Tracer — records each step's timing and results for debugging
        # Sends data to Langfuse if configured, otherwise just logs locally
        self._tracer = get_tracer()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user's query through the full RAG pipeline.

        This is the main method you call for synchronous (non-streaming) queries.
        The full answer is generated before returning.

        Args:
            request: QueryRequest object with the user's question and options

        Returns:
            QueryResponse with answer text, citations, timing, and metadata

        Raises:
            Exception: If any pipeline step fails (with tracing logged)
        """
        # Record when we started — used to calculate total latency at the end
        start = time.monotonic()
        # Per-stage timing (perf_counter for high-resolution wall time)
        t0 = time.perf_counter()

        # Start a trace — creates a unique ID to track this query through all steps
        trace_id = await self._tracer.start_trace(request.query)

        try:
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: ROUTE — Understand the query and choose retrieval strategy
            # ─────────────────────────────────────────────────────────────────
            # The router classifies the query type (factual? analytical? multi-hop?)
            # and chooses the best retrieval strategy (hybrid? keyword? semantic?)
            # It may also expand the query into multiple paraphrases for better recall
            decision = await self._router.route(
                query=request.query,
                strategy_override=request.strategy_override,  # User can force a strategy
                metadata_filter=request.metadata_filter,       # User can filter by doc type etc.
            )
            t_router = time.perf_counter()
            # Log routing decision to trace (query_type, strategy chosen, etc.)
            await self._tracer.log_routing(trace_id, decision)

            # ─────────────────────────────────────────────────────────────────
            # STEP 2: RETRIEVE — Search the vector store for relevant chunks
            # ─────────────────────────────────────────────────────────────────
            # Uses the strategy from Step 1 to search for relevant document chunks.
            # Returns the top_k most relevant chunks as RetrievedChunk objects.
            top_k = request.top_k or settings.retrieval_top_k  # From .env or request
            retrieval_result = await self._executor.execute(decision, top_k=top_k, corpus=request.corpus)
            t_retrieve = time.perf_counter()
            # Log retrieval results (how many found, scores, latency)
            await self._tracer.log_retrieval(trace_id, retrieval_result)

            # Handle case where no relevant chunks were found
            if not retrieval_result.chunks:
                logger.warning(
                    "retrieval_empty",
                    query=request.query,
                    query_type=decision.query_type.value,
                    strategy=decision.primary_strategy.value,
                    top_k=top_k,
                )
                try:
                    from monitoring.metrics import retrieval_empty as _re
                    _re.labels(strategy=decision.primary_strategy.value).inc()
                except Exception as _e:
                    logger.debug("Metrics recording skipped: %s", _e)
                # Return a graceful "I couldn't find anything" response
                return self._empty_response(request.query, decision, start)

            # Snapshot pre-rerank candidates before reranking mutates order/scores
            pre_rerank_chunks = list(retrieval_result.chunks)

            # ─────────────────────────────────────────────────────────────────
            # STEP 3: RERANK — Re-score chunks to find the most relevant ones
            # ─────────────────────────────────────────────────────────────────
            # The vector store returns chunks sorted by embedding similarity.
            # The reranker (cross-encoder) does a deeper analysis of each chunk
            # against the specific query and re-sorts them by true relevance.
            # This reduces retrieval_top_k chunks → rerank_top_k chunks.
            reranked = await self._reranker.rerank(
                query=request.query,
                chunks=retrieval_result.chunks,
                top_k=settings.rerank_top_k,   # Keep only the top N after reranking
            )
            t_rerank = time.perf_counter()
            # Log reranking results (which chunks survived, new scores)
            await self._tracer.log_reranking(trace_id, reranked)

            # ─────────────────────────────────────────────────────────────────
            # STEP 4: BUILD PROMPT — Pack chunks into an LLM-ready prompt
            # ─────────────────────────────────────────────────────────────────
            # Formats the reranked chunks into a structured prompt.
            # Selects the right prompt template for the query type.
            # Labels each chunk as [Source 1], [Source 2], etc. for citations.
            # Respects the token budget — won't overfill the context window.
            prompt = self._prompt_builder.build(
                query=request.query,
                chunks=reranked,
                query_type=decision.query_type,  # Affects which template is used
            )
            t_prompt = time.perf_counter()

            # ─────────────────────────────────────────────────────────────────
            # STEP 5: GENERATE — Call the LLM to produce the final answer
            # ─────────────────────────────────────────────────────────────────
            # Per-query verification override:
            # If the request opts in via verify_claims=True and the deployed
            # default strategy is basic, upgrade to self_rag for this single
            # query. No effect when strategy is already self_rag (no-op) or
            # when verify_claims=False (default).
            effective_strategy = settings.generation_strategy
            if request.verify_claims and effective_strategy != "self_rag":
                effective_strategy = "self_rag"

            _self_rag_stats: Optional[dict] = None
            if effective_strategy == "self_rag":
                from generation.advanced_generation import SelfRAGGenerator
                self_rag_gen = SelfRAGGenerator(
                    max_additional_retrievals=settings.self_rag_max_additional_retrievals
                )
                self_rag_result = await self_rag_gen.generate(
                    query=request.query,
                    initial_chunks=reranked,
                    prompt_builder=self._prompt_builder,
                    retrieval_executor=self._executor,
                    llm_service=self._generation,
                )
                gen_answer = self_rag_result.answer
                gen_citations = prompt.citations  # Built from reranked chunks in Step 4
                gen_model = settings.llm_model
                gen_tokens = self_rag_result.total_tokens
                gen_cached = False
                _self_rag_stats = {
                    "verified_claims": self_rag_result.verified_claims,
                    "unsupported_claims": self_rag_result.unsupported_claims,
                    "additional_retrievals": self_rag_result.additional_retrievals,
                    "faithfulness_score": self_rag_result.faithfulness_score,
                    "regenerated": self_rag_result.additional_retrievals > 0,
                }
            elif effective_strategy == "flare":
                from generation.advanced_generation import FLAREGenerator
                flare_gen = FLAREGenerator(
                    llm_service=self._generation,
                    retrieval_executor=self._executor,
                    prompt_builder=self._prompt_builder,
                    max_retrieval_rounds=settings.flare_max_retrieval_rounds,
                )
                flare_result = await flare_gen.generate(
                    query=request.query,
                    initial_chunks=reranked,
                )
                logger.info(
                    "flare_complete",
                    retrieval_rounds=flare_result.retrieval_rounds,
                    novel_tokens_per_round=flare_result.novel_tokens_per_round,
                )
                # Build citations from the full merged chunk pool
                flare_prompt = self._prompt_builder.build(
                    query=request.query,
                    chunks=flare_result.all_chunks,
                    query_type=decision.query_type,
                )
                gen_answer = flare_result.answer
                gen_citations = flare_prompt.citations
                gen_model = settings.llm_model
                gen_tokens = flare_result.total_tokens
                gen_cached = False
            else:
                # Standard path: single LLM call with cache support
                result = await self._generation.generate(
                    query=request.query,
                    prompt=prompt,
                    query_type=decision.query_type,
                    strategy_used=decision.primary_strategy,
                )
                await self._tracer.log_generation(trace_id, result)
                gen_answer = result.answer
                gen_citations = result.citations
                gen_model = result.model_used
                gen_tokens = result.total_tokens
                gen_cached = result.cached

            t_generate = time.perf_counter()

            # Strip trailing citation markers → clean answer + attributed spans
            gen_answer, attributed_spans = _extract_attributed_spans(gen_answer)

            # Post-answer: generate follow-up questions (graceful fallback)
            follow_ups: list[str] = []
            try:
                follow_ups = await self._generation.generate_followups(
                    question=request.query,
                    answer=gen_answer,
                )
            except Exception as e:
                logger.warning("Follow-up generation failed: %s", e)

        except Exception as e:
            # If anything goes wrong, log the error and re-raise it
            # The API layer will catch this and return a 500 error to the user
            logger.error("RAG pipeline error: %s", e, exc_info=True)
            await self._tracer.log_error(trace_id, str(e))
            raise

        # Calculate total time from start to finish
        total_latency = (time.monotonic() - start) * 1000  # Convert to milliseconds
        await self._tracer.end_trace(trace_id, total_latency)

        # Build per-stage timing dict (milliseconds)
        stage_timings = {
            "router_ms": (t_router - t0) * 1000,
            "retrieve_ms": (t_retrieve - t_router) * 1000,
            "rerank_ms": (t_rerank - t_retrieve) * 1000,
            "prompt_ms": (t_prompt - t_rerank) * 1000,
            "generate_ms": (t_generate - t_prompt) * 1000,
            "total_ms": (t_generate - t0) * 1000,
        }

        try:
            from monitoring.metrics import stage_duration, generation_tokens, self_rag_claims as _src
            _strat = decision.primary_strategy.value
            for _k, _ms in stage_timings.items():
                if _k != "total_ms":
                    stage_duration.labels(stage=_k[:-3], strategy=_strat).observe(_ms / 1000.0)
            _prov = settings.llm_provider.value
            # Use per-direction split when available (standard path only); fall back to total.
            if effective_strategy not in ("self_rag", "flare") and result.input_tokens:
                generation_tokens.labels(direction="input", provider=_prov).inc(result.input_tokens)
                generation_tokens.labels(direction="output", provider=_prov).inc(result.output_tokens)
            else:
                generation_tokens.labels(direction="total", provider=_prov).inc(gen_tokens)
            if _self_rag_stats:
                _src.labels(outcome="verified").inc(len(_self_rag_stats["verified_claims"]))
                _src.labels(outcome="unsupported").inc(len(_self_rag_stats["unsupported_claims"]))
        except Exception as _e:
            logger.debug("Metrics recording skipped: %s", _e)

        # Build pre-rerank candidate list; mark which chunks survived to citations
        cited_chunk_ids = {c.chunk_id for c in gen_citations}
        retrieval_candidates = [
            {
                "rank": rc.rank,
                "doc_id": str(rc.chunk.doc_id),
                "chunk_id": str(rc.chunk.chunk_id),
                "source": rc.chunk.metadata.get("source", ""),
                "score": round(rc.score, 4),
                "used_in_answer": str(rc.chunk.chunk_id) in cited_chunk_ids,
                "excerpt": rc.chunk.content[:200],
            }
            for rc in pre_rerank_chunks
        ]

        # ─────────────────────────────────────────────────────────────────────
        # FORMAT RESPONSE — Build the final API response
        # ─────────────────────────────────────────────────────────────────────
        # Convert internal objects to a clean API-friendly format
        return QueryResponse(
            answer=gen_answer,
            citations=[
                {
                    "source": c.source,
                    "title": c.title,
                    "excerpt": c.excerpt,
                    "score": round(c.score, 4),
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                }
                for c in gen_citations
            ],
            query_type=decision.query_type.value,
            strategy_used=decision.primary_strategy.value,
            model_used=gen_model,
            total_tokens=gen_tokens,
            latency_ms=round(total_latency, 2),
            cached=gen_cached,
            self_rag_stats=_self_rag_stats,
            follow_up_questions=follow_ups if follow_ups else None,
            stage_timings=stage_timings,
            retrieval_candidates=retrieval_candidates,
            attributed_spans=attributed_spans if attributed_spans else None,
        )

    async def stream_query(
        self,
        request: QueryRequest,
    ) -> AsyncIterator[str]:
        """
        Streaming version of query() — answer words appear as they are generated.

        Steps 1-4 (routing, retrieval, reranking, prompt building) run normally.
        Only Step 5 (generation) streams — tokens arrive one by one.

        Used by the POST /query/stream API endpoint with Server-Sent Events.

        Args:
            request: Same QueryRequest as query()

        Yields:
            str: Individual text tokens as they stream from the LLM
        """
        # Steps 1-3 are identical to the synchronous path above
        decision = await self._router.route(
            query=request.query,
            strategy_override=request.strategy_override,
            metadata_filter=request.metadata_filter,
        )

        retrieval_result = await self._executor.execute(decision, corpus=request.corpus)

        # If nothing found, yield a single explanation message and stop
        if not retrieval_result.chunks:
            yield "I could not find relevant information in the documents to answer this question."
            return

        reranked = await self._reranker.rerank(
            query=request.query,
            chunks=retrieval_result.chunks,
            top_k=settings.rerank_top_k,
        )

        prompt = self._prompt_builder.build(
            query=request.query,
            chunks=reranked,
            query_type=decision.query_type,
        )

        # Step 5: Stream generation — yields tokens as the LLM produces them
        # Each yielded value is a small piece of the answer (word or partial word)
        async for token in self._generation.stream(request.query, prompt):
            yield token  # Caller sends each token to the client via SSE

    def _empty_response(self, query: str, decision: RoutingDecision, start: float) -> QueryResponse:
        """
        Returns a friendly 'nothing found' response when retrieval finds nothing.

        This happens when:
         - No documents have been indexed yet
         - The query doesn't match any indexed content
         - All retrieval strategies return empty results
        """
        return QueryResponse(
            answer=(
                "I could not find relevant information in the indexed documents "
                "to answer this question. Please verify the documents are indexed "
                "and try rephrasing your query."
            ),
            citations=[],                               # No sources to cite
            query_type=decision.query_type.value,
            strategy_used=decision.primary_strategy.value,
            model_used=settings.llm_model,             # LLM was not actually called
            total_tokens=0,                             # No tokens used
            latency_ms=round((time.monotonic() - start) * 1000, 2),
            cached=False,
        )
