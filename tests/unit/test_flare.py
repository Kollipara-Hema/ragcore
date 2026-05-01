"""
Unit tests for FLARE-inspired generator orchestrator integration.

These tests verify that:
  - generation_strategy == "flare" → FLAREGenerator.generate() is called
  - generation_strategy == "basic" → FLAREGenerator is never called
  - Iterative retrieval fires when dollar-token novelty is detected
  - Re-retrieval query includes the original query AND novel tokens
  - Chunk pool is deduped across retrieval rounds
  - Loop terminates at flare_max_retrieval_rounds even with persistent novelty
"""
from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id="chunk-001", content="The 2024 IRA limit is $6,500."):
    chunk = MagicMock()
    chunk.chunk.chunk_id = chunk_id
    chunk.chunk.content = content
    chunk.score = 0.9
    return chunk


def _make_retrieval_result(chunks=None):
    result = MagicMock()
    result.chunks = chunks if chunks is not None else [_make_chunk()]
    return result


def _make_routing_decision():
    from utils.models import QueryType, RetrievalStrategy
    decision = MagicMock()
    decision.query_type = QueryType.FACTUAL
    decision.primary_strategy = RetrievalStrategy.HYBRID
    decision.metadata_filter = None
    return decision


def _make_prompt(citations=None):
    prompt = MagicMock()
    prompt.citations = citations or []
    prompt.messages = [{"role": "user", "content": "test"}]
    for c in prompt.citations:
        c.chunk_id = "cite-001"
    return prompt


def _make_generation_result(answer="basic answer"):
    from utils.models import QueryType, RetrievalStrategy
    result = MagicMock()
    result.answer = answer
    result.citations = []
    result.model_used = "llama-3.1-70b-versatile"
    result.total_tokens = 100
    result.cached = False
    result.query_type = QueryType.FACTUAL
    result.strategy_used = RetrievalStrategy.HYBRID
    return result


def _make_flare_result(answer="flare answer", chunk_ids=("chunk-001",), rounds=1):
    from generation.advanced_generation import FLAREResult
    chunks = [_make_chunk(cid) for cid in chunk_ids]
    return FLAREResult(
        answer=answer,
        all_chunks=chunks,
        retrieval_rounds=rounds,
        total_tokens=200,
        novel_tokens_per_round=[[] for _ in range(rounds)],
    )


def _build_orchestrator(mock_generation):
    from orchestrator import RAGOrchestrator
    mock_router = AsyncMock()
    mock_router.route = AsyncMock(return_value=_make_routing_decision())

    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=_make_retrieval_result())

    orch = RAGOrchestrator(
        router=mock_router,
        executor=mock_executor,
        generation=mock_generation,
    )
    mock_reranker = AsyncMock()
    mock_reranker.rerank = AsyncMock(return_value=[_make_chunk()])
    orch._reranker = mock_reranker

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build = MagicMock(return_value=_make_prompt())
    orch._prompt_builder = mock_prompt_builder

    return orch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFLAREOrchestration:

    def test_flare_dispatches_when_strategy_set(self):
        """GENERATION_STRATEGY=flare causes orchestrator to call FLAREGenerator.generate()."""
        from utils.models import QueryRequest

        mock_generation = AsyncMock()
        orch = _build_orchestrator(mock_generation)

        fake_flare_result = _make_flare_result()

        with patch("orchestrator.settings") as mock_settings:
            mock_settings.generation_strategy = "flare"
            mock_settings.flare_max_retrieval_rounds = 3
            mock_settings.retrieval_top_k = 20
            mock_settings.rerank_top_k = 5
            mock_settings.llm_model = "llama-3.1-70b-versatile"

            with patch(
                "generation.advanced_generation.FLAREGenerator.generate",
                new_callable=AsyncMock,
                return_value=fake_flare_result,
            ) as mock_flare_generate:
                response = asyncio.run(orch.query(QueryRequest(query="What is X?")))

        mock_flare_generate.assert_called_once()
        assert response.answer == "flare answer"
        mock_generation.generate.assert_not_called()

    def test_basic_path_unaffected_by_flare_code(self):
        """GENERATION_STRATEGY=basic never calls FLAREGenerator."""
        from utils.models import QueryRequest

        mock_generation = AsyncMock()
        mock_generation.generate = AsyncMock(return_value=_make_generation_result())
        orch = _build_orchestrator(mock_generation)

        with patch("orchestrator.settings") as mock_settings:
            mock_settings.generation_strategy = "basic"
            mock_settings.retrieval_top_k = 20
            mock_settings.rerank_top_k = 5
            mock_settings.llm_model = "llama-3.1-70b-versatile"

            with patch(
                "generation.advanced_generation.FLAREGenerator.generate",
                new_callable=AsyncMock,
            ) as mock_flare_generate:
                response = asyncio.run(orch.query(QueryRequest(query="What is X?")))

        mock_generation.generate.assert_called_once()
        mock_flare_generate.assert_not_called()
        assert response.answer == "basic answer"

    def test_flare_iterative_retrieval_on_novel_dollar_tokens(self):
        """When LLM emits a $-token absent from chunks, execute() is called a second time."""
        from generation.advanced_generation import FLAREGenerator
        from utils.models import QueryType, RetrievalStrategy

        # Initial chunk contains $6,500; LLM answer will also mention $1,000 (novel)
        initial_chunk = _make_chunk(chunk_id="chunk-001", content="The limit is $6,500.")
        extra_chunk = _make_chunk(chunk_id="chunk-002", content="The catch-up contribution is $1,000.")

        mock_retrieval = AsyncMock()
        mock_retrieval.execute = AsyncMock(
            return_value=_make_retrieval_result(chunks=[extra_chunk])
        )

        # Round 0: answer has novel $1,000. Round 1: no novel tokens → stop.
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[
            # Round 0: $1,000 is novel (not in chunk-001)
            _make_generation_result(answer="The limit is $6,500 but catch-up is $1,000."),
            # Round 1: both tokens now in chunk pool (chunk-002 has $1,000) → no novel → stop
            _make_generation_result(answer="The IRA limit is $6,500 and catch-up is $1,000."),
        ])

        mock_pb = MagicMock()
        mock_pb.build = MagicMock(return_value=_make_prompt())

        gen = FLAREGenerator(
            llm_service=mock_llm,
            retrieval_executor=mock_retrieval,
            prompt_builder=mock_pb,
            max_retrieval_rounds=3,
        )

        result = asyncio.run(gen.generate(
            query="What is the IRA limit?",
            initial_chunks=[initial_chunk],
        ))

        # execute() must be called at least once for re-retrieval
        assert mock_retrieval.execute.call_count >= 1
        assert result.retrieval_rounds >= 2

    def test_flare_single_iteration_when_no_novel_tokens(self):
        """When LLM answer's $-tokens are all present in chunks, loop exits after round 0."""
        from generation.advanced_generation import FLAREGenerator

        initial_chunk = _make_chunk(chunk_id="chunk-001", content="The limit is $6,500.")

        mock_retrieval = AsyncMock()
        mock_retrieval.execute = AsyncMock(return_value=_make_retrieval_result())

        mock_llm = AsyncMock()
        # Answer only mentions $6,500 — already in the chunk
        mock_llm.generate = AsyncMock(
            return_value=_make_generation_result(answer="The IRA limit is $6,500 for 2023.")
        )

        mock_pb = MagicMock()
        mock_pb.build = MagicMock(return_value=_make_prompt())

        gen = FLAREGenerator(
            llm_service=mock_llm,
            retrieval_executor=mock_retrieval,
            prompt_builder=mock_pb,
            max_retrieval_rounds=3,
        )

        result = asyncio.run(gen.generate(
            query="What is the IRA limit?",
            initial_chunks=[initial_chunk],
        ))

        mock_retrieval.execute.assert_not_called()
        assert result.retrieval_rounds == 1
        assert result.novel_tokens_per_round == [[]]

    def test_flare_max_iterations_respected(self):
        """Loop terminates at max_retrieval_rounds even when novel tokens keep appearing."""
        from generation.advanced_generation import FLAREGenerator

        initial_chunk = _make_chunk(chunk_id="chunk-001", content="Some info.")

        # Each re-retrieval returns a distinct chunk
        extra_chunks = [
            _make_chunk(chunk_id=f"extra-{i}", content=f"Extra content {i}.")
            for i in range(5)
        ]

        call_count = [0]
        def make_extra_retrieval(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return _make_retrieval_result(chunks=[extra_chunks[idx % len(extra_chunks)]])

        mock_retrieval = AsyncMock()
        mock_retrieval.execute = AsyncMock(side_effect=make_extra_retrieval)

        # Every LLM round emits a fresh novel $-token not in any chunk
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[
            _make_generation_result(answer=f"Answer with $1,{i:03d}.") for i in range(10)
        ])

        mock_pb = MagicMock()
        mock_pb.build = MagicMock(return_value=_make_prompt())

        max_rounds = 2
        gen = FLAREGenerator(
            llm_service=mock_llm,
            retrieval_executor=mock_retrieval,
            prompt_builder=mock_pb,
            max_retrieval_rounds=max_rounds,
        )

        result = asyncio.run(gen.generate(
            query="Tell me about IRA limits.",
            initial_chunks=[initial_chunk],
        ))

        # With max_retrieval_rounds=2: round 0, re-retrieve, round 1, re-retrieve, round 2 → stop
        assert mock_retrieval.execute.call_count == max_rounds
        assert result.retrieval_rounds == max_rounds + 1

    def test_flare_re_retrieval_query_includes_novel_tokens(self):
        """The re-retrieval query must contain both the original query and the novel $-token."""
        from generation.advanced_generation import FLAREGenerator
        from retrieval.router.query_router import RoutingDecision

        initial_chunk = _make_chunk(chunk_id="chunk-001", content="The limit is $6,500.")

        captured_decisions: list[RoutingDecision] = []

        async def capture_execute(decision, top_k=5):
            captured_decisions.append(decision)
            return _make_retrieval_result(chunks=[
                _make_chunk(chunk_id="chunk-002", content="Catch-up is $1,000.")
            ])

        mock_retrieval = AsyncMock()
        mock_retrieval.execute = AsyncMock(side_effect=capture_execute)

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[
            # Round 0: novel $1,000 not in chunk
            _make_generation_result(answer="Limit is $6,500 and catch-up is $1,000."),
            # Round 1: now $1,000 is in the new chunk → no novel tokens → stop
            _make_generation_result(answer="Limit is $6,500 and catch-up is $1,000."),
        ])

        mock_pb = MagicMock()
        mock_pb.build = MagicMock(return_value=_make_prompt())

        original_query = "What is the IRA catch-up provision?"
        gen = FLAREGenerator(
            llm_service=mock_llm,
            retrieval_executor=mock_retrieval,
            prompt_builder=mock_pb,
            max_retrieval_rounds=3,
        )

        asyncio.run(gen.generate(
            query=original_query,
            initial_chunks=[initial_chunk],
        ))

        assert len(captured_decisions) >= 1
        re_query = captured_decisions[0].expanded_queries[0]
        assert original_query in re_query
        assert "$1,000" in re_query

    def test_flare_merges_and_dedupes_chunks(self):
        """Re-retrieved chunks are merged into all_chunks with deduplication by chunk_id."""
        from generation.advanced_generation import FLAREGenerator

        chunk_a = _make_chunk(chunk_id="A", content="Limit is $6,500.")
        chunk_b = _make_chunk(chunk_id="B", content="Another source $6,500.")
        chunk_c = _make_chunk(chunk_id="C", content="More context $1,000.")

        # Re-retrieval returns B (duplicate) and C (new)
        mock_retrieval = AsyncMock()
        mock_retrieval.execute = AsyncMock(
            return_value=_make_retrieval_result(chunks=[chunk_b, chunk_c])
        )

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[
            # Round 0: novel $1,000
            _make_generation_result(answer="Limit is $6,500, catch-up is $1,000."),
            # Round 1: no novel tokens
            _make_generation_result(answer="Limit is $6,500 and $1,000 catch-up."),
        ])

        mock_pb = MagicMock()
        mock_pb.build = MagicMock(return_value=_make_prompt())

        gen = FLAREGenerator(
            llm_service=mock_llm,
            retrieval_executor=mock_retrieval,
            prompt_builder=mock_pb,
            max_retrieval_rounds=3,
        )

        result = asyncio.run(gen.generate(
            query="What is the IRA catch-up?",
            initial_chunks=[chunk_a, chunk_b],  # A and B initially
        ))

        result_ids = {str(c.chunk.chunk_id) for c in result.all_chunks}
        # B must not be duplicated; C must be added
        assert result_ids == {"A", "B", "C"}
        assert len(result.all_chunks) == 3

    def test_flare_exits_when_reretrieval_adds_no_new_chunks(self):
        """Loop exits immediately when re-retrieval returns only already-seen chunk_ids."""
        from generation.advanced_generation import FLAREGenerator

        # Initial chunk has $6,500; retrieval will return the same chunk (no new ones)
        initial_chunk = _make_chunk(chunk_id="seen-001", content="The limit is $6,500.")

        mock_retrieval = AsyncMock()
        # Re-retrieval always returns the same chunk already in the pool
        mock_retrieval.execute = AsyncMock(
            return_value=_make_retrieval_result(chunks=[initial_chunk])
        )

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[
            # Round 0: $1,000 is novel → triggers re-retrieval
            _make_generation_result(answer="The limit is $6,500 and catch-up is $1,000."),
            # Round 1 (after re-retrieval adds nothing → loop should have broken, never called)
            _make_generation_result(answer="Should not reach this."),
        ])

        mock_pb = MagicMock()
        mock_pb.build = MagicMock(return_value=_make_prompt())

        gen = FLAREGenerator(
            llm_service=mock_llm,
            retrieval_executor=mock_retrieval,
            prompt_builder=mock_pb,
            max_retrieval_rounds=3,
        )

        result = asyncio.run(gen.generate(
            query="What is the IRA limit?",
            initial_chunks=[initial_chunk],
        ))

        # execute() called exactly once (the re-retrieval attempt)
        mock_retrieval.execute.assert_called_once()
        # LLM called exactly once (round 0 only; loop broke before round 1 generation)
        assert mock_llm.generate.call_count == 1
        assert result.answer == "The limit is $6,500 and catch-up is $1,000."
