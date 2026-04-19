"""
RAG evaluation metrics.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Evaluator:
    """Deprecated placeholder. Use RAGEvaluator or RetrievalEvaluator directly."""

    def evaluate(self, csv_path: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "Evaluator.evaluate() was a placeholder with hardcoded values. "
            "Use RAGEvaluator.run_against_orchestrator() to evaluate against a live system, "
            "or RetrievalEvaluator / GenerationEvaluator for component-level metrics."
        )


@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0        # is the answer supported by retrieved context?
    answer_relevance: float = 0.0    # does the answer address the query?
    hallucination_rate: float = 0.0  # fraction of claims not in context
    sample_count: int = 0


@dataclass
class LatencyMetrics:
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0


@dataclass
class CostMetrics:
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_per_query_usd: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class RetrievalMetrics:
    hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    sample_count: int = 0


@dataclass
class EvalSample:
    query: str
    ground_truth: str
    retrieved_doc_ids: List[str] = field(default_factory=list)
    relevant_doc_ids: List[str] = field(default_factory=list)
    retrieved_contexts: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None
    latency_ms: Optional[float] = None
    total_tokens: Optional[int] = None


@dataclass
class EvalReport:
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    timestamp: str = ""

    def summary(self) -> str:
        return (
            f"=== RAG Evaluation Report ===\n"
            f"Retrieval  — Hit Rate: {self.retrieval.hit_rate:.3f} | "
            f"MRR: {self.retrieval.mrr:.3f} | "
            f"NDCG@5: {self.retrieval.ndcg_at_5:.3f} | "
            f"Precision: {self.retrieval.context_precision:.3f} | "
            f"Recall: {self.retrieval.context_recall:.3f}\n"
            f"Generation — Faithfulness: {self.generation.faithfulness:.3f} | "
            f"Relevance: {self.generation.answer_relevance:.3f} | "
            f"Hallucination: {self.generation.hallucination_rate:.3f}\n"
            f"Latency    — P50: {self.latency.p50_ms:.0f}ms | "
            f"P95: {self.latency.p95_ms:.0f}ms | "
            f"P99: {self.latency.p99_ms:.0f}ms\n"
            f"Cost       — $/query: ${self.cost.cost_per_query_usd:.5f} | "
            f"Cache hit: {self.cost.cache_hit_rate:.1%}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    def evaluate(self, samples: list[EvalSample]) -> RetrievalMetrics:
        if not samples:
            return RetrievalMetrics()

        hit_count = 0
        reciprocal_ranks: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        ndcg_scores: list[float] = []

        for sample in samples:
            if not sample.retrieved_doc_ids or not sample.relevant_doc_ids:
                continue

            relevant_set = set(sample.relevant_doc_ids)

            # FiQA ground-truth labels are at the document level, but retrieval
            # operates at the chunk level. Multiple chunks from the same document
            # share a doc_id after UUID translation. Deduping by first-occurrence
            # rank aligns metric granularity with label granularity.
            seen: set[str] = set()
            retrieved: list[str] = []
            for d in sample.retrieved_doc_ids:
                if d not in seen:
                    seen.add(d)
                    retrieved.append(d)

            # Hit rate
            if any(d in relevant_set for d in retrieved):
                hit_count += 1

            # MRR
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # Precision@K
            hits = sum(1 for d in retrieved if d in relevant_set)
            precisions.append(hits / len(retrieved) if retrieved else 0.0)

            # Recall
            recalls.append(hits / len(relevant_set) if relevant_set else 0.0)

            # NDCG@5
            ndcg_scores.append(self._ndcg_at_k(retrieved[:5], relevant_set))

        n = len(samples)
        return RetrievalMetrics(
            hit_rate=hit_count / n,
            mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
            ndcg_at_5=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
            context_precision=sum(precisions) / len(precisions) if precisions else 0.0,
            context_recall=sum(recalls) / len(recalls) if recalls else 0.0,
            sample_count=n,
        )

    def _ndcg_at_k(self, retrieved: list[str], relevant: set[str], k: int = 5) -> float:
        import math
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, doc_id in enumerate(retrieved[:k])
            if doc_id in relevant
        )
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / ideal if ideal > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Generation evaluator (RAGAS-based)
# ─────────────────────────────────────────────────────────────────────────────

class GenerationEvaluator:
    """
    Uses RAGAS library for automated LLM-based evaluation.
    Falls back to simple heuristics if RAGAS is not installed.
    """

    async def evaluate(self, samples: list[EvalSample]) -> GenerationMetrics:
        try:
            return await self._ragas_evaluate(samples)
        except ImportError:
            logger.warning("RAGAS not installed; using heuristic evaluation.")
            return await self._heuristic_evaluate(samples)

    async def _ragas_evaluate(self, samples: list[EvalSample]) -> GenerationMetrics:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        data = {
            "question": [s.query for s in samples],
            "answer": [s.generated_answer or "" for s in samples],
            "contexts": [s.retrieved_contexts or [] for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

        return GenerationMetrics(
            faithfulness=result["faithfulness"],
            answer_relevance=result["answer_relevancy"],
            hallucination_rate=1.0 - result["faithfulness"],
            sample_count=len(samples),
        )

    async def _heuristic_evaluate(self, samples: list[EvalSample]) -> GenerationMetrics:
        """
        Simple heuristic: check if key terms from retrieved contexts
        appear in the generated answer.
        """
        faithfulness_scores: list[float] = []

        for sample in samples:
            if not sample.generated_answer or not sample.retrieved_contexts:
                continue
            answer_lower = sample.generated_answer.lower()
            context_words = set()
            for ctx in sample.retrieved_contexts:
                words = set(ctx.lower().split())
                context_words |= words

            answer_words = set(answer_lower.split())
            # Rough proxy: what fraction of answer words appear in context
            overlap = len(answer_words & context_words)
            score = min(1.0, overlap / max(len(answer_words), 1))
            faithfulness_scores.append(score)

        avg = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        return GenerationMetrics(
            faithfulness=avg,
            answer_relevance=avg,  # proxy
            hallucination_rate=1.0 - avg,
            sample_count=len(samples),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Latency evaluator
# ─────────────────────────────────────────────────────────────────────────────

class LatencyEvaluator:
    def evaluate(self, samples: list[EvalSample]) -> LatencyMetrics:
        latencies = sorted(
            s.latency_ms for s in samples if s.latency_ms is not None
        )
        if not latencies:
            return LatencyMetrics()

        n = len(latencies)
        return LatencyMetrics(
            p50_ms=latencies[int(n * 0.50)],
            p95_ms=latencies[int(n * 0.95)],
            p99_ms=latencies[int(n * 0.99)],
            mean_ms=sum(latencies) / n,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cost evaluator
# ─────────────────────────────────────────────────────────────────────────────

class CostEvaluator:
    # Approximate costs per 1M tokens (update these regularly)
    COST_PER_M_TOKENS = {
        "gpt-4o": 10.0,
        "gpt-4o-mini": 0.30,
        "claude-sonnet-4-6": 9.0,
        "text-embedding-3-large": 0.13,
    }

    def evaluate(self, samples: list[EvalSample], model: str = "gpt-4o") -> CostMetrics:
        total_tokens = sum(s.total_tokens or 0 for s in samples)
        cost_per_m = self.COST_PER_M_TOKENS.get(model, 10.0)
        total_cost = (total_tokens / 1_000_000) * cost_per_m
        n = len(samples)
        return CostMetrics(
            total_tokens=total_tokens,
            estimated_cost_usd=total_cost,
            cost_per_query_usd=total_cost / n if n > 0 else 0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluator:
    def __init__(self):
        self._retrieval = RetrievalEvaluator()
        self._generation = GenerationEvaluator()
        self._latency = LatencyEvaluator()
        self._cost = CostEvaluator()

    async def run_evaluation(
        self,
        samples: list[EvalSample],
        model: str = "gpt-4o",
    ) -> EvalReport:
        from datetime import datetime, timezone

        retrieval_metrics = self._retrieval.evaluate(samples)
        generation_metrics = await self._generation.evaluate(samples)
        latency_metrics = self._latency.evaluate(samples)
        cost_metrics = self._cost.evaluate(samples, model)

        report = EvalReport(
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            latency=latency_metrics,
            cost=cost_metrics,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("\n%s", report.summary())
        return report

    async def run_against_orchestrator(
        self,
        golden_dataset: list[dict],
        orchestrator,
    ) -> EvalReport:
        """
        Run evaluation against a live orchestrator.
        golden_dataset: list of {query, ground_truth, relevant_doc_ids}
        """
        from utils.models import QueryRequest

        samples: list[EvalSample] = []
        for item in golden_dataset:
            request = QueryRequest(query=item["query"])
            start = time.monotonic()
            try:
                response = await orchestrator.query(request)
                latency = (time.monotonic() - start) * 1000
                samples.append(EvalSample(
                    query=item["query"],
                    ground_truth=item.get("ground_truth", ""),
                    relevant_doc_ids=item.get("relevant_doc_ids", []),
                    generated_answer=response.answer,
                    retrieved_doc_ids=[c.get("doc_id") for c in response.citations],
                    retrieved_contexts=[c.get("excerpt") for c in response.citations],
                    latency_ms=latency,
                    total_tokens=response.total_tokens,
                ))
            except Exception as e:
                logger.error("Eval query failed: %s", e)

        return await self.run_evaluation(samples)
