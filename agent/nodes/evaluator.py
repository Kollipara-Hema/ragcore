"""
Evaluator node — scores the generated answer and decides whether to retry.

Scoring heuristics (no LLM call required):
  - Empty or very short answer → low score
  - Answer references "could not find" → low score
  - At least one citation → higher score
  - Answer length proportional to query complexity → bonus

If evaluation is enabled, uses RAGAS for LLM-based scoring.
"""
from __future__ import annotations

import logging

from agent.state import AgentState
from config.settings import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_THRESHOLD = 0.6   # retry if confidence below this


async def _calculate_confidence(state: AgentState) -> tuple[float, str]:
    if settings.enable_evaluation and settings.eval_strategy == "ragas":
        return await _ragas_confidence(state)
    else:
        return _heuristic_confidence(state)


def _heuristic_confidence(state: AgentState) -> tuple[float, str]:
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    query_type = state.get("query_type", "semantic")

    if not answer or len(answer) < 30:
        return 0.1, "Answer is empty or too short"

    if "could not find" in answer.lower() or "no relevant" in answer.lower():
        return 0.2, "Answer indicates no information found"

    score = 0.5

    if citations:
        score += 0.2

    if len(answer) > 100:
        score += 0.1

    if query_type in ("analytical", "comparative", "multi_hop") and len(answer) > 300:
        score += 0.1

    if len(citations) >= 2:
        score += 0.1

    return min(score, 1.0), "Answer meets quality threshold"


async def _ragas_confidence(state: AgentState) -> tuple[float, str]:
    """Use RAGAS for LLM-based evaluation of single sample."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        query = state.get("query", "")
        answer = state.get("answer", "")
        contexts = [c.get("excerpt", "") for c in state.get("citations", [])]

        if not answer or not contexts:
            return 0.1, "Missing answer or contexts for RAGAS evaluation"

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

        faithfulness_score = result["faithfulness"]
        relevance_score = result["answer_relevancy"]

        # Combine scores: average of faithfulness and relevance
        confidence = (faithfulness_score + relevance_score) / 2.0

        return confidence, f"RAGAS: faithfulness={faithfulness_score:.2f}, relevance={relevance_score:.2f}"

    except ImportError:
        logger.warning("RAGAS not installed; falling back to heuristics.")
        return _heuristic_confidence(state)
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return _heuristic_confidence(state)


async def evaluator_node(state: AgentState) -> dict:
    """Score the answer and flag for retry if quality is insufficient."""
    retry_count = state.get("retry_count", 0)
    confidence, reasoning = await _calculate_confidence(state)

    needs_retry = confidence < RETRY_THRESHOLD and retry_count < MAX_RETRIES
    if needs_retry:
        logger.info(
            "evaluator: confidence=%.2f below threshold — scheduling retry %d/%d",
            confidence, retry_count + 1, MAX_RETRIES,
        )
    else:
        logger.info("evaluator: confidence=%.2f — answer accepted", confidence)

    return {
        "confidence": confidence,
        "eval_reasoning": reasoning,
        "needs_retry": needs_retry,
        "retry_count": retry_count + (1 if needs_retry else 0),
    }


def should_retry(state: AgentState) -> str:
    """Conditional edge: route back to retriever if retry needed, else finish."""
    return "retriever" if state.get("needs_retry") else "end"
