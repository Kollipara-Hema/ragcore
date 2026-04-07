"""
Evaluator node — scores the generated answer and decides whether to retry.

Scoring heuristics (no LLM call required):
  - Empty or very short answer → low score
  - Answer references "could not find" → low score
  - At least one citation → higher score
  - Answer length proportional to query complexity → bonus
"""
from __future__ import annotations

import logging

from agent.state import AgentState
from config.settings import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_THRESHOLD = 0.6   # retry if confidence below this


def _calculate_confidence(state: AgentState) -> tuple[float, str]:
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


async def evaluator_node(state: AgentState) -> dict:
    """Score the answer and flag for retry if quality is insufficient."""
    retry_count = state.get("retry_count", 0)
    confidence, reasoning = _calculate_confidence(state)

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
