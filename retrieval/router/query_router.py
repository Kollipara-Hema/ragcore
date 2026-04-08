"""
Query understanding and routing module.

Pipeline:
  1. Classify query type (factual, semantic, multi-hop, lookup, analytical)
  2. Parse any metadata constraints (date ranges, doc types, authors)
  3. Decide which retrieval strategy fits best
  4. Optionally expand query into sub-queries
  5. Provide fallback strategy if primary fails
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from utils.models import QueryType, RetrievalStrategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing decision output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    query_type: QueryType
    primary_strategy: RetrievalStrategy
    fallback_strategy: RetrievalStrategy
    metadata_filter: Optional[dict] = None
    expanded_queries: list[str] = field(default_factory=list)
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based heuristics (fast, zero-cost, runs first)
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicRouter:
    """
    Fast regex/keyword-based pre-classifier.
    Catches obvious patterns before invoking LLM classification.
    """

    LOOKUP_PATTERNS = [
        r"\b(what is|define|definition of|meaning of)\b",
        r"\b(when was|what date|what year|published in)\b",
        r"\b(who is|who wrote|who created|author of)\b",
    ]

    ANALYTICAL_PATTERNS = [
        r"\b(compare|contrast|difference between|similarities)\b",
        r"\b(summarize|summary|overview|key points)\b",
        r"\b(analyze|analysis|evaluate|assessment)\b",
        r"\b(pros and cons|advantages|disadvantages)\b",
    ]

    MULTI_HOP_PATTERNS = [
        r"\b(and then|which then|that also|and also)\b",
        r"\b(relationship between|how does .* affect|impact of .* on)\b",
        r"\b(first .* then|step by step)\b",
    ]

    METADATA_PATTERNS = {
        "doc_type": r"\b(in (the )?pdf|in (the )?document|in (the )?report)\b",
        "date_range": r"\b(from \d{4}|between \d{4}|since \d{4}|after \d{4})\b",
        "author": r"\b(by [A-Z][a-z]+ [A-Z][a-z]+|written by|authored by)\b",
    }

    def classify(self, query: str) -> Optional[QueryType]:
        q = query.lower()
        for pat in self.MULTI_HOP_PATTERNS:
            if re.search(pat, q):
                return QueryType.MULTI_HOP
        for pat in self.ANALYTICAL_PATTERNS:
            if re.search(pat, q):
                return QueryType.ANALYTICAL
        for pat in self.LOOKUP_PATTERNS:
            if re.search(pat, q):
                return QueryType.LOOKUP
        return None  # Uncertain — defer to LLM

    def extract_metadata_hints(self, query: str) -> dict:
        hints = {}
        for field_name, pattern in self.METADATA_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                hints[field_name] = True  # signal that this filter type may be relevant
        return hints


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based classifier (used when heuristics are uncertain)
# ─────────────────────────────────────────────────────────────────────────────

class LLMQueryClassifier:
    """
    Uses a small/fast LLM call to classify ambiguous queries.
    Results are cached to avoid repeat classification of similar queries.
    """

    SYSTEM_PROMPT = """You are a query classifier for a document retrieval system.
Classify the user query into exactly one of these types:
- factual: direct factual question with a specific answer
- semantic: conceptual or explanatory question
- multi_hop: requires combining information from multiple sources
- analytical: requires synthesis, comparison, or analysis
- lookup: exact lookup by ID, name, date, or keyword
- comparative: comparing two or more things

Also extract any metadata filters (author, doc_type, date_range, tags).

Respond ONLY with valid JSON, no explanation:
{
  "query_type": "<type>",
  "metadata_filter": {},
  "confidence": 0.0-1.0
}"""

    async def classify(self, query: str) -> tuple[QueryType, dict, float]:
        try:
            from openai import AsyncOpenAI
            import json

            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # cheap + fast for classification
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            qt = QueryType(result.get("query_type", "semantic"))
            meta_filter = result.get("metadata_filter", {})
            confidence = result.get("confidence", 0.8)
            return qt, meta_filter, confidence

        except Exception as e:
            logger.warning("LLM classification failed: %s — defaulting to semantic", e)
            return QueryType.SEMANTIC, {}, 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Query expander (multi-query strategy)
# ─────────────────────────────────────────────────────────────────────────────

class QueryExpander:
    """
    Generates N paraphrased variants of the original query.
    Improves recall for vague or ambiguous queries.
    """

    async def expand(self, query: str, n: int = None) -> list[str]:
        n = n or settings.max_expanded_queries
        if not settings.enable_query_expansion:
            return [query]

        try:
            from openai import AsyncOpenAI
            import json

            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Generate {n} semantically diverse paraphrases of the query to improve document retrieval. "
                            "Vary vocabulary and phrasing. Return a JSON array of strings only."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            # Handle both {"queries": [...]} and bare array
            if isinstance(data, list):
                variants = data
            else:
                variants = data.get("queries", data.get("paraphrases", []))

            all_queries = [query] + [v for v in variants if v != query]
            return all_queries[:n + 1]

        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            return [query]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy selector
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_MAP: dict[QueryType, tuple[RetrievalStrategy, RetrievalStrategy]] = {
    # (primary, fallback)
    QueryType.FACTUAL:      (RetrievalStrategy.HYBRID,        RetrievalStrategy.SEMANTIC),
    QueryType.LOOKUP:       (RetrievalStrategy.KEYWORD,       RetrievalStrategy.SEMANTIC),
    QueryType.SEMANTIC:     (RetrievalStrategy.SEMANTIC,      RetrievalStrategy.HYBRID),
    QueryType.MULTI_HOP:    (RetrievalStrategy.MULTI_QUERY,   RetrievalStrategy.HYBRID),
    QueryType.ANALYTICAL:   (RetrievalStrategy.HYBRID,        RetrievalStrategy.SEMANTIC),
    QueryType.COMPARATIVE:  (RetrievalStrategy.MULTI_QUERY,   RetrievalStrategy.HYBRID),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main router
# ─────────────────────────────────────────────────────────────────────────────

class QueryRouter:
    def __init__(self):
        self._heuristic = HeuristicRouter()
        self._llm_classifier = LLMQueryClassifier()
        self._expander = QueryExpander()

    async def route(
        self,
        query: str,
        strategy_override: Optional[RetrievalStrategy] = None,
        metadata_filter: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Main entry point. Returns a RoutingDecision describing:
          - query type
          - retrieval strategy (or override if specified)
          - expanded queries
          - metadata filters
        """
        # Step 1: Fast heuristic pass
        heuristic_type = self._heuristic.classify(query)
        meta_hints = self._heuristic.extract_metadata_hints(query)

        # Step 2: LLM classification if heuristics were uncertain
        if heuristic_type is None:
            query_type, llm_meta, confidence = await self._llm_classifier.classify(query)
        else:
            query_type, llm_meta, confidence = heuristic_type, {}, 0.9

        # Merge metadata filters
        combined_filter = {}
        if meta_hints:
            combined_filter.update(meta_hints)
        if llm_meta:
            combined_filter.update({k: v for k, v in llm_meta.items() if v})
        if metadata_filter:
            combined_filter.update(metadata_filter)

        # Step 3: Select strategy
        if strategy_override:
            primary = strategy_override
            _, fallback = STRATEGY_MAP.get(query_type, (RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC))
        else:
            primary, fallback = STRATEGY_MAP.get(
                query_type,
                (RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC),
            )

        # Step 4: Query expansion (for multi-hop and analytical)
        expanded = [query]
        if primary == RetrievalStrategy.MULTI_QUERY or query_type in {QueryType.MULTI_HOP, QueryType.ANALYTICAL}:
            expanded = await self._expander.expand(query)

        reasoning = (
            f"Query classified as '{query_type.value}' (confidence={confidence:.2f}). "
            f"Strategy: {primary.value}. "
            f"Expanded to {len(expanded)} queries."
        )
        logger.info(reasoning)

        return RoutingDecision(
            query_type=query_type,
            primary_strategy=primary,
            fallback_strategy=fallback,
            metadata_filter=combined_filter if combined_filter else None,
            expanded_queries=expanded,
            reasoning=reasoning,
        )
