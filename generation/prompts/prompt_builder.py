"""
Prompt construction layer.

Responsibilities:
  - Select the right prompt template per query type
  - Pack retrieved chunks into the context window efficiently
  - Format citations for attribution
  - Handle token budget (avoid overflowing the context window)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from utils.models import QueryType, RetrievedChunk, Citation

logger = logging.getLogger(__name__)

# Token budget: reserve this many tokens for the answer
ANSWER_BUFFER_TOKENS = 1500
# Rough char-to-token ratio (English text)
CHARS_PER_TOKEN = 4


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_BASE = """You are a precise document intelligence assistant. Answer questions \
based ONLY on the provided context. If the answer is not in the context, say \
"I could not find this in the provided documents." Never speculate beyond the sources.

When you use information from a source, cite it using [Source N] notation where N \
is the source number provided. Always include citations."""

TEMPLATES: dict[str, str] = {
    "factual": """{system}

Context sources:
{context}

Question: {query}

Provide a concise, factual answer with citations. Be direct.""",

    "semantic": """{system}

Context sources:
{context}

Question: {query}

Provide a clear, well-structured answer. Explain the concept thoroughly using \
only the provided sources. Include citations throughout.""",

    "analytical": """{system}

Context sources:
{context}

Question: {query}

Provide a structured analytical response. Organize your answer with key points, \
supporting evidence from sources, and a brief conclusion. Cite all claims.""",

    "multi_hop": """{system}

Context sources:
{context}

Question: {query}

This question may require connecting information across multiple sources. \
Trace the logical chain step by step, citing each source used.""",

    "comparative": """{system}

Context sources:
{context}

Question: {query}

Compare and contrast systematically. Use a structured format if helpful. \
Cite the source for each claim.""",

    "default": """{system}

Context sources:
{context}

Question: {query}

Answer based on the provided sources with citations.""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Constructed prompt output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstructedPrompt:
    messages: list[dict]
    citations: list[Citation]
    chunks_used: int
    estimated_tokens: int


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    def __init__(self, max_context_tokens: int = 6000):
        self.max_context_tokens = max_context_tokens

    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_type: QueryType,
        system_extra: Optional[str] = None,
    ) -> ConstructedPrompt:
        """
        Build the final prompt dict for the LLM.
        Packs as many chunks as fit within the token budget.
        """
        # Pack context within token budget
        context_parts, citations, token_count = self._pack_context(chunks)

        context_str = "\n\n".join(context_parts)

        # Select template
        template_key = query_type.value if query_type.value in TEMPLATES else "default"
        template = TEMPLATES[template_key]

        system = SYSTEM_BASE
        if system_extra:
            system += f"\n\n{system_extra}"

        user_message = template.format(
            system=system,
            context=context_str,
            query=query,
        )

        # For OpenAI-style chat completions
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._build_user_message(context_str, query, template_key)},
        ]

        estimated_tokens = len(user_message) // CHARS_PER_TOKEN

        return ConstructedPrompt(
            messages=messages,
            citations=citations,
            chunks_used=len(context_parts),
            estimated_tokens=estimated_tokens,
        )

    def _pack_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[str], list[Citation], int]:
        """
        Greedily pack chunks into context window.
        Each chunk gets a [Source N] label used for citation.
        """
        context_parts: list[str] = []
        citations: list[Citation] = []
        token_budget = self.max_context_tokens - ANSWER_BUFFER_TOKENS
        used_tokens = 0

        for i, rc in enumerate(chunks):
            # For sentence-window chunks, use the full window text for context
            content = rc.chunk.metadata.get("window_text") or rc.chunk.content
            chunk_tokens = len(content) // CHARS_PER_TOKEN

            if used_tokens + chunk_tokens > token_budget:
                logger.debug("Context window full at chunk %d (%d tokens used)", i, used_tokens)
                break

            source_label = f"[Source {i+1}]"
            title = rc.chunk.metadata.get("title", "Unknown")
            source = rc.chunk.metadata.get("source", "Unknown")
            doc_type = rc.chunk.metadata.get("doc_type", "")

            header = f"{source_label} {title} ({doc_type})"
            part = f"{header}\n{content}"
            context_parts.append(part)
            used_tokens += chunk_tokens

            # Build citation record
            excerpt = content[:200].replace("\n", " ") + ("..." if len(content) > 200 else "")
            citations.append(Citation(
                chunk_id=str(rc.chunk.chunk_id),
                doc_id=str(rc.chunk.doc_id),
                source=source,
                title=title,
                excerpt=excerpt,
                score=rc.score,
            ))

        return context_parts, citations, used_tokens

    def _build_user_message(self, context: str, query: str, template_key: str) -> str:
        instructions = {
            "factual": "Answer directly and concisely with citations.",
            "semantic": "Explain thoroughly using only the provided sources.",
            "analytical": "Analyze systematically with structured key points.",
            "multi_hop": "Connect information across sources step by step.",
            "comparative": "Compare systematically, one dimension at a time.",
        }
        instruction = instructions.get(template_key, "Answer with citations.")

        return f"""Here are the relevant source documents:

{context}

---

Question: {query}

{instruction}"""
