"""
=============================================================================
generation/advanced_generation.py  —  Advanced Generation
=============================================================================
WHAT THIS FILE ADDS:
    Three advanced generation strategies that go beyond basic RAG:

    1. SelfRAG
       - After generating an answer, checks its own claims against sources
       - If a claim is not supported by retrieved context → retrieves more
       - Reduces hallucination significantly
       - Paper: "Self-RAG: Learning to Retrieve, Generate, and Critique"

    2. FLAREGenerator (Forward-Looking Active Retrieval)
       - Generates the answer token-by-token
       - When it produces a low-confidence statement, pauses and retrieves more
       - Retrieves exactly when needed, not just at the start
       - Better for long answers that require multiple retrieval passes

    3. AgenticRAG
       - Multi-turn system with tool use
       - The LLM decides when to retrieve, what to search for, and when to stop
       - Can handle complex multi-part questions by breaking them down
       - Most powerful but slowest and most expensive

HOW TO USE:
    # Self-RAG (recommended starting point):
    from generation.advanced_generation import SelfRAGGenerator
    generator = SelfRAGGenerator()
    result = await generator.generate(query, initial_chunks, prompt_builder)

    # Agentic (for complex multi-hop questions):
    from generation.advanced_generation import AgenticRAG
    agent = AgenticRAG(retrieval_executor, prompt_builder)
    result = await agent.run("Compare the revenue figures across all three quarters")
=============================================================================
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that LLMs add around JSON output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    return stripped


# =============================================================================
# PHASE 5A: Self-RAG
# =============================================================================

@dataclass
class SelfRAGResult:
    """Result from Self-RAG generation including verification details."""
    answer: str                          # Final verified answer
    original_answer: str                 # Answer before verification
    verified_claims: list[str]           # Claims verified against sources
    unsupported_claims: list[str]        # Claims not found in sources
    additional_retrievals: int           # How many extra retrievals were done
    total_tokens: int = 0
    faithfulness_score: float = 0.0     # Fraction of claims supported


class SelfRAGGenerator:
    """
    Self-RAG: Generate → Critique → Retrieve more if needed → Regenerate.

    STANDARD RAG PROBLEM:
        The LLM sometimes makes claims that are not in the retrieved chunks.
        These are hallucinations — the model "fills in" missing information.

    HOW SELF-RAG FIXES THIS:
        Step 1: Generate an initial answer from retrieved chunks
        Step 2: Extract each claim from the answer
        Step 3: For each claim, check if it appears in the retrieved context
        Step 4: For unsupported claims, retrieve additional chunks
        Step 5: Regenerate the answer with all available evidence
        Step 6: Report which claims are supported vs unsupported

    RESULT:
        - Lower hallucination rate
        - Answer only contains claims backed by sources
        - Unsupported claims are flagged or removed

    COST:
        - 2-3x more LLM calls than basic RAG
        - Worth it for high-stakes applications (legal, medical, financial)
    """

    # Prompt to extract individual claims from an answer
    CLAIM_EXTRACTION_PROMPT = """Extract every factual claim from this answer.
Each claim should be one standalone statement that can be verified.
Return ONLY a JSON array of strings.
Example: ["The company was founded in 2010.", "Revenue was $4.2B in Q3."]"""

    # Prompt to verify if a claim is supported by provided context
    VERIFICATION_PROMPT = """Does the following context support this claim?
Answer ONLY with JSON: {{"supported": true/false, "evidence": "quote from context or empty string"}}

Claim: {claim}

Context:
{context}"""

    def __init__(self, max_additional_retrievals: int = 2):
        """
        Args:
            max_additional_retrievals: How many extra retrieval rounds to do
                                      for unsupported claims (prevents infinite loops)
        """
        self.max_additional_retrievals = max_additional_retrievals

    async def generate(
        self,
        query: str,
        initial_chunks,           # list[RetrievedChunk]
        prompt_builder,           # PromptBuilder instance
        retrieval_executor,       # RetrievalExecutor for additional retrieval
        llm_service,              # GenerationService for LLM calls
        *,
        store_override=None,      # BaseVectorStore; pin re-retrieval to a specific store
    ) -> SelfRAGResult:
        """
        Run the Self-RAG pipeline.

        Args:
            query: User's question
            initial_chunks: Chunks from initial retrieval
            prompt_builder: For building prompts
            retrieval_executor: For additional retrieval if needed
            llm_service: For LLM generation calls

        Returns:
            SelfRAGResult with verified answer and verification details
        """
        from utils.models import QueryType, RetrievalStrategy

        all_chunks = list(initial_chunks)  # Start with initially retrieved chunks

        # ── STEP 1: Generate initial answer ──────────────────────────────────
        logger.info("Self-RAG Step 1: Generating initial answer")

        prompt = prompt_builder.build(
            query=query,
            chunks=all_chunks,
            query_type=QueryType.FACTUAL,
        )
        initial_result = await llm_service.generate(
            query=query,
            prompt=prompt,
            query_type=QueryType.FACTUAL,
            strategy_used=RetrievalStrategy.HYBRID,
        )
        original_answer = initial_result.answer

        # ── STEP 2: Extract individual claims from the answer ─────────────────
        logger.info("Self-RAG Step 2: Extracting claims from answer")
        claims = await self._extract_claims(original_answer, llm_service)
        logger.info("Extracted %d claims", len(claims))

        # ── STEP 3: Verify each claim against retrieved context ───────────────
        logger.info("Self-RAG Step 3: Verifying claims")
        context_text = "\n\n".join(c.chunk.content for c in all_chunks)

        verified_claims = []
        unsupported_claims = []
        additional_retrievals = 0

        for claim in claims:
            is_supported, evidence = await self._verify_claim(claim, context_text, llm_service)

            if is_supported:
                verified_claims.append(claim)
            else:
                unsupported_claims.append(claim)

        # ── STEP 4: Retrieve additional context for unsupported claims ────────
        if unsupported_claims and additional_retrievals < self.max_additional_retrievals:
            logger.info(
                "Self-RAG Step 4: %d unsupported claims — retrieving more context",
                len(unsupported_claims)
            )

            for unsupported_claim in unsupported_claims[:3]:  # Limit to 3 claims
                try:
                    from retrieval.router.query_router import RoutingDecision
                    from utils.models import QueryType, RetrievalStrategy

                    # Retrieve chunks specifically for this unsupported claim
                    decision = RoutingDecision(
                        query_type=QueryType.FACTUAL,
                        primary_strategy=RetrievalStrategy.HYBRID,
                        fallback_strategy=RetrievalStrategy.SEMANTIC,
                        expanded_queries=[unsupported_claim],  # Search for the claim
                    )
                    # Pin re-retrieval to the same store the orchestrator used
                    # for initial retrieval. Without this, a session-scoped
                    # query that opts into Self-RAG would fetch additional
                    # chunks from the public default corpus — a silent
                    # isolation leak. store_override=None preserves today's
                    # behavior for non-session callers.
                    extra_result = await retrieval_executor.execute(
                        decision, top_k=5, store_override=store_override,
                    )
                    all_chunks.extend(extra_result.chunks)
                    additional_retrievals += 1

                except Exception as e:
                    logger.warning("Additional retrieval failed: %s", e)

        # ── STEP 5: Regenerate answer with all available context ──────────────
        if additional_retrievals > 0:
            logger.info(
                "Self-RAG Step 5: Regenerating with %d total chunks",
                len(all_chunks)
            )
            # Deduplicate chunks (some may appear in both initial and additional)
            seen_ids = set()
            unique_chunks = []
            for chunk in all_chunks:
                cid = str(chunk.chunk.chunk_id)
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    unique_chunks.append(chunk)

            # Sort by score and take top chunks
            unique_chunks.sort(key=lambda c: c.score, reverse=True)

            final_prompt = prompt_builder.build(
                query=query,
                chunks=unique_chunks[:10],  # Use top 10 chunks
                query_type=QueryType.ANALYTICAL,
            )

            # Add instruction to only use verified information
            final_prompt.messages[-1]["content"] += (
                "\n\nIMPORTANT: Only include claims that are directly supported "
                "by the provided sources. If a claim cannot be verified from the "
                "sources, omit it or explicitly state it is uncertain."
            )

            final_result = await llm_service.generate(
                query=query,
                prompt=final_prompt,
                query_type=QueryType.ANALYTICAL,
                strategy_used=RetrievalStrategy.HYBRID,
            )
            final_answer = final_result.answer
            total_tokens = initial_result.total_tokens + final_result.total_tokens
        else:
            # No additional retrieval needed — use original answer
            final_answer = original_answer
            total_tokens = initial_result.total_tokens

        # Calculate faithfulness score
        total_claims = len(claims)
        faithfulness = len(verified_claims) / total_claims if total_claims > 0 else 1.0

        return SelfRAGResult(
            answer=final_answer,
            original_answer=original_answer,
            verified_claims=verified_claims,
            unsupported_claims=unsupported_claims,
            additional_retrievals=additional_retrievals,
            total_tokens=total_tokens,
            faithfulness_score=faithfulness,
        )

    async def _extract_claims(self, answer: str, llm_service) -> list[str]:
        """Ask the LLM to break the answer into individual verifiable claims."""
        try:
            from generation.prompts.prompt_builder import ConstructedPrompt
            from utils.models import QueryType, RetrievalStrategy

            prompt = ConstructedPrompt(
                messages=[
                    {"role": "system", "content": self.CLAIM_EXTRACTION_PROMPT},
                    {"role": "user", "content": answer},
                ],
                citations=[],
                chunks_used=0,
                estimated_tokens=0,
            )
            result = await llm_service.generate(
                query=answer,
                prompt=prompt,
                query_type=QueryType.FACTUAL,
                strategy_used=RetrievalStrategy.HYBRID,
            )
            data = json.loads(_strip_markdown_fences(result.answer))
            if isinstance(data, list):
                claims = data
            elif isinstance(data, dict):
                claims = data.get("claims") or data.get("statements") or data.get("items") or []
                if not isinstance(claims, list):
                    claims = []
            else:
                claims = []
            return [c for c in claims if isinstance(c, str) and len(c) > 10]

        except Exception as e:
            logger.warning("Claim extraction failed: %s", e)
            # Fall back to sentence splitting
            return re.split(r'(?<=[.!?])\s+', answer)

    async def _verify_claim(
        self, claim: str, context: str, llm_service
    ) -> tuple[bool, str]:
        """Check if a claim is supported by the context."""
        try:
            from generation.prompts.prompt_builder import ConstructedPrompt
            from utils.models import QueryType, RetrievalStrategy

            prompt_text = self.VERIFICATION_PROMPT.format(
                claim=claim,
                context=context[:3000],   # Limit context to save tokens
            )
            prompt = ConstructedPrompt(
                messages=[{"role": "user", "content": prompt_text}],
                citations=[],
                chunks_used=0,
                estimated_tokens=0,
            )
            result = await llm_service.generate(
                query=claim,
                prompt=prompt,
                query_type=QueryType.FACTUAL,
                strategy_used=RetrievalStrategy.HYBRID,
            )
            data = json.loads(_strip_markdown_fences(result.answer))
            if isinstance(data, dict):
                supported = data.get("supported", False)
                if isinstance(supported, str):
                    supported = supported.lower() in ("true", "yes", "supported", "1")
                return bool(supported), data.get("evidence", "")
            # LLM returned a bare value despite json_object mode
            raw = str(data).lower().strip("'\"")
            return raw in ("true", "yes", "supported", "1"), ""

        except Exception as e:
            logger.warning("Claim verification failed: %s — type=%s", e, type(e).__name__)
            return False, ""  # Fail closed: unparseable response → unsupported


# =============================================================================
# PHASE 5B: FLARE Generator
# =============================================================================

@dataclass
class FLAREResult:
    """Result from FLARE-inspired generation including iterative retrieval details."""
    answer: str
    all_chunks: list          # RetrievedChunk objects from all retrieval rounds
    retrieval_rounds: int
    total_tokens: int
    novel_tokens_per_round: list[list[str]]  # For observability


class FLAREGenerator:
    """
    FLARE-inspired variant. Uses dollar-token novelty as the re-retrieval trigger,
    not the paper's token-logprob mechanism.

    The paper (Jiang et al. 2023) checks per-token confidence via logprobs and
    re-retrieves when a sentence's average log-prob falls below a threshold.
    Groq's Llama 3.3 70B endpoint does not expose per-token logprobs.

    The original [UNCERTAIN: topic] self-signaling approach was abandoned:
    Llama 3.3 70B's RLHF fine-tuning causes it to emit confident, complete
    answers and reliably ignore structural self-interruption instructions.
    Verified empirically over two test rounds against the existing prompt.

    Heuristic: after each generation round, extract $-prefixed numeric tokens
    from the response using r'\\$\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?'. Compare
    against the same tokens present in the current chunk pool. Any $-token in
    the response but absent from all chunks is a potential hallucination —
    re-retrieve using (original_query + novel tokens) as the query, then
    regenerate with the expanded chunk pool.

    Validated on: "How does the 2024 IRA contribution limit compare to 2023,
    and what's the catch-up provision for people over 50?" — the hallucinated
    $1,000 catch-up figure correctly fires the heuristic while grounded
    $6,500 and $7,000 figures do not.
    """

    def __init__(
        self,
        llm_service,
        retrieval_executor,
        prompt_builder,
        max_retrieval_rounds: int = 3,
    ):
        self._llm = llm_service
        self._retrieval = retrieval_executor
        self._prompt_builder = prompt_builder
        self.max_retrieval_rounds = max_retrieval_rounds

    @staticmethod
    def _extract_dollar_tokens(text: str) -> set[str]:
        tokens: set[str] = set()
        # Standard comma-grouped form: $18,000 or $1,234.56 or $100
        # Skip matches followed immediately by k/K (handled by abbrev path below)
        for m in re.finditer(r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?', text):
            if m.end() < len(text) and text[m.end()].lower() == 'k':
                continue
            tokens.add(m.group())
        # Abbreviated thousands: $18k, $54K → normalized to canonical $18,000 / $54,000
        # so chunk-side $18k and response-side $18,000 compare as equal
        for m in re.finditer(r'\$(\d+)[kK]\b', text):
            tokens.add(f"${int(m.group(1)) * 1000:,}")
        return tokens

    async def generate(
        self,
        query: str,
        initial_chunks,
        *,
        store_override=None,
    ) -> FLAREResult:
        """
        Generate an answer with iterative retrieval triggered by dollar-token novelty.

        store_override pins iterative re-retrieval to a specific store. Without
        it, a session-scoped query under generation_strategy='flare' would
        pull additional chunks from the public default corpus mid-generation,
        leaking past the session-isolation boundary established by the API
        handler. None preserves today's behavior for non-session callers.
        """
        from utils.models import QueryType, RetrievalStrategy
        from retrieval.router.query_router import RoutingDecision

        all_chunks = list(initial_chunks)
        novel_per_round: list[list[str]] = []
        total_tokens = 0
        answer = ""

        for round_idx in range(self.max_retrieval_rounds + 1):
            logger.info("FLARE round %d/%d", round_idx + 1, self.max_retrieval_rounds + 1)

            prompt = self._prompt_builder.build(
                query=query,
                chunks=all_chunks,
                query_type=QueryType.ANALYTICAL,
            )
            result = await self._llm.generate(
                query=query,
                prompt=prompt,
                query_type=QueryType.ANALYTICAL,
                strategy_used=RetrievalStrategy.HYBRID,
            )
            total_tokens += result.total_tokens
            answer = result.answer

            answer_dollars = self._extract_dollar_tokens(answer)
            chunk_dollars = self._extract_dollar_tokens(
                " ".join(c.chunk.content for c in all_chunks)
            )
            novel = answer_dollars - chunk_dollars
            novel_per_round.append(sorted(novel))

            if not novel or round_idx == self.max_retrieval_rounds:
                break

            re_query = f"{query} {' '.join(sorted(novel))}"
            logger.info("FLARE: novel dollar tokens %s — re-querying", sorted(novel))

            decision = RoutingDecision(
                query_type=QueryType.FACTUAL,
                primary_strategy=RetrievalStrategy.HYBRID,
                fallback_strategy=RetrievalStrategy.SEMANTIC,
                expanded_queries=[re_query],
            )
            extra = await self._retrieval.execute(
                decision, top_k=5, store_override=store_override,
            )

            existing_ids = {str(c.chunk.chunk_id) for c in all_chunks}
            new_chunks = [
                c for c in extra.chunks
                if str(c.chunk.chunk_id) not in existing_ids
            ]
            if not new_chunks:
                logger.info("FLARE: retriever exhausted — no new chunks, stopping")
                break
            all_chunks.extend(new_chunks)
            logger.info("FLARE: added %d new chunks", len(new_chunks))

        return FLAREResult(
            answer=answer,
            all_chunks=all_chunks,
            retrieval_rounds=len(novel_per_round),
            total_tokens=total_tokens,
            novel_tokens_per_round=novel_per_round,
        )


# =============================================================================
# PHASE 5C: AgenticRAG
# =============================================================================

@dataclass
class AgentAction:
    """One step taken by the agent during reasoning."""
    action_type: str      # "retrieve", "generate", "clarify", "done"
    input: str            # What was searched or asked
    output: str           # What was retrieved or generated
    reasoning: str = ""   # Why the agent took this action


class AgenticRAG:
    """
    Multi-turn agentic RAG system with tool use.

    WHAT MAKES IT "AGENTIC":
        - The LLM decides WHAT to do, not just executes a fixed pipeline
        - Has access to tools: search, retrieve, clarify, generate
        - Can break complex questions into sub-questions
        - Can do multiple retrieval rounds with different queries
        - Stops when it has enough information to answer confidently

    EXAMPLE — "Compare Q1 and Q3 revenue and explain the difference":
        Turn 1: Agent decides → "I need Q1 data" → retrieves Q1 revenue
        Turn 2: Agent decides → "I need Q3 data" → retrieves Q3 revenue
        Turn 3: Agent decides → "I have enough" → generates comparison answer

    TOOLS AVAILABLE:
        search(query)      → runs hybrid search, returns chunks
        retrieve_by_date(date_range) → filters by date metadata
        clarify(question)  → asks a clarifying question
        finish(answer)     → produces the final answer

    WHEN TO USE:
        - Complex multi-part questions
        - Questions that require comparing across multiple documents
        - When you don't know upfront which retrieval strategy to use
        - Research-style questions with multiple sub-questions

    COST:
        - Multiple LLM calls per query (3-8 typical)
        - Each retrieval adds latency
        - Best for asynchronous/batch use cases
    """

    # System prompt that defines the agent's behavior and available tools
    AGENT_SYSTEM_PROMPT = """You are a document research assistant with access to a knowledge base.
You can search for information to answer questions thoroughly and accurately.

AVAILABLE TOOLS (respond with ONLY JSON):
{
  "action": "search",
  "query": "specific search query"
}
OR
{
  "action": "finish",
  "answer": "your complete answer with citations"
}

RULES:
1. Search for specific information you need — be precise with queries
2. You can search multiple times with different queries
3. Only include information found in search results
4. When you have enough information, use "finish" with a complete answer
5. Maximum 5 search rounds — finish before then"""

    def __init__(
        self,
        retrieval_executor,
        prompt_builder,
        max_turns: int = 5,
    ):
        """
        Args:
            retrieval_executor: For running retrieval at each agent step
            prompt_builder: For building prompts
            max_turns: Maximum number of search rounds before forcing an answer
        """
        self.retrieval_executor = retrieval_executor
        self.prompt_builder = prompt_builder
        self.max_turns = max_turns

    async def run(self, query: str) -> dict:
        """
        Run the agentic RAG loop.

        The agent alternates between:
        1. Deciding what action to take (LLM call)
        2. Executing that action (retrieval)
        Until it decides to "finish" with an answer.

        Returns:
            Dict with 'answer', 'actions', 'total_turns', 'total_tokens'
        """
        from openai import AsyncOpenAI
        from config.settings import settings
        from retrieval.router.query_router import RoutingDecision
        from utils.models import QueryType, RetrievalStrategy

        client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Conversation history — grows as the agent takes actions
        # This is what the LLM reads to decide its next action
        messages = [
            {"role": "system", "content": self.AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}"},
        ]

        all_retrieved_chunks = []   # All chunks found across all searches
        actions_taken: list[AgentAction] = []
        total_tokens = 0
        final_answer = ""

        # Agent loop — runs until "finish" action or max_turns reached
        for turn in range(self.max_turns):
            logger.info("Agentic RAG turn %d/%d", turn + 1, self.max_turns)

            try:
                # ── LLM DECIDES NEXT ACTION ───────────────────────────────────
                # Give the agent the conversation so far and ask what to do next
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )

                total_tokens += response.usage.total_tokens if response.usage else 0
                action_text = response.choices[0].message.content
                action_data = json.loads(action_text)
                action_type = action_data.get("action", "finish")

                # ── EXECUTE THE ACTION ────────────────────────────────────────

                if action_type == "finish":
                    # Agent has enough information — extract the answer
                    final_answer = action_data.get("answer", "")
                    actions_taken.append(AgentAction(
                        action_type="finish",
                        input=query,
                        output=final_answer,
                        reasoning="Agent determined sufficient information gathered",
                    ))
                    logger.info("Agent finished after %d turns", turn + 1)
                    break

                elif action_type == "search":
                    # Agent wants to search for something specific
                    search_query = action_data.get("query", query)
                    logger.info("Agent searching: '%s'", search_query[:60])

                    # Run retrieval for the agent's chosen query
                    decision = RoutingDecision(
                        query_type=QueryType.SEMANTIC,
                        primary_strategy=RetrievalStrategy.HYBRID,
                        fallback_strategy=RetrievalStrategy.SEMANTIC,
                        expanded_queries=[search_query],
                    )
                    # ISOLATION: routes to public corpus (store_override defaults None). If this
                    # is ever wired to a user-reachable path that can carry an X-Session-Id, it
                    # MUST thread store_override or it leaks public chunks into a session context.
                    # See _resolve_query_target.
                    result = await self.retrieval_executor.execute(decision, top_k=5)

                    # Format retrieved chunks as text for the agent to read
                    if result.chunks:
                        chunks_text = "\n\n".join([
                            f"[Source {i+1}] {rc.chunk.metadata.get('title', 'Unknown')}\n"
                            f"{rc.chunk.content}"
                            for i, rc in enumerate(result.chunks[:5])
                        ])
                        all_retrieved_chunks.extend(result.chunks)
                    else:
                        chunks_text = "No relevant information found for this query."

                    # Add the agent's action to conversation history
                    messages.append({
                        "role": "assistant",
                        "content": action_text,
                    })
                    # Add the search results as a "tool result"
                    messages.append({
                        "role": "user",
                        "content": f"Search results:\n{chunks_text}\n\nContinue.",
                    })

                    actions_taken.append(AgentAction(
                        action_type="search",
                        input=search_query,
                        output=chunks_text[:200],  # Abbreviated for logging
                    ))

                else:
                    # Unknown action — stop to prevent infinite loops
                    logger.warning("Unknown agent action: %s", action_type)
                    break

            except Exception as e:
                logger.error("Agent error on turn %d: %s", turn, e)
                break

        # If agent never called "finish", generate a final answer from all found chunks
        if not final_answer and all_retrieved_chunks:
            logger.info("Agent reached max turns — generating answer from collected context")
            prompt = self.prompt_builder.build(
                query=query,
                chunks=all_retrieved_chunks[:10],
                query_type=QueryType.ANALYTICAL,
            )
            from generation.llm_service import get_generation_service
            llm = get_generation_service()
            result = await llm.generate(
                query=query,
                prompt=prompt,
                query_type=QueryType.ANALYTICAL,
                strategy_used=RetrievalStrategy.HYBRID,
            )
            final_answer = result.answer
            total_tokens += result.total_tokens

        return {
            "answer": final_answer or "I was unable to find sufficient information to answer this question.",
            "actions": [
                {
                    "type": a.action_type,
                    "input": a.input,
                    "reasoning": a.reasoning,
                }
                for a in actions_taken
            ],
            "total_turns": len(actions_taken),
            "total_tokens": total_tokens,
            "chunks_retrieved": len(all_retrieved_chunks),
        }
