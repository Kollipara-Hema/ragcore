"""
=============================================================================
reranking/advanced_reranker.py  —  Phase 4: Advanced Reranking
=============================================================================
WHAT RERANKING DOES (recap):
    After the vector store retrieves the top-20 candidate chunks,
    the reranker scores each (query, chunk) pair more carefully
    and returns only the top-5 most relevant.

    Think of it as a two-stage process:
        Stage 1 (fast): Vector store retrieves 20 candidates
        Stage 2 (precise): Reranker picks the best 5 from those 20

WHAT THIS FILE ADDS:
    1. MonoT5Reranker
       - Uses a T5 model trained specifically for passage reranking
       - Better multilingual support than the cross-encoder
       - Larger model = better quality but slower

    2. LLMReranker
       - Uses GPT-4o-mini to score each chunk's relevance
       - Highest quality reranking available
       - Costs more (LLM API call per chunk) but very accurate
       - Best for: when answer quality is critical, cost is secondary

    3. EnsembleReranker
       - Combines scores from multiple rerankers
       - More robust than any single reranker
       - Each reranker votes on the final ranking

    4. CrossEncoderFineTuner
       - Fine-tunes the cross-encoder reranker on your domain data
       - Domain-specific reranking improves quality significantly

HOW TO USE:
    from reranking.advanced_reranker import LLMReranker
    reranker = LLMReranker()
    reranked = await reranker.rerank(query, chunks, top_k=5)
=============================================================================
"""

from __future__ import annotations
import asyncio
import json
import logging

from utils.models import RetrievedChunk

logger = logging.getLogger(__name__)


# =============================================================================
# RERANKER 1: MonoT5Reranker
# =============================================================================

class MonoT5Reranker:
    """
    MonoT5: Passage reranking using a T5 sequence-to-sequence model.

    HOW IT WORKS:
        Input: "Query: {query} Document: {chunk} Relevant:"
        Output: probability of "true" vs "false"
        Score = probability that the document is relevant to the query

    MODELS AVAILABLE (in order of quality/size):
        castorini/monot5-base-msmarco       (220M params, fast)
        castorini/monot5-large-msmarco      (770M params, better)
        castorini/monot5-3b-msmarco         (3B params, best, slow)

    VS CROSS-ENCODER:
        - MonoT5 is a generative model (produces text)
        - Cross-encoder is a discriminative model (produces a score directly)
        - MonoT5 handles multilingual text better
        - MonoT5 is slower but more thorough

    WHEN TO USE:
        - Documents in multiple languages
        - When cross-encoder performance is not good enough
        - As part of EnsembleReranker for better stability
    """

    def __init__(
        self,
        model_name: str = "castorini/monot5-base-msmarco",
        batch_size: int = 8,
    ):
        """
        Args:
            model_name: Which MonoT5 variant to use
            batch_size: How many (query, chunk) pairs to score at once
                       Reduce if running out of GPU/CPU memory
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Load MonoT5 model (lazy loading — only when first used)."""
        if self._model is None:
            try:
                from transformers import T5ForConditionalGeneration, T5Tokenizer
                import torch

                logger.info("Loading MonoT5 reranker: %s", self.model_name)

                self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self._model.eval()

                # Move model to available device if possible, using torch to avoid unused import warning
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model.to(device)

                # Get token IDs for "true" and "false" — used for scoring
                # MonoT5 outputs these words, we compare their probabilities
                self._true_token_id = self._tokenizer.encode("true")[0]
                self._false_token_id = self._tokenizer.encode("false")[0]

            except ImportError:
                raise ImportError(
                    "Install transformers: pip install transformers sentencepiece"
                )

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks using MonoT5 relevance scoring.

        Args:
            query: The user's question
            chunks: Candidate chunks from vector retrieval
            top_k: How many to return after reranking

        Returns:
            Top-k chunks sorted by MonoT5 relevance score
        """
        if not chunks:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_rerank, query, chunks, top_k
        )

    def _sync_rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """CPU/GPU-bound reranking — runs in thread pool."""
        import torch

        self._load()

        scores = []

        # Process in batches to manage memory
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            # Format each (query, chunk) pair as MonoT5 input
            # The model was trained with this exact format
            inputs = [
                f"Query: {query} Document: {rc.chunk.content[:512]} Relevant:"
                for rc in batch
            ]

            # Tokenize all inputs in the batch
            encoded = self._tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Run the model
            with torch.no_grad():
                # We need the logits for the first generated token
                # MonoT5 generates "true" or "false" — we want P("true")
                decoder_input_ids = torch.zeros(
                    (len(batch), 1), dtype=torch.long
                )  # Start token

                outputs = self._model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )

                # Get logits for the "true" and "false" tokens
                logits = outputs.logits[:, 0, :]  # First generated token
                true_logit = logits[:, self._true_token_id]
                false_logit = logits[:, self._false_token_id]

                # Compute P(true) using softmax over true/false
                import torch.nn.functional as F
                probs = F.softmax(
                    torch.stack([true_logit, false_logit], dim=1), dim=1
                )
                relevance_probs = probs[:, 0].tolist()  # P(true) for each chunk

            scores.extend(relevance_probs)

        # Assign scores and sort
        for rc, score in zip(chunks, scores):
            rc.score = float(score)

        chunks.sort(key=lambda rc: rc.score, reverse=True)

        for rank, rc in enumerate(chunks[:top_k]):
            rc.rank = rank

        return chunks[:top_k]


# =============================================================================
# RERANKER 2: LLMReranker
# =============================================================================

class LLMReranker:
    """
    Uses GPT-4o-mini to score each (query, chunk) pair.

    HOW IT WORKS:
        For each chunk, asks the LLM:
        "On a scale of 1-10, how relevant is this passage to the query?"
        The numeric score is used to re-rank the chunks.

    PROS:
        - Highest quality reranking available
        - Understands nuance, implication, and context
        - No model to download/host

    CONS:
        - Costs money (LLM API call per chunk)
        - Slower than local rerankers (~1-2s per chunk)
        - Best used for top-K (e.g., rerank only top 10 candidates)

    COST ESTIMATE:
        20 chunks × ~200 tokens each = ~4000 tokens
        gpt-4o-mini at $0.30/1M tokens = ~$0.0012 per query
        At 1000 queries/day = ~$1.20/day for reranking alone

    WHEN TO USE:
        - High-value queries where accuracy matters most
        - As part of EnsembleReranker
        - When latency is not critical (reports, async workflows)
    """

    # Prompt that asks the LLM to score relevance
    SCORING_PROMPT = """You are a relevance scoring assistant.
Given a query and a document passage, rate how relevant the passage is
to answering the query on a scale from 1 to 10.

1 = Completely irrelevant
5 = Somewhat relevant, partially addresses the query
10 = Highly relevant, directly answers the query

Respond with ONLY a JSON object: {"score": <number>}
No explanation needed."""

    def __init__(self, model: str = "gpt-4o-mini", max_concurrent: int = 5):
        """
        Args:
            model: Which LLM to use for scoring (use cheap/fast model)
            max_concurrent: Max parallel API calls (prevents rate limiting)
        """
        self.model = model
        # Semaphore limits concurrent API calls to avoid rate limits
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks by asking an LLM to score each one.

        All scoring API calls run in parallel (limited by max_concurrent).
        """
        if not chunks:
            return []

        # Score all chunks in parallel
        score_tasks = [
            self._score_chunk(query, rc)
            for rc in chunks
        ]
        scores = await asyncio.gather(*score_tasks, return_exceptions=True)

        # Apply scores — use original score as fallback if LLM call failed
        for rc, score in zip(chunks, scores):
            if isinstance(score, float):
                rc.score = score
            else:
                logger.warning("LLM scoring failed for a chunk: %s", score)
                # Keep original score (don't penalize failures)

        # Sort by score and return top_k
        chunks.sort(key=lambda rc: rc.score, reverse=True)
        for rank, rc in enumerate(chunks[:top_k]):
            rc.rank = rank

        return chunks[:top_k]

    async def _score_chunk(self, query: str, rc: RetrievedChunk) -> float:
        """
        Score a single (query, chunk) pair using the LLM.

        Uses a semaphore to limit concurrent API calls.
        """
        async with self._semaphore:  # Limit concurrent calls
            try:
                from openai import AsyncOpenAI
                from config.settings import settings

                client = AsyncOpenAI(api_key=settings.openai_api_key)

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SCORING_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Query: {query}\n\n"
                                f"Passage: {rc.chunk.content[:800]}"  # Limit length
                            ),
                        },
                    ],
                    temperature=0,         # Consistent scoring
                    max_tokens=20,         # Only need {"score": X}
                    response_format={"type": "json_object"},
                )

                data = json.loads(response.choices[0].message.content)
                raw_score = float(data.get("score", 5))

                # Normalize from 1-10 scale to 0-1 scale
                return (raw_score - 1) / 9.0

            except Exception as e:
                logger.warning("LLM scoring error: %s", e)
                return rc.score   # Return original score on failure


# =============================================================================
# RERANKER 3: EnsembleReranker
# =============================================================================

class EnsembleReranker:
    """
    Combines scores from multiple rerankers using weighted averaging.

    WHY ENSEMBLE:
        Different rerankers have different strengths:
        - Cross-encoder: fast, good for English
        - MonoT5: better multilingual
        - LLM reranker: best quality but expensive

        Combining them is more robust than any single reranker.

    HOW IT WORKS:
        Final score = w1 * score1 + w2 * score2 + w3 * score3
        where w1 + w2 + w3 = 1.0

    EXAMPLE CONFIGURATION:
        Cross-encoder (weight 0.6) + LLM reranker (weight 0.4)
        → Fast and high quality at reasonable cost
    """

    def __init__(self, rerankers_and_weights: list[tuple]):
        """
        Args:
            rerankers_and_weights: List of (reranker, weight) tuples.
                Weights should sum to 1.0.
                Example: [(CrossEncoderReranker(), 0.6), (LLMReranker(), 0.4)]
        """
        self.rerankers = [(r, w) for r, w in rerankers_and_weights]

        # Validate that weights sum to 1.0
        total_weight = sum(w for _, w in self.rerankers)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                "Reranker weights sum to %.2f, not 1.0 — normalizing",
                total_weight
            )
            # Normalize weights
            self.rerankers = [
                (r, w / total_weight) for r, w in self.rerankers
            ]

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Run all rerankers, combine scores with weights, return top-k.
        """
        if not chunks:
            return []

        # Build a lookup from chunk_id → chunk for merging scores
        chunk_map = {str(rc.chunk.chunk_id): rc for rc in chunks}

        # Initialize combined scores to zero
        combined_scores = {chunk_id: 0.0 for chunk_id in chunk_map}

        # Run each reranker and add its weighted score
        for reranker, weight in self.rerankers:
            try:
                # Each reranker returns chunks with its own scores
                reranked = await reranker.rerank(query, list(chunks), len(chunks))

                # Add weighted score for each chunk
                for rc in reranked:
                    cid = str(rc.chunk.chunk_id)
                    if cid in combined_scores:
                        combined_scores[cid] += weight * rc.score

            except Exception as e:
                logger.warning("Reranker %s failed: %s", type(reranker).__name__, e)
                # Continue with other rerankers — don't fail completely

        # Apply combined scores and sort
        for cid, score in combined_scores.items():
            chunk_map[cid].score = score

        result = sorted(chunk_map.values(), key=lambda rc: rc.score, reverse=True)

        for rank, rc in enumerate(result[:top_k]):
            rc.rank = rank

        return result[:top_k]


# =============================================================================
# FINE-TUNER: Train cross-encoder on your domain data
# =============================================================================

class CrossEncoderFineTuner:
    """
    Fine-tune a cross-encoder reranker on domain-specific data.

    WHY FINE-TUNE THE RERANKER:
        The base cross-encoder (ms-marco-MiniLM) is trained on web search data.
        For medical, legal, or technical documents, it may not know which
        passages are truly relevant for your specific query types.

    WHAT YOU NEED:
        Labeled pairs: (query, positive_chunk, negative_chunk)
        - positive_chunk: A chunk that correctly answers the query
        - negative_chunk: A chunk retrieved but does NOT answer the query

        You can generate these from user feedback:
        - Thumbs up on a result → positive pair
        - Thumbs down on a result → negative pair
        Or from your evaluation dataset.

    RESULT:
        A fine-tuned model saved to ./fine_tuned_reranker/
        Update settings.reranker_model to use it.
    """

    def __init__(self, base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            base_model: Base cross-encoder to start from
        """
        self.base_model = base_model

    async def fine_tune(
        self,
        training_triplets: list[tuple[str, str, str]],
        output_dir: str = "./fine_tuned_reranker",
        num_epochs: int = 3,
    ) -> str:
        """
        Fine-tune the cross-encoder using labeled positive/negative pairs.

        Args:
            training_triplets: List of (query, positive_chunk, negative_chunk)
            output_dir: Where to save the fine-tuned model
            num_epochs: Number of training epochs

        Returns:
            Path to saved model
        """
        try:
            from sentence_transformers import CrossEncoder, InputExample
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        logger.info(
            "Fine-tuning reranker on %d triplets for %d epochs",
            len(training_triplets), num_epochs
        )

        model = CrossEncoder(
            self.base_model,
            num_labels=1,        # Binary relevance score
            max_length=512,
        )

        # Create training examples
        # Positive pairs get label 1.0 (relevant)
        # Negative pairs get label 0.0 (not relevant)
        train_examples = []
        for query, positive, negative in training_triplets:
            train_examples.append(
                InputExample(texts=[query, positive], label=1.0)
            )
            train_examples.append(
                InputExample(texts=[query, negative], label=0.0)
            )

        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=16,
        )

        # Train the model
        model.fit(
            train_dataloader=train_dataloader,
            epochs=num_epochs,
            output_path=output_dir,
            show_progress_bar=True,
        )

        logger.info("Fine-tuned reranker saved to: %s", output_dir)
        logger.info("To use: set RERANKER_MODEL=%s in .env", output_dir)
        return output_dir
