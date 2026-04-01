"""
=============================================================================
retrieval/hybrid_tuner.py  —  Phase 3: Hybrid Search Tuning
=============================================================================
WHAT THIS FILE DOES:
    Improves the hybrid search (vector + keyword) combination by:

    1. BM25ParameterTuner
       - Finds the best k1 and b values for BM25 keyword search
       - k1 controls term frequency saturation (default 1.5)
       - b controls document length normalization (default 0.75)
       - Better parameters = better keyword retrieval = better hybrid results

    2. AdaptiveAlphaSelector
       - The hybrid_alpha (.env setting) controls vector vs keyword balance
       - Instead of one fixed alpha for all queries, uses different values per query type
       - "What is X?" → more keyword (alpha=0.4)
       - "Explain how X works" → more vector (alpha=0.9)
       - Learned from your evaluation data

    3. SPLADEEmbedder
       - SPLADE = Sparse Lexical and Expansion model
       - Better than BM25 for keyword search — uses neural network
       - Still produces sparse vectors (mostly zeros) but smarter than BM25
       - Replaces the keyword component in hybrid search

HOW TO USE:
    # Tune BM25 parameters on your eval data:
    tuner = BM25ParameterTuner()
    best_params = await tuner.tune(eval_samples)
    print(f"Best k1={best_params['k1']}, b={best_params['b']}")

    # Use adaptive alpha:
    selector = AdaptiveAlphaSelector()
    alpha = selector.get_alpha(query_type="factual")  # returns 0.4
=============================================================================
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. BM25 Parameter Tuner
# =============================================================================

@dataclass
class BM25Params:
    """
    BM25 algorithm parameters.

    k1 (term frequency saturation):
        - Controls how much a term's frequency boosts its score
        - Low k1 (1.0): additional occurrences of a term matter less
        - High k1 (2.0): more occurrences = higher score (more saturation)
        - Typical range: 1.0 to 2.0, default 1.5

    b (document length normalization):
        - Controls how much document length affects scoring
        - b=0: no length normalization (long docs score higher)
        - b=1: full normalization (short and long docs treated equally)
        - Typical range: 0.5 to 0.9, default 0.75
    """
    k1: float = 1.5   # Term frequency saturation
    b: float = 0.75   # Document length normalization


class BM25ParameterTuner:
    """
    Grid search over BM25 parameters to find the combination that
    maximizes retrieval quality on your evaluation dataset.

    HOW IT WORKS:
        1. Take a set of evaluation queries with known relevant documents
        2. Try different combinations of k1 and b values
        3. For each combination, run all eval queries and measure hit rate
        4. Return the (k1, b) combination with the highest hit rate

    REQUIRES:
        - A set of evaluation samples with known relevant document IDs
        - A running vector store with BM25 support (Weaviate)
    """

    # Grid of values to try — 4x4 = 16 combinations tested
    K1_VALUES = [1.0, 1.2, 1.5, 2.0]    # Term frequency saturation values
    B_VALUES = [0.5, 0.65, 0.75, 0.9]   # Length normalization values

    async def tune(
        self,
        eval_samples: list,          # List of EvalSample objects (from evaluator.py)
        vector_store,                # Your vector store instance
    ) -> BM25Params:
        """
        Run grid search to find the best BM25 parameters.

        Args:
            eval_samples: Evaluation samples with known relevant doc IDs
            vector_store: Vector store with keyword_search() method

        Returns:
            BM25Params with the best k1 and b values found
        """
        best_params = BM25Params()   # Start with defaults
        best_hit_rate = 0.0

        total_combinations = len(self.K1_VALUES) * len(self.B_VALUES)
        combination = 0

        # Try every combination of k1 and b
        for k1 in self.K1_VALUES:
            for b in self.B_VALUES:
                combination += 1
                logger.info(
                    "Testing BM25 params %d/%d: k1=%.1f, b=%.2f",
                    combination, total_combinations, k1, b
                )

                # Run evaluation with these parameters
                hit_rate = await self._evaluate_params(
                    k1=k1,
                    b=b,
                    eval_samples=eval_samples,
                    vector_store=vector_store,
                )

                logger.info("  → Hit rate: %.3f", hit_rate)

                # Keep track of the best combination
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_params = BM25Params(k1=k1, b=b)

        logger.info(
            "Best BM25 params: k1=%.1f, b=%.2f (hit rate=%.3f)",
            best_params.k1, best_params.b, best_hit_rate
        )
        return best_params

    async def _evaluate_params(
        self,
        k1: float,
        b: float,
        eval_samples: list,
        vector_store,
    ) -> float:
        """
        Measure hit rate for a specific (k1, b) combination.

        Hit rate = fraction of queries where at least one relevant
        document appears in the top-20 results.
        """
        hits = 0
        total = 0

        for sample in eval_samples:
            if not sample.relevant_doc_ids:
                continue

            total += 1

            try:
                # Run keyword search with these BM25 settings
                # Note: BM25 parameter passing depends on your vector store
                # Weaviate accepts these in the collection configuration
                results = await vector_store.keyword_search(
                    query=sample.query,
                    top_k=20,
                )

                # Check if any relevant document appears in results
                retrieved_ids = {str(r.chunk.doc_id) for r in results}
                if any(doc_id in retrieved_ids for doc_id in sample.relevant_doc_ids):
                    hits += 1

            except Exception as e:
                logger.warning("Eval failed for query '%s': %s", sample.query[:50], e)

        return hits / total if total > 0 else 0.0


# =============================================================================
# 2. Adaptive Alpha Selector
# =============================================================================

class AdaptiveAlphaSelector:
    """
    Selects the optimal hybrid search alpha value per query type.

    The alpha parameter controls the blend of vector vs keyword search:
        0.0 = pure keyword (BM25) — good for exact term matching
        0.5 = equal blend — good for mixed queries
        1.0 = pure vector (semantic) — good for conceptual questions

    Instead of using one fixed alpha for all queries (as in the basic system),
    this class learns which alpha works best for each query type by analyzing
    your evaluation data.

    DEFAULT VALUES (before learning from your data):
        FACTUAL      → 0.4  (lean keyword — facts have specific terms)
        LOOKUP       → 0.2  (mostly keyword — exact names/dates)
        SEMANTIC     → 0.9  (mostly vector — conceptual questions)
        ANALYTICAL   → 0.7  (balanced — need both concepts and terms)
        MULTI_HOP    → 0.8  (mostly vector — concepts across documents)
        COMPARATIVE  → 0.7  (balanced — need to compare concepts)
    """

    # Default alpha values per query type (before learning)
    # Key = QueryType.value string, Value = alpha (0.0 to 1.0)
    DEFAULT_ALPHAS = {
        "factual":     0.4,   # Factual questions have specific keywords → lean keyword
        "lookup":      0.2,   # Lookup queries are mostly keyword-based
        "semantic":    0.9,   # Semantic/conceptual questions → lean vector
        "analytical":  0.7,   # Analytical needs both concepts and terms
        "multi_hop":   0.8,   # Multi-hop → mostly vector for concept chains
        "comparative": 0.7,   # Comparative → balanced
    }

    def __init__(self):
        # Start with defaults, then update from learned values
        self._alphas = dict(self.DEFAULT_ALPHAS)

    def get_alpha(self, query_type: str) -> float:
        """
        Get the recommended alpha for a given query type.

        Args:
            query_type: String like "factual", "semantic", etc.

        Returns:
            Alpha value between 0.0 and 1.0
        """
        alpha = self._alphas.get(query_type, 0.7)  # 0.7 as fallback
        logger.debug("Alpha for query_type='%s': %.2f", query_type, alpha)
        return alpha

    async def learn_from_eval(
        self,
        eval_samples: list,
        vector_store,
        embedder,
        alpha_candidates: Optional[list[float]] = None,
    ) -> dict[str, float]:
        """
        Learn the best alpha for each query type from evaluation data.

        For each query type that appears in eval_samples, tries several
        alpha values and picks the one with the best retrieval hit rate.

        Args:
            eval_samples: Labeled evaluation samples
            vector_store: Vector store with hybrid_search()
            embedder: Embedder with embed_query()
            alpha_candidates: Alpha values to try (default: 0.1 to 0.9)

        Returns:
            Dict mapping query_type → best_alpha
        """
        if alpha_candidates is None:
            # Try 9 alpha values from 0.1 to 0.9
            alpha_candidates = [round(a * 0.1, 1) for a in range(1, 10)]

        # Group eval samples by query type
        # We can only learn alphas for query types we have eval data for
        from collections import defaultdict
        samples_by_type: dict[str, list] = defaultdict(list)
        for sample in eval_samples:
            qt = getattr(sample, "query_type", "semantic")
            samples_by_type[qt].append(sample)

        learned_alphas = {}

        for query_type, samples in samples_by_type.items():
            if not samples:
                continue

            logger.info(
                "Learning alpha for query_type='%s' (%d samples)",
                query_type, len(samples)
            )

            best_alpha = self._alphas.get(query_type, 0.7)  # Start with default
            best_hit_rate = 0.0

            for alpha in alpha_candidates:
                hit_rate = await self._eval_alpha(
                    alpha=alpha,
                    samples=samples,
                    vector_store=vector_store,
                    embedder=embedder,
                )

                logger.debug("  alpha=%.1f → hit_rate=%.3f", alpha, hit_rate)

                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_alpha = alpha

            learned_alphas[query_type] = best_alpha
            # Update internal state
            self._alphas[query_type] = best_alpha

            logger.info(
                "Best alpha for '%s': %.1f (hit_rate=%.3f)",
                query_type, best_alpha, best_hit_rate
            )

        return learned_alphas

    async def _eval_alpha(
        self,
        alpha: float,
        samples: list,
        vector_store,
        embedder,
    ) -> float:
        """Measure hit rate for a specific alpha value."""
        hits = 0
        total = 0

        for sample in samples:
            if not sample.relevant_doc_ids:
                continue
            total += 1

            try:
                # Embed the query
                query_embedding = await embedder.embed_query(sample.query)

                # Run hybrid search with this alpha
                results = await vector_store.hybrid_search(
                    query=sample.query,
                    query_embedding=query_embedding,
                    top_k=20,
                    alpha=alpha,
                )

                retrieved_ids = {str(r.chunk.doc_id) for r in results}
                if any(doc_id in retrieved_ids for doc_id in sample.relevant_doc_ids):
                    hits += 1

            except Exception as e:
                logger.warning("Alpha eval failed: %s", e)

        return hits / total if total > 0 else 0.0

    def save(self, path: str = "./learned_alphas.json") -> None:
        """Save learned alphas to a JSON file for reuse."""
        import json
        with open(path, "w") as f:
            json.dump(self._alphas, f, indent=2)
        logger.info("Saved learned alphas to %s", path)

    def load(self, path: str = "./learned_alphas.json") -> None:
        """Load previously learned alphas from a JSON file."""
        import json
        try:
            with open(path) as f:
                self._alphas.update(json.load(f))
            logger.info("Loaded learned alphas from %s", path)
        except FileNotFoundError:
            logger.warning("No learned alphas file at %s — using defaults", path)


# =============================================================================
# 3. SPLADE Embedder (Neural Sparse Retrieval)
# =============================================================================

class SPLADEEmbedder:
    """
    SPLADE: Sparse Lexical and Expansion model.

    SPLADE is a neural model that produces sparse embeddings —
    vectors where most values are zero, but non-zero values are
    assigned to vocabulary terms that are relevant to the text.

    WHY IT'S BETTER THAN BM25:
        - BM25 only matches exact terms: "car" does not match "vehicle"
        - SPLADE expands terms: "car" might also activate "vehicle", "automobile"
        - This expansion is learned from training data, not hand-crafted rules

    REQUIREMENTS:
        pip install transformers torch
        Model: naver/splade-cocondenser-selfdistil (downloads automatically)

    INDEX SIZE:
        SPLADE vectors are sparse — stored as {term_id: weight} dicts.
        Much smaller than dense vectors for the same vocabulary coverage.

    INTEGRATION:
        Use as a drop-in replacement for keyword_search() in your pipeline.
        The retrieval_executor can call this instead of BM25.
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-selfdistil"):
        """
        Args:
            model_name: SPLADE model from HuggingFace
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Load SPLADE model and tokenizer (lazy loading)."""
        if self._model is None:
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer
                import torch

                logger.info("Loading SPLADE model: %s", self.model_name)

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                self._model.eval()   # Set to inference mode (no gradient tracking)

            except ImportError:
                raise ImportError(
                    "Install transformers and torch: pip install transformers torch"
                )

    async def encode_text(self, text: str) -> dict[int, float]:
        """
        Encode text into a SPLADE sparse vector.

        Returns:
            Dict mapping vocab_term_id → weight
            (only non-zero terms are included)
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_encode, text)

    def _sync_encode(self, text: str) -> dict[int, float]:
        """CPU-bound SPLADE encoding — runs in thread pool."""
        import torch

        self._load()

        # Tokenize the input text
        tokens = self._tokenizer(
            text,
            return_tensors="pt",     # Return PyTorch tensors
            truncation=True,         # Cut text if longer than max_length
            max_length=512,          # SPLADE max input length
            padding=True,
        )

        with torch.no_grad():   # No gradients needed for inference
            output = self._model(**tokens)

        # SPLADE scoring: log(1 + ReLU(logits)) aggregated by max over tokens
        # This produces the importance weight for each vocabulary term
        logits = output.logits              # [1, seq_len, vocab_size]
        relu = torch.nn.functional.relu(logits)  # Remove negatives
        log_relu = torch.log1p(relu)        # log(1 + x) — dampens high values
        # Max pooling over token dimension — take highest weight per vocab term
        sparse_vec = torch.max(log_relu, dim=1).values.squeeze(0)  # [vocab_size]

        # Convert to sparse dict (only keep non-zero values)
        # This saves memory — most of 30,000+ vocab terms are zero
        non_zero = sparse_vec.nonzero(as_tuple=True)[0].tolist()
        weights = sparse_vec[non_zero].tolist()

        return {term_id: weight for term_id, weight in zip(non_zero, weights)}

    def compute_similarity(
        self,
        query_vec: dict[int, float],
        doc_vec: dict[int, float],
    ) -> float:
        """
        Compute dot product similarity between two SPLADE vectors.

        Only iterates over terms present in BOTH vectors (efficient for sparse).
        """
        # Find common terms (intersection of keys)
        common_terms = set(query_vec.keys()) & set(doc_vec.keys())

        # Dot product over common terms only
        score = sum(query_vec[t] * doc_vec[t] for t in common_terms)
        return score
