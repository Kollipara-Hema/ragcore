"""
=============================================================================
embeddings/advanced_embeddings.py  —  Phase 2: Advanced Embeddings
=============================================================================
WHAT THIS FILE ADDS:
    1. MatryoshkaEmbedder
       - Produces embeddings at multiple dimensions (1024, 512, 256, 128)
       - Smaller dimensions = faster search, slightly lower quality
       - Larger dimensions = slower search, higher quality
       - You can tune the dimension at query time without re-indexing

    2. ColBERTEmbedder  (Late Interaction)
       - Produces one embedding vector PER TOKEN (not per chunk)
       - At retrieval time: compares every query token to every doc token
       - More accurate than single-vector embeddings for long queries
       - Trade-off: larger index size, slower but more precise

    3. EmbeddingFineTuner
       - Fine-tunes a BGE model on YOUR specific domain data
       - Improves retrieval quality for domain-specific vocabulary
       - Requires: pairs of (query, relevant_document) examples

HOW TO USE:
    # Matryoshka — choose dimension at query time:
    from embeddings.advanced_embeddings import MatryoshkaEmbedder
    embedder = MatryoshkaEmbedder(dimension=512)   # smaller = faster
    embedding = await embedder.embed_query("What is X?")

    # Fine-tune on your data:
    from embeddings.advanced_embeddings import EmbeddingFineTuner
    tuner = EmbeddingFineTuner()
    await tuner.fine_tune(training_pairs, output_dir="./my_fine_tuned_model")
=============================================================================
"""

from __future__ import annotations
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDER 1: MatryoshkaEmbedder
# =============================================================================

class MatryoshkaEmbedder:
    """
    Matryoshka Representation Learning (MRL) embeddings.

    Named after Russian nesting dolls — a 1024-dim embedding contains
    a valid 512-dim embedding in its first 512 values, which contains
    a valid 256-dim embedding, and so on.

    This means you can:
    - Index at full 1024 dimensions
    - Query at 256 dimensions for speed
    - No re-indexing needed when changing dimension

    SUPPORTED MODELS (already trained with Matryoshka):
        - "nomic-ai/nomic-embed-text-v1.5"  (best open source Matryoshka model)
        - "text-embedding-3-large"           (OpenAI, supports dimension param)
        - "text-embedding-3-small"           (OpenAI, cheaper)
    """

    # Valid dimensions for Matryoshka models
    VALID_DIMENSIONS = [1536, 1024, 768, 512, 256, 128, 64]

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        dimension: int = 512,
    ):
        """
        Args:
            model_name: HuggingFace model that supports Matryoshka
            dimension: Output embedding size. Smaller = faster, larger = more accurate.
                      Must be one of: 1536, 1024, 768, 512, 256, 128, 64
        """
        # Validate dimension choice
        if dimension not in self.VALID_DIMENSIONS:
            raise ValueError(
                f"dimension must be one of {self.VALID_DIMENSIONS}, got {dimension}"
            )

        self.model_name = model_name
        self.dimension = dimension
        self._model = None   # Lazy-loaded — only downloaded when first used

    def _get_model(self):
        """Load the model on first use (lazy loading saves startup time)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading Matryoshka model: %s (will truncate to %d dims)",
                self.model_name, self.dimension
            )
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts and truncate to the configured dimension.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors, each of length self.dimension
        """
        import asyncio
        import numpy as np

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, texts)

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        """CPU-bound embedding — runs in thread pool to avoid blocking."""
        import numpy as np

        model = self._get_model()

        # Get full-dimension embeddings
        full_embeddings = model.encode(
            texts,
            normalize_embeddings=True,   # L2 normalize so cosine sim = dot product
            show_progress_bar=False,
        )

        # Truncate to requested dimension (first N values)
        # Matryoshka models guarantee the first N values are meaningful
        truncated = full_embeddings[:, :self.dimension]

        # Re-normalize after truncation (truncation breaks normalization)
        import numpy as np
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        truncated = truncated / norms

        return truncated.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.
        Nomic model uses a prefix to distinguish queries from documents.
        """
        # Nomic model requires task prefix for queries
        if "nomic" in self.model_name:
            prefixed_query = f"search_query: {query}"
        else:
            prefixed_query = query

        results = await self.embed_texts([prefixed_query])
        return results[0]

    async def embed_document(self, text: str) -> list[float]:
        """Embed a document chunk (uses document prefix for Nomic models)."""
        if "nomic" in self.model_name:
            prefixed = f"search_document: {text}"
        else:
            prefixed = text

        results = await self.embed_texts([prefixed])
        return results[0]


# =============================================================================
# EMBEDDER 2: ColBERTEmbedder  (Late Interaction)
# =============================================================================

class ColBERTEmbedder:
    """
    ColBERT: Contextualized Late Interaction over BERT.

    REGULAR EMBEDDING:
        Document → [one vector of 1024 numbers]
        Query → [one vector of 1024 numbers]
        Score = cosine similarity of the two vectors

    COLBERT (late interaction):
        Document → [one vector per token, e.g. 128 tokens × 128 dims]
        Query → [one vector per token, e.g. 12 tokens × 128 dims]
        Score = for each query token, find its best matching doc token,
                sum those max scores  (called "MaxSim")

    WHY IT'S BETTER:
        - "What is the capital of France?" — "capital" token can match
          "Paris is the capital city" more precisely than a single vector
        - Each word in the query independently finds its best match
        - Much better for long, complex queries

    TRADE-OFFS:
        - Index is 128x larger (one vector per token vs per chunk)
        - Requires special vector store support (Weaviate supports this)
        - Slower indexing, but retrieval is still fast

    WHEN TO USE:
        - Long, complex multi-part questions
        - Technical queries with specific terminology
        - When retrieval quality is more important than index size
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        Args:
            model_name: ColBERT model from HuggingFace
        """
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Load ColBERT model (lazy loading)."""
        if self._model is None:
            try:
                # RAGatouille provides an easy ColBERT interface
                from ragatouille import RAGPretrainedModel
                logger.info("Loading ColBERT model: %s", self.model_name)
                self._model = RAGPretrainedModel.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError(
                    "Install ragatouille for ColBERT support: pip install ragatouille"
                )
        return self._model

    def index_documents(self, documents: list[str], index_name: str = "my_index"):
        """
        Build a ColBERT index from a list of document texts.

        Unlike regular embedders, ColBERT builds its own index format.
        This replaces the vector store for ColBERT-indexed documents.

        Args:
            documents: List of document text strings
            index_name: Name for the on-disk index directory
        """
        model = self._get_model()
        # ColBERT indexes the documents itself — creates token-level embeddings
        model.index(
            collection=documents,
            index_name=index_name,
            max_document_length=512,    # Max tokens per document
            split_documents=True,       # Auto-split long documents
        )
        logger.info("ColBERT index built: %s (%d docs)", index_name, len(documents))

    def search(
        self,
        query: str,
        index_name: str = "my_index",
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search the ColBERT index using MaxSim scoring.

        Args:
            query: Search query string
            index_name: Which index to search
            top_k: Number of results to return

        Returns:
            List of dicts with 'content', 'score', 'rank'
        """
        model = self._get_model()
        results = model.search(query=query, index_name=index_name, k=top_k)
        return results


# =============================================================================
# FINE-TUNER: Train BGE on your domain data
# =============================================================================

class EmbeddingFineTuner:
    """
    Fine-tune a pre-trained embedding model on your specific domain data.

    WHY FINE-TUNE:
        - Pre-trained models (BGE, OpenAI) are trained on general internet text
        - Your documents may use specialized vocabulary:
          medical terms, legal language, company-specific jargon, etc.
        - Fine-tuning teaches the model that YOUR domain's similar concepts
          should have similar embeddings

    WHAT YOU NEED:
        Training pairs — examples of (query, relevant_document):
        [
            ("What is the refund policy?", "Customers may return items within 30 days..."),
            ("How to reset password?", "To reset your password, click Forgot Password..."),
            ...
        ]
        More pairs = better results. Aim for 500+ pairs minimum.

    WHAT YOU GET:
        A fine-tuned model saved to ./fine_tuned_model/
        Use it by setting: EMBEDDING_MODEL_BGE=./fine_tuned_model in .env

    HOW TO GENERATE TRAINING DATA (if you don't have it):
        Use generate_training_data() below — it uses an LLM to create
        synthetic Q&A pairs from your existing documents.
    """

    def __init__(self, base_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Args:
            base_model: Which pre-trained model to start from.
                       Using the same model you use for production = no dimension change.
        """
        self.base_model = base_model

    async def fine_tune(
        self,
        training_pairs: list[tuple[str, str]],
        output_dir: str = "./fine_tuned_model",
        num_epochs: int = 3,
        batch_size: int = 16,
    ) -> str:
        """
        Fine-tune the embedding model using contrastive learning.

        CONTRASTIVE LEARNING:
            - Positive pair: (query, relevant document) → embeddings should be similar
            - The model learns to push related text closer in embedding space

        Args:
            training_pairs: List of (query, relevant_document) tuples
            output_dir: Where to save the fine-tuned model
            num_epochs: How many times to train on the full dataset (3-5 is typical)
            batch_size: How many pairs per training step (reduce if out of RAM)

        Returns:
            Path to the saved model directory
        """
        try:
            from sentence_transformers import SentenceTransformer, InputExample, losses
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers torch")

        logger.info(
            "Fine-tuning %s on %d pairs for %d epochs",
            self.base_model, len(training_pairs), num_epochs
        )

        # Load the base model to fine-tune
        model = SentenceTransformer(self.base_model)

        # Convert training pairs to InputExample format
        # Each InputExample has the query text, document text, and similarity label
        train_examples = [
            InputExample(texts=[query, doc], label=1.0)  # 1.0 = these should be similar
            for query, doc in training_pairs
        ]

        # Create a data loader to feed batches during training
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,          # Randomize order each epoch
            batch_size=batch_size,
        )

        # MultipleNegativesRankingLoss:
        # For each positive pair (query, doc) in the batch,
        # all OTHER documents in the batch are treated as negatives.
        # The model learns to rank the true doc above all others.
        # This is the standard loss for embedding model fine-tuning.
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Calculate total training steps
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

        # Run training
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,   # Gradual learning rate increase at start
            output_path=output_dir,      # Save here when done
            show_progress_bar=True,      # Show training progress bar
        )

        logger.info("Fine-tuned model saved to: %s", output_dir)
        logger.info("To use: set EMBEDDING_MODEL_BGE=%s in .env", output_dir)
        return output_dir

    async def generate_training_data(
        self,
        documents: list[str],
        questions_per_doc: int = 5,
        output_file: str = "./training_pairs.json",
    ) -> list[tuple[str, str]]:
        """
        Automatically generate training Q&A pairs from your documents using an LLM.

        This is useful when you don't have existing query-document pairs.
        The LLM reads each document and generates realistic questions a user
        might ask that this document would answer.

        Args:
            documents: List of document text strings
            questions_per_doc: How many questions to generate per document
            output_file: Save training pairs here as JSON for later reuse

        Returns:
            List of (question, document) training pairs
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        from config.settings import settings
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        all_pairs = []
        total = len(documents)

        for i, doc in enumerate(documents):
            logger.info("Generating questions for doc %d/%d", i + 1, total)

            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",   # Cheap model for data generation
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"Generate exactly {questions_per_doc} realistic questions "
                                f"that a user might ask where this document provides the answer. "
                                f"Return ONLY a JSON array of question strings. No explanation."
                            ),
                        },
                        {
                            "role": "user",
                            "content": doc[:2000],  # Limit doc length for efficiency
                        },
                    ],
                    temperature=0.7,   # Some variety in question phrasing
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                data = json.loads(content)

                # Handle various JSON response formats
                if isinstance(data, list):
                    questions = data
                elif isinstance(data, dict):
                    questions = data.get("questions", data.get("items", []))
                else:
                    questions = []

                # Add (question, document) pairs to training set
                for q in questions:
                    if isinstance(q, str) and len(q) > 10:
                        all_pairs.append((q, doc))

            except Exception as e:
                logger.warning("Failed to generate questions for doc %d: %s", i, e)
                continue

        # Save to file for reuse (avoid regenerating every time)
        import json
        with open(output_file, "w") as f:
            json.dump(all_pairs, f, indent=2)

        logger.info(
            "Generated %d training pairs, saved to %s",
            len(all_pairs), output_file
        )
        return all_pairs


import json  # Needed for generate_training_data
