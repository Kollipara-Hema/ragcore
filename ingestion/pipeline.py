"""
Ingestion pipeline orchestrator.

Sync path (small docs): load → clean → chunk → embed → index
Async path (large docs): Celery task queue for background processing

Both paths produce the same result; async allows upload to return immediately
with a job_id while processing continues in background.
"""
from __future__ import annotations
import asyncio
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional
from uuid import uuid4

from config.settings import settings
from ingestion.loaders.document_loaders import loader_registry
from ingestion.chunkers.chunkers import get_chunker
from embeddings.embedder import get_embedder
from vectorstore.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning utilities
# ─────────────────────────────────────────────────────────────────────────────

class TextCleaner:
    """Cleans and normalizes extracted text before chunking."""

    def clean(self, text: str) -> str:
        # Normalize unicode (e.g., curly quotes → straight)
        text = unicodedata.normalize("NFKC", text)
        # Remove null bytes and control chars (except newlines/tabs)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse excessive blank lines (>2 → 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse excessive spaces within lines
        text = re.sub(r"[ \t]{3,}", "  ", text)
        # Remove header/footer boilerplate (page numbers, running heads)
        text = re.sub(r"\n\s*(Page \d+|^\d+$)\s*\n", "\n", text, flags=re.MULTILINE)
        return text.strip()

    def detect_language(self, text: str) -> str:
        """Simple language detection. Replace with langdetect for production."""
        try:
            from langdetect import detect
            return detect(text[:500])
        except Exception:
            return "en"

    def is_low_quality(self, text: str, min_length: int = 50) -> bool:
        """Flag chunks/docs that are too short or garbled to be useful."""
        if len(text.strip()) < min_length:
            return True
        # Check for suspiciously high non-alphanumeric ratio (garbled OCR)
        alphanumeric = sum(1 for c in text if c.isalnum())
        if len(text) > 0 and alphanumeric / len(text) < 0.3:
            return True
        return False


cleaner = TextCleaner()


# ─────────────────────────────────────────────────────────────────────────────
# Sync ingestion pipeline (small docs, direct call)
# ─────────────────────────────────────────────────────────────────────────────

class IngestionPipeline:
    def __init__(
        self,
        chunking_strategy: Optional[str] = None,
        embedder=None,
        vector_store=None,
    ):
        self.chunker = get_chunker(chunking_strategy)
        self.embedder = embedder or get_embedder()
        self.store = vector_store or get_vector_store()

    async def ingest_file(
        self,
        file_path: str | Path,
        metadata_overrides: Optional[dict] = None,
    ) -> tuple[str, int]:
        """
        Full ingestion pipeline for a single file.
        Returns (doc_id, chunk_count).
        """
        logger.info("Ingesting file: %s", file_path)

        # Step 1: Load
        docs = loader_registry.load(file_path, **(metadata_overrides or {}))
        if not docs:
            raise ValueError(f"No content extracted from {file_path}")

        total_chunks = 0
        doc_id = None

        for doc in docs:
            doc_id = str(doc.doc_id)

            # Step 2: Clean
            doc.content = cleaner.clean(doc.content)
            if cleaner.is_low_quality(doc.content):
                logger.warning("Low quality content detected in %s; skipping.", file_path)
                continue

            doc.metadata.language = cleaner.detect_language(doc.content)

            # Apply any metadata overrides
            if metadata_overrides:
                for key, val in metadata_overrides.items():
                    if hasattr(doc.metadata, key):
                        setattr(doc.metadata, key, val)
                    else:
                        doc.metadata.custom[key] = val

            # Step 3: Chunk
            chunks = self.chunker.chunk(doc)
            logger.info("Created %d chunks from %s", len(chunks), file_path)

            # Filter low-quality chunks
            chunks = [c for c in chunks if not cleaner.is_low_quality(c.content, min_length=30)]

            # Step 4: Embed (only child chunks or all if not hierarchical)
            chunks_to_embed = [
                c for c in chunks
                if c.metadata.get("is_child_chunk", True) and not c.metadata.get("is_parent_chunk")
            ]
            if not chunks_to_embed:
                chunks_to_embed = chunks  # embed all if no parent/child distinction

            await self.embedder.embed_chunks(chunks_to_embed)

            # Step 5: Index
            indexed = await self.store.upsert(chunks_to_embed)
            total_chunks += indexed
            logger.info("Indexed %d chunks for doc %s", indexed, doc_id)

        return doc_id or str(uuid4()), total_chunks

    async def ingest_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> tuple[str, int]:
        """Ingest raw text directly (no file loading needed)."""
        from utils.models import Document, DocumentMetadata
        meta = DocumentMetadata(
            source="direct_input",
            doc_type="txt",
            title=metadata.get("title", "Untitled") if metadata else "Untitled",
            custom=metadata or {},
        )
        doc = Document(content=text, metadata=meta)
        doc.content = cleaner.clean(doc.content)
        chunks = self.chunker.chunk(doc)
        chunks = [c for c in chunks if not cleaner.is_low_quality(c.content)]
        await self.embedder.embed_chunks(chunks)
        indexed = await self.store.upsert(chunks)
        return str(doc.doc_id), indexed


# ─────────────────────────────────────────────────────────────────────────────
# Celery tasks (async ingestion for large files)
# ─────────────────────────────────────────────────────────────────────────────

def make_celery_app():
    try:
        from celery import Celery
        app = Celery(
            "rag_ingestion",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend,
        )
        app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            worker_concurrency=settings.ingestion_concurrency,
            task_acks_late=True,
            worker_prefetch_multiplier=1,
        )
        return app
    except ImportError:
        logger.warning("Celery not installed; async ingestion unavailable.")
        return None


celery_app = make_celery_app()


if celery_app:
    @celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
    def ingest_file_task(self, file_path: str, metadata: Optional[dict] = None):
        """
        Celery task for background ingestion.
        Call via: ingest_file_task.delay(file_path, metadata)
        """
        try:
            pipeline = IngestionPipeline()
            doc_id, count = asyncio.run(pipeline.ingest_file(file_path, metadata))
            logger.info("Task completed: doc_id=%s, chunks=%d", doc_id, count)
            return {"doc_id": doc_id, "chunk_count": count, "status": "indexed"}
        except Exception as exc:
            logger.error("Ingestion task failed: %s", exc)
            self.retry(exc=exc)
else:
    # Fallback when Celery is not available
    ingest_file_task = None
