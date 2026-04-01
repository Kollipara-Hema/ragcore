"""
=============================================================================
ingestion/chunkers/chunkers.py — ALL Chunking Strategies (Single File)
=============================================================================
PURPOSE:
    Splits documents into smaller pieces before embedding and indexing.
    This is the ONLY chunking file — advanced_chunkers.py is removed.

ALL 7 STRATEGIES:

    BASIC:
    1. FixedSizeChunker         — splits every N characters. Fastest.
    2. SemanticChunker          — splits at topic boundaries. Best default.
    3. HierarchicalChunker      — parent + child chunks. Best for long PDFs.
    4. SentenceWindowChunker    — indexes sentences, stores surrounding context.

    ADVANCED (Phase 1):
    5. PropositionalChunker     — LLM extracts atomic facts. Most precise.
    6. TableAwareChunker        — detects and preserves tables intact.
    7. DocumentStructureChunker — splits at section headings.

SET IN .env:
    CHUNKING_STRATEGY=semantic        (recommended default)
    CHUNKING_STRATEGY=fixed           (fastest, good for testing)
    CHUNKING_STRATEGY=hierarchical    (best for long PDFs)
    CHUNKING_STRATEGY=sentence        (dense technical text)
    CHUNKING_STRATEGY=propositional   (precise Q&A, costs LLM calls)
    CHUNKING_STRATEGY=table_aware     (financial reports with tables)
    CHUNKING_STRATEGY=structure       (manuals, wikis with headings)

HOW TO USE:
    from ingestion.chunkers.chunkers import get_chunker
    chunker = get_chunker()               # reads CHUNKING_STRATEGY from .env
    chunker = get_chunker("table_aware")  # override directly
    chunks  = chunker.chunk(my_document)
=============================================================================
"""

from __future__ import annotations
import asyncio     # For running async LLM calls from sync context
import json        # For parsing LLM JSON responses
import logging     # For log messages
import re          # For regex pattern matching
from abc import ABC, abstractmethod
from typing import Optional
from uuid import uuid4

from utils.models import Chunk, Document
from config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# BASE INTERFACE — every chunker must implement chunk()
# =============================================================================

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into a list of Chunk objects."""
        ...


# =============================================================================
# 1. FixedSizeChunker
# =============================================================================

class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size character windows with overlap.

    EXAMPLE (chunk_size=20, overlap=5):
        "The cat sat on the mat and the dog sat too"
        Chunk 1: "The cat sat on the m"
        Chunk 2: "n the mat and the do"  ← overlaps by 5 chars
        Chunk 3: "d the dog sat too"

    BEST FOR: Quick testing, homogeneous text (logs, transcripts)
    SET:      CHUNKING_STRATEGY=fixed in .env
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Break at whitespace to avoid cutting mid-word
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    doc_id=document.doc_id,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    metadata=document.metadata.__dict__.copy(),
                ))
                idx += 1

            # Advance by chunk_size minus overlap so consecutive chunks share text
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks


# =============================================================================
# 2. SemanticChunker
# =============================================================================

class SemanticChunker(BaseChunker):
    """
    Splits text where the cosine similarity between consecutive sentences drops.
    A similarity drop = topic change = good split point.

    HOW IT WORKS:
        1. Split text into sentences
        2. Embed each sentence with a small local model (all-MiniLM-L6-v2)
        3. Measure similarity between consecutive sentences
        4. Split where similarity < breakpoint_threshold

    BEST FOR: Most documents — recommended default
    SET:      CHUNKING_STRATEGY=semantic in .env
    """

    def __init__(self, breakpoint_threshold: float = 0.85, min_chunk_size: int = 200):
        """
        Args:
            breakpoint_threshold: Split when similarity drops below this (0.0-1.0)
                                  Higher = more splits. Lower = fewer, larger chunks.
            min_chunk_size: Merge chunks smaller than this to avoid tiny noisy chunks
        """
        self.threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — falling back to FixedSizeChunker.\n"
                "Install it: pip install sentence-transformers"
            )
            return FixedSizeChunker().chunk(document)

        # Use a small fast model just for splitting decisions (not the main embedder)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sentences = self._split_sentences(document.content)

        if len(sentences) <= 1:
            return FixedSizeChunker().chunk(document)

        # Embed all sentences and compute consecutive similarities
        embeddings = model.encode(sentences, batch_size=64, normalize_embeddings=True)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

        # Find where similarity drops below threshold = split points
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        chunks: list[Chunk] = []
        for idx in range(len(breakpoints) - 1):
            start_sent = breakpoints[idx]
            end_sent = breakpoints[idx + 1]
            chunk_text = " ".join(sentences[start_sent:end_sent]).strip()

            # Merge tiny chunks into the previous one
            if len(chunk_text) < self.min_chunk_size and chunks:
                chunks[-1] = Chunk(
                    content=chunks[-1].content + " " + chunk_text,
                    doc_id=chunks[-1].doc_id,
                    chunk_index=chunks[-1].chunk_index,
                    start_char=chunks[-1].start_char,
                    end_char=chunks[-1].end_char + len(chunk_text),
                    metadata=chunks[-1].metadata,
                )
                continue

            if chunk_text:
                start_char = document.content.find(sentences[start_sent])
                chunks.append(Chunk(
                    content=chunk_text,
                    doc_id=document.doc_id,
                    chunk_index=idx,
                    start_char=max(0, start_char),
                    end_char=max(0, start_char) + len(chunk_text),
                    metadata=document.metadata.__dict__.copy(),
                ))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# 3. HierarchicalChunker
# =============================================================================

class HierarchicalChunker(BaseChunker):
    """
    Creates TWO levels of chunks from each document:

        Parent chunk (large ~1024 chars) ← stored for context, NOT embedded
        ├── Child chunk 1 (small ~256 chars) ← embedded and indexed
        ├── Child chunk 2 (small ~256 chars) ← embedded and indexed
        └── Child chunk 3 (small ~256 chars) ← embedded and indexed

    HOW RETRIEVAL USES THIS:
        Step 1: Vector search finds the most relevant CHILD chunks (precise)
        Step 2: System returns their PARENT chunks to the LLM (full context)
        Result: precise matching + rich context in the answer

    BEST FOR: Long PDFs (50+ pages), legal documents, research papers
    SET:      CHUNKING_STRATEGY=hierarchical in .env
    """

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
        overlap: int = 32,
    ):
        self.parent_chunker = FixedSizeChunker(parent_chunk_size, overlap)
        self.child_chunker = FixedSizeChunker(child_chunk_size, overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        parent_chunks = self.parent_chunker.chunk(document)

        for parent in parent_chunks:
            # Tag parent — stored for retrieval context but NOT embedded
            parent.metadata["is_parent_chunk"] = True
            all_chunks.append(parent)

            # Create child chunks from the parent text
            child_doc = Document(
                content=parent.content,
                metadata=document.metadata,
                doc_id=document.doc_id,
            )
            for child in self.child_chunker.chunk(child_doc):
                child.parent_chunk_id = parent.chunk_id
                child.metadata["is_child_chunk"] = True
                child.metadata["parent_chunk_id"] = str(parent.chunk_id)
                all_chunks.append(child)

        return all_chunks


# =============================================================================
# 4. SentenceWindowChunker
# =============================================================================

class SentenceWindowChunker(BaseChunker):
    """
    Indexes individual sentences but stores surrounding sentences as context.

    EXAMPLE (window_size=2, sentence S3 is indexed):
        Embedded (what gets searched): "S3"
        Stored in metadata (what LLM reads): "S1 S2 S3 S4 S5"

    This gives precise retrieval (sentence-level) with rich LLM context.

    BEST FOR: Dense technical text, legal documents, research papers
    SET:      CHUNKING_STRATEGY=sentence in .env
    """

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Sentences before AND after to store as context.
                        window_size=3 → 3 + sentence + 3 = 7 sentences stored
        """
        self.window = window_size

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', document.content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        chunks: list[Chunk] = []

        for i, sentence in enumerate(sentences):
            start_w = max(0, i - self.window)
            end_w = min(len(sentences), i + self.window + 1)
            window_text = " ".join(sentences[start_w:end_w])

            meta = document.metadata.__dict__.copy()
            meta["window_text"] = window_text    # Full context for LLM
            meta["sentence_index"] = i

            chunks.append(Chunk(
                content=sentence,        # Only sentence is embedded
                doc_id=document.doc_id,
                chunk_index=i,
                metadata=meta,
            ))

        return chunks


# =============================================================================
# 5. PropositionalChunker  (Phase 1)
# =============================================================================

class PropositionalChunker(BaseChunker):
    """
    Uses an LLM to break paragraphs into atomic facts (propositions).

    EXAMPLE:
        Input: "Apple was founded in 1976 by Steve Jobs and became the
                first company to reach $1T market cap in 2018."

        Output chunks:
            "Apple was founded in 1976."
            "Apple was founded by Steve Jobs."
            "Apple was the first company to reach $1 trillion market cap."
            "Apple reached $1 trillion market cap in 2018."

    Each chunk = one standalone verifiable fact.

    PROS:  Most precise retrieval possible — exact fact matching
    CONS:  Requires LLM call per paragraph (slower, costs API credits)
    BEST FOR: Knowledge bases, FAQ systems, precise factual Q&A
    SET:      CHUNKING_STRATEGY=propositional in .env

    COST ESTIMATE: ~500 tokens per paragraph → gpt-4o-mini costs ~$0.0002/paragraph
    """

    EXTRACTION_PROMPT = """Extract atomic propositions from the text.
Each proposition must be:
- One single self-contained factual statement
- Understandable without reading other propositions
- Specific enough to be searched independently

Return ONLY a JSON array of strings. No explanation, no markdown.
Example: ["The company was founded in 1995.", "Revenue grew 15% in Q3."]"""

    def __init__(self, min_proposition_length: int = 20, fallback_on_error: bool = True):
        self.min_length = min_proposition_length
        self.fallback = fallback_on_error
        # Used if LLM extraction fails
        self._fallback_chunker = SemanticChunker()

    def chunk(self, document: Document) -> list[Chunk]:
        """Bridges sync interface to async LLM calls."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an async context — create new loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_chunk(document))
                    return future.result()
            return loop.run_until_complete(self._async_chunk(document))
        except Exception:
            return asyncio.run(self._async_chunk(document))

    async def _async_chunk(self, document: Document) -> list[Chunk]:
        paragraphs = re.split(r'\n\s*\n', document.content)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]

        all_chunks: list[Chunk] = []
        chunk_index = 0

        for para in paragraphs:
            propositions = await self._extract_propositions(para)

            for prop in propositions:
                prop = prop.strip()
                if len(prop) < self.min_length:
                    continue

                meta = document.metadata.__dict__.copy()
                meta["original_paragraph"] = para[:500]
                meta["chunking_method"] = "propositional"

                all_chunks.append(Chunk(
                    content=prop,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    metadata=meta,
                ))
                chunk_index += 1

        if not all_chunks:
            logger.warning("PropositionalChunker: no propositions extracted — using fallback")
            return self._fallback_chunker.chunk(document)

        logger.info("PropositionalChunker: extracted %d propositions", len(all_chunks))
        return all_chunks

    async def _extract_propositions(self, paragraph: str) -> list[str]:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.EXTRACTION_PROMPT},
                    {"role": "user", "content": paragraph},
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            if isinstance(data, list):
                return data
            for key in ["propositions", "statements", "facts", "items"]:
                if key in data:
                    return data[key]
            return []

        except Exception as e:
            logger.warning("Proposition extraction failed: %s", e)
            if self.fallback:
                return re.split(r'(?<=[.!?])\s+', paragraph)
            return []


# =============================================================================
# 6. TableAwareChunker  (Phase 1)
# =============================================================================

class TableAwareChunker(BaseChunker):
    """
    Detects tables in documents and keeps them as single chunks.

    WHY TABLES NEED SPECIAL TREATMENT:
        A table split mid-row loses all meaning.
        "Revenue | Q1 | Q2" is useless without its data rows.
        This chunker keeps every table whole.

    SUPPORTS:
        - Markdown tables  (| col1 | col2 |)
        - HTML tables      (<table>...</table>)

    PROS:  Tables are never split — always retrieved as complete units
    CONS:  Large tables become large chunks (may approach token limits)
    BEST FOR: Financial reports, scientific papers, any doc with data tables
    SET:      CHUNKING_STRATEGY=table_aware in .env
    """

    # Matches markdown table rows: one or more lines starting and ending with |
    MARKDOWN_TABLE_PATTERN = re.compile(r'(\|.+\|[ \t]*\n)+', re.MULTILINE)

    # Matches HTML tables: everything between <table> and </table>
    HTML_TABLE_PATTERN = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE | re.DOTALL)

    def __init__(self, max_table_tokens: int = 1500):
        """
        Args:
            max_table_tokens: Tables larger than this are split into row batches
                             to prevent exceeding the LLM context window
        """
        self.max_table_tokens = max_table_tokens
        self._text_chunker = SemanticChunker()

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        all_chunks: list[Chunk] = []
        chunk_index = 0

        table_regions = self._find_table_regions(text)

        if not table_regions:
            logger.debug("No tables found — using semantic chunking")
            return self._text_chunker.chunk(document)

        logger.info("Found %d table regions in document", len(table_regions))
        last_end = 0

        for table_start, table_end, table_text in table_regions:

            # Process text before this table with semantic chunker
            pre_table_text = text[last_end:table_start].strip()
            if pre_table_text:
                temp_doc = Document(
                    content=pre_table_text,
                    metadata=document.metadata,
                    doc_id=document.doc_id,
                )
                for c in self._text_chunker.chunk(temp_doc):
                    c.chunk_index = chunk_index
                    c.metadata["chunking_method"] = "table_aware_text"
                    all_chunks.append(c)
                    chunk_index += 1

            # Process the table itself
            estimated_tokens = len(table_text) // 4  # ~4 chars per token

            if estimated_tokens <= self.max_table_tokens:
                meta = document.metadata.__dict__.copy()
                meta["is_table"] = True
                meta["chunking_method"] = "table_aware_table"
                all_chunks.append(Chunk(
                    content=table_text,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    start_char=table_start,
                    end_char=table_end,
                    metadata=meta,
                ))
                chunk_index += 1
            else:
                # Table too large — split into row groups preserving header
                for c in self._split_large_table(table_text, document, chunk_index):
                    all_chunks.append(c)
                    chunk_index += 1

            last_end = table_end

        # Process text after the last table
        remaining = text[last_end:].strip()
        if remaining:
            temp_doc = Document(
                content=remaining,
                metadata=document.metadata,
                doc_id=document.doc_id,
            )
            for c in self._text_chunker.chunk(temp_doc):
                c.chunk_index = chunk_index
                c.metadata["chunking_method"] = "table_aware_text"
                all_chunks.append(c)
                chunk_index += 1

        return all_chunks

    def _find_table_regions(self, text: str) -> list[tuple]:
        regions = []
        for match in self.MARKDOWN_TABLE_PATTERN.finditer(text):
            regions.append((match.start(), match.end(), match.group()))
        for match in self.HTML_TABLE_PATTERN.finditer(text):
            regions.append((match.start(), match.end(), match.group()))
        regions.sort(key=lambda x: x[0])
        return regions

    def _split_large_table(self, table_text: str, document: Document, start_index: int) -> list[Chunk]:
        """Split oversized tables into row batches, preserving header in each batch."""
        lines = table_text.strip().split('\n')
        if not lines:
            return []

        header = lines[0]
        separator = lines[1] if len(lines) > 1 and '---' in lines[1] else ""
        data_rows = lines[2:] if separator else lines[1:]
        chunks = []

        for i in range(0, len(data_rows), 20):
            batch = data_rows[i:i + 20]
            chunk_text = "\n".join([header, separator] + batch) if separator else "\n".join([header] + batch)
            meta = document.metadata.__dict__.copy()
            meta["is_table"] = True
            meta["is_table_split"] = True
            meta["table_row_range"] = f"{i+1}-{i+len(batch)}"
            chunks.append(Chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                chunk_index=start_index + len(chunks),
                metadata=meta,
            ))
        return chunks


# =============================================================================
# 7. DocumentStructureChunker  (Phase 1)
# =============================================================================

class DocumentStructureChunker(BaseChunker):
    """
    Splits documents at section boundaries defined by headings.

    DETECTS:
        Markdown: # Title, ## Subtitle, ### Sub-subtitle
        HTML:     <h1>, <h2>, <h3>

    EXAMPLE:
        # Introduction
        ...text...       → Chunk 1 with metadata section_title="Introduction"

        ## Methods
        ...text...       → Chunk 2 with metadata section_title="Methods"

    PROS:  Sections stay together, section title added to metadata for filtering
    CONS:  Requires well-structured documents with headings
    BEST FOR: Reports, manuals, wikis, technical documentation
    SET:      CHUNKING_STRATEGY=structure in .env
    """

    MARKDOWN_HEADING = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    HTML_HEADING = re.compile(r'<h([1-6])[^>]*>([^<]+)</h\1>', re.IGNORECASE)

    def __init__(self, max_section_chars: int = 2000):
        """
        Args:
            max_section_chars: Sections larger than this are split further
                              using SemanticChunker to avoid huge chunks
        """
        self.max_section_chars = max_section_chars
        self._overflow_chunker = SemanticChunker()

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._extract_sections(document.content)

        if not sections:
            logger.debug("No headings found — using semantic chunking")
            return self._overflow_chunker.chunk(document)

        logger.info("Found %d sections in document", len(sections))
        all_chunks: list[Chunk] = []
        chunk_index = 0

        for section_title, section_text, heading_level in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            if len(section_text) <= self.max_section_chars:
                # Section fits in one chunk — keep together
                meta = document.metadata.__dict__.copy()
                meta["section_title"] = section_title
                meta["heading_level"] = heading_level
                meta["chunking_method"] = "structure"
                all_chunks.append(Chunk(
                    content=section_text,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    metadata=meta,
                ))
                chunk_index += 1
            else:
                # Section too large — split further
                temp_doc = Document(
                    content=section_text,
                    metadata=document.metadata,
                    doc_id=document.doc_id,
                )
                for sub in self._overflow_chunker.chunk(temp_doc):
                    sub.chunk_index = chunk_index
                    sub.metadata["section_title"] = section_title
                    sub.metadata["heading_level"] = heading_level
                    sub.metadata["chunking_method"] = "structure_split"
                    all_chunks.append(sub)
                    chunk_index += 1

        return all_chunks

    def _extract_sections(self, text: str) -> list[tuple]:
        """Returns list of (title, content, level) tuples."""
        # Try markdown headings first
        matches = list(self.MARKDOWN_HEADING.finditer(text))
        if matches:
            sections = []
            for i, match in enumerate(matches):
                level = len(match.group(1))
                title = match.group(2).strip()
                content_start = match.end()
                content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((title, text[content_start:content_end].strip(), level))
            return sections

        # Try HTML headings
        matches = list(self.HTML_HEADING.finditer(text))
        if matches:
            sections = []
            for i, match in enumerate(matches):
                level = int(match.group(1))
                title = match.group(2).strip()
                content_start = match.end()
                content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((title, text[content_start:content_end].strip(), level))
            return sections

        return []


# =============================================================================
# FACTORY — get the right chunker from .env setting or direct argument
# =============================================================================

def get_chunker(strategy: str = None) -> BaseChunker:
    """
    Returns the correct chunker based on strategy name.

    Args:
        strategy: One of the 7 strategy names, or None to read from .env

    Returns:
        An instance of the appropriate BaseChunker subclass

    Raises:
        ValueError: If strategy name is not recognised
    """
    strategy = strategy or settings.chunking_strategy

    mapping = {
        # Basic strategies
        "fixed":          FixedSizeChunker,
        "semantic":       SemanticChunker,
        "hierarchical":   HierarchicalChunker,
        "sentence":       SentenceWindowChunker,
        # Phase 1 advanced strategies
        "propositional":  PropositionalChunker,
        "table_aware":    TableAwareChunker,
        "structure":      DocumentStructureChunker,
    }

    cls = mapping.get(strategy)
    if not cls:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Choose from: {list(mapping.keys())}"
        )

    return cls()
