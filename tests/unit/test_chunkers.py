"""Unit tests for chunkers."""
from __future__ import annotations

import signal

import pytest

from ingestion.chunkers.chunkers import FixedSizeChunker
from utils.models import Document, DocumentMetadata


def _doc(text: str) -> Document:
    return Document(content=text, metadata=DocumentMetadata(source="test", doc_type="txt"))


class _Watchdog:
    """SIGALRM-based timeout. Prevents a test regression from hanging the suite."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def __enter__(self) -> "_Watchdog":
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc_info) -> None:
        signal.alarm(0)

    @staticmethod
    def _handler(_signum, _frame):
        raise TimeoutError("watchdog elapsed")


def test_fixed_size_chunker_handles_dense_no_space_run():
    """
    Regression: FixedSizeChunker infinite-looped when the only spaces in a
    chunk_size window fell within the first chunk_overlap chars. rfind would
    return a boundary <= start + chunk_overlap, end was reset to that boundary,
    and new_start = end - chunk_overlap collapsed back to start.

    Fixture: a single space at exactly position chunk_overlap. Pre-fix this
    hangs; the watchdog catches a regression rather than hanging the suite.
    """
    chunk_size, chunk_overlap = 512, 64
    # space at position chunk_overlap (64); next space well outside the window
    text = "x" * chunk_overlap + " " + "y" * (chunk_size - chunk_overlap - 1) + "z" * 200

    with _Watchdog(seconds=5):
        chunks = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).chunk(_doc(text))

    assert len(chunks) > 0
    assert all(len(c.content) > 0 for c in chunks), "no empty chunks"
    assert all(len(c.content) <= chunk_size for c in chunks), "no chunk exceeds chunk_size"

    # Coverage: the union of [start_char, end_char) ranges must cover [0, len(text)).
    ordered = sorted(chunks, key=lambda c: c.start_char)
    assert ordered[0].start_char == 0, "first chunk must start at 0"
    assert ordered[-1].end_char == len(text), "last chunk must end at len(text)"
    for prev, nxt in zip(ordered, ordered[1:]):
        assert nxt.start_char <= prev.end_char, (
            f"gap between chunks: prev.end_char={prev.end_char} nxt.start_char={nxt.start_char}"
        )


def test_fixed_size_chunker_preserves_word_boundary_when_room_exists():
    """
    Sanity: snap-to-space still fires when a usable boundary exists past
    start + chunk_overlap. Each non-final chunk ends with a complete word.
    """
    text = " ".join(["hello", "world"] * 200)  # spaces every 6 chars
    chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=20).chunk(_doc(text))

    assert len(chunks) > 1
    for c in chunks[:-1]:
        # Last token of each non-final chunk must be a complete word.
        last_token = c.content.rstrip().split()[-1]
        assert last_token in ("hello", "world"), f"chunk ended mid-word: {c.content[-20:]!r}"


def test_fixed_size_chunker_short_input_single_chunk():
    """Edge case the previous hang masked: input shorter than chunk_size produces one chunk."""
    text = "the quick brown fox jumps over the lazy dog"
    chunks = FixedSizeChunker(chunk_size=512, chunk_overlap=64).chunk(_doc(text))
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)
