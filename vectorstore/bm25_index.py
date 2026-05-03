"""
BM25 sparse index helper. Owned by a vector store via composition.
corpus[i] must correspond to the caller's metadata[i] — positional invariant.
"""
from __future__ import annotations
import pickle
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def build(self, corpus: List[str]) -> None:
        """Rebuild the index from scratch. Caller must pass corpus in metadata order."""
        self._corpus = list(corpus)
        if not self._corpus:
            self._bm25 = None
            return
        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def upsert(self, new_texts: List[str]) -> None:
        """Append texts and rebuild. BM25Okapi has no incremental add; full rebuild required."""
        self._corpus.extend(new_texts)
        if not self._corpus:
            self._bm25 = None
            return
        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return (corpus_index, score) pairs for non-zero scores, sorted descending, up to top_k."""
        if self._bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        candidates = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def reset(self) -> None:
        self._corpus = []
        self._bm25 = None

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._corpus, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            corpus = pickle.load(f)
        self.build(corpus)
