"""In-memory BM25 index over corpus chunks for keyword retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    chunk_id: int
    content: str
    score: float


class BM25Index:
    """
    Holds a BM25Okapi index built from a flat list of (chunk_id, text) pairs.
    Rebuilt on demand after new ingestion; kept in-memory during a server
    session. For larger corpora, this should move to a persistent store like
    Elasticsearch or PostgreSQL full-text search.
    """

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._texts: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, corpus: list[dict]) -> None:
        """corpus: list of {chunk_id: int, content: str}"""
        if not corpus:
            self._ids = []
            self._texts = []
            self._bm25 = None
            return
        self._ids = [c["chunk_id"] for c in corpus]
        self._texts = [c["content"] for c in corpus]
        tokenised = [self._tokenise(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        if self._bm25 is None or not self._ids:
            return []

        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip(self._ids, self._texts, scores),
            key=lambda x: x[2],
            reverse=True,
        )[:top_k]

        return [
            BM25Result(chunk_id=cid, content=text, score=float(score))
            for cid, text, score in ranked
            if score > 0
        ]

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return text.lower().split()
