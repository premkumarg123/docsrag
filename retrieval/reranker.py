"""Cross-encoder reranker for precision refinement after recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from retrieval.hybrid_search import SearchResult


@dataclass
class RerankedResult:
    chunk_id: int
    content: str
    rerank_score: float
    original_rrf_score: float
    metadata: dict


class CrossEncoderReranker:
    """
    Re-scores (query, chunk) pairs with a cross-encoder, which reads both
    texts jointly and outperforms bi-encoder similarity for precision.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~66MB, runs on CPU).
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # lazy-load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[RerankedResult]:
        if not results:
            return []

        pairs = [(query, r.content) for r in results]
        scores: List[float] = self.model.predict(pairs).tolist()

        ranked = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            RerankedResult(
                chunk_id=r.chunk_id,
                content=r.content,
                rerank_score=float(score),
                original_rrf_score=r.rrf_score,
                metadata=r.metadata,
            )
            for r, score in ranked
        ]
