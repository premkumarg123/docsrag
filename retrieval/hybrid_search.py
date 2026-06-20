"""Reciprocal Rank Fusion over vector + BM25 results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from retrieval.vector_store import VectorStore, ChunkResult
from retrieval.bm25_index import BM25Index


@dataclass
class SearchResult:
    chunk_id: int
    content: str
    rrf_score: float
    vector_score: float
    bm25_score: float
    metadata: dict


class HybridSearcher:
    """
    Combines dense (vector) and sparse (BM25) rankings via RRF.

    RRF(d) = Σ 1 / (k + rank(d))   where k=60 per the original paper.

    Vector recall is better for semantic paraphrase; BM25 is better for
    exact keyword and rare-term queries. RRF reliably beats either alone.
    """

    RRF_K = 60

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index) -> None:
        self._vs = vector_store
        self._bm25 = bm25_index

    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        fetch_k: int = 40,
    ) -> List[SearchResult]:
        # Retrieve from both sources
        vec_results = self._vs.similarity_search(query_embedding, top_k=fetch_k)
        bm25_results = self._bm25.search(query_text, top_k=fetch_k)

        # Build rank maps  {chunk_id -> rank (1-indexed)}
        vec_rank = {r.chunk_id: i + 1 for i, r in enumerate(vec_results)}
        bm25_rank = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

        # Score lookup
        vec_score_map = {r.chunk_id: r.score for r in vec_results}
        bm25_score_map = {r.chunk_id: r.score for r in bm25_results}
        content_map = {r.chunk_id: r.content for r in vec_results}
        content_map.update({r.chunk_id: r.content for r in bm25_results})
        meta_map = {r.chunk_id: getattr(r, "metadata", {}) for r in vec_results}

        all_ids = set(vec_rank) | set(bm25_rank)

        rrf_scores: dict[int, float] = {}
        for cid in all_ids:
            vec_rrf = vector_weight / (self.RRF_K + vec_rank.get(cid, fetch_k + 1))
            bm25_rrf = bm25_weight / (self.RRF_K + bm25_rank.get(cid, fetch_k + 1))
            rrf_scores[cid] = vec_rrf + bm25_rrf

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(
                chunk_id=cid,
                content=content_map.get(cid, ""),
                rrf_score=score,
                vector_score=vec_score_map.get(cid, 0.0),
                bm25_score=bm25_score_map.get(cid, 0.0),
                metadata=meta_map.get(cid, {}),
            )
            for cid, score in ranked
        ]
