"""
Retrieval quality metrics: Recall@k, MRR, NDCG@k.

All functions take:
  retrieved_ids  — ordered list of chunk IDs returned by the retriever
  relevant_ids   — set of gold chunk IDs for the query
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant chunks found in top-k results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean Reciprocal Rank — position of the first relevant result."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain @ k (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k_values: list[int] | None = None,
) -> dict:
    """Return a dict of all retrieval metrics for one query."""
    k_values = k_values or [1, 3, 5, 10]
    result: dict = {"mrr": mrr(retrieved_ids, relevant_ids)}
    for k in k_values:
        result[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return result
