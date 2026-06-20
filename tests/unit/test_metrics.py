"""Unit tests for retrieval metrics."""

import pytest

from eval.metrics import mrr, ndcg_at_k, recall_at_k, retrieval_metrics


def test_recall_at_k_perfect():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_recall_at_k_zero():
    assert recall_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0


def test_recall_at_k_partial():
    score = recall_at_k([1, 4, 5], {1, 2}, k=3)
    assert score == 0.5


def test_recall_at_k_cutoff():
    # Relevant item at rank 4, but k=3 → should not count
    assert recall_at_k([4, 5, 6, 1], {1}, k=3) == 0.0


def test_mrr_first_hit():
    assert mrr([1, 2, 3], {1}) == 1.0


def test_mrr_second_hit():
    assert mrr([2, 1, 3], {1}) == pytest.approx(0.5)


def test_mrr_no_hit():
    assert mrr([4, 5, 6], {1, 2}) == 0.0


def test_ndcg_perfect():
    score = ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3)
    assert score == pytest.approx(1.0)


def test_ndcg_empty_relevant():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_retrieval_metrics_returns_all_keys():
    result = retrieval_metrics([1, 2, 3], {1}, k_values=[1, 5])
    assert "mrr" in result
    assert "recall@1" in result
    assert "recall@5" in result
    assert "ndcg@1" in result
    assert "ndcg@5" in result
