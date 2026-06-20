"""Unit tests for BM25 and RRF fusion logic."""

import pytest
from unittest.mock import MagicMock
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_search import HybridSearcher


# ── BM25 ─────────────────────────────────────────────────────────────────────

def test_bm25_returns_relevant_result():
    idx = BM25Index()
    idx.build([
        {"chunk_id": 1, "content": "Python is a programming language"},
        {"chunk_id": 2, "content": "The weather is sunny today"},
        {"chunk_id": 3, "content": "Python decorators enable metaprogramming"},
    ])
    results = idx.search("Python programming", top_k=2)
    ids = [r.chunk_id for r in results]
    assert 1 in ids or 3 in ids


def test_bm25_empty_index_returns_empty():
    idx = BM25Index()
    assert idx.search("anything") == []


def test_bm25_zero_score_filtered():
    idx = BM25Index()
    idx.build([{"chunk_id": 1, "content": "completely unrelated text"}])
    results = idx.search("xyzzy frobozz", top_k=5)
    # BM25 should return empty for unseen terms
    assert all(r.score > 0 for r in results)


# ── RRF Fusion ───────────────────────────────────────────────────────────────

def test_rrf_combines_results():
    mock_vs = MagicMock()
    mock_bm25 = MagicMock()

    from retrieval.vector_store import ChunkResult
    from retrieval.bm25_index import BM25Result

    mock_vs.similarity_search.return_value = [
        ChunkResult(chunk_id=1, document_id=1, content="Doc A", score=0.9, metadata={}),
        ChunkResult(chunk_id=2, document_id=1, content="Doc B", score=0.7, metadata={}),
    ]
    mock_bm25.search.return_value = [
        BM25Result(chunk_id=2, content="Doc B", score=5.0),
        BM25Result(chunk_id=3, content="Doc C", score=3.0),
    ]

    searcher = HybridSearcher(mock_vs, mock_bm25)
    results = searcher.search([0.1] * 384, "test query", top_k=3)

    ids = [r.chunk_id for r in results]
    # chunk 2 appears in both lists → should rank highest
    assert ids[0] == 2
    assert len(results) <= 3
