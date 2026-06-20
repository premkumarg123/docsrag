"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def sample_texts():
    return [
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language models. "
        "It retrieves relevant passages from a corpus before generating an answer.",
        "Hybrid search merges dense vector similarity with sparse BM25 keyword matching. "
        "Reciprocal Rank Fusion combines both ranked lists without requiring score normalisation.",
        "A cross-encoder reads the query and document jointly, producing higher-precision "
        "relevance scores than bi-encoders that embed each independently.",
        "Recall@k measures what fraction of relevant documents appear in the top-k results. "
        "MRR measures the position of the first relevant result across queries.",
        "Faithfulness evaluation checks that every claim in the generated answer "
        "can be traced back to the retrieved context passages.",
    ]
