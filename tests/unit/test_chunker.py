"""Unit tests for the recursive chunker."""

import pytest
from ingestion.chunker import RecursiveChunker


def test_short_text_single_chunk():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk("This is a short text.")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_long_text_splits_into_multiple_chunks():
    long_text = " ".join(["word"] * 1000)
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(long_text)
    assert len(chunks) > 1


def test_chunk_indices_are_sequential():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph."
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(text)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_metadata_propagated():
    chunker = RecursiveChunker()
    meta = {"source": "test.pdf", "page": 1}
    chunks = chunker.chunk("Some text for testing metadata.", metadata=meta)
    for c in chunks:
        assert c.metadata["source"] == "test.pdf"


def test_token_count_is_positive():
    chunker = RecursiveChunker()
    chunks = chunker.chunk("Hello world this is a test sentence.")
    assert all(c.token_count > 0 for c in chunks)


def test_empty_text_returns_empty():
    chunker = RecursiveChunker()
    chunks = chunker.chunk("")
    assert chunks == []
