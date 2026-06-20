#!/usr/bin/env python3
"""
Seed the database with sample documents about RAG, retrieval, and evaluation.
Useful for local development and testing the query endpoint immediately.

Usage:
  DATABASE_URL=postgresql://docsrag:docsrag@localhost:5432/docsrag \
  ANTHROPIC_API_KEY=sk-ant-... \
  python scripts/seed_sample_docs.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_DOCUMENTS = [
    {
        "name": "rag_overview.txt",
        "content": (
            "Retrieval-Augmented Generation (RAG) is an AI architecture that combines "
            "information retrieval with large language model generation. Instead of relying "
            "solely on parametric knowledge baked into model weights, RAG retrieves relevant "
            "passages from a document corpus at query time and injects them as context before "
            "generating an answer. This approach reduces hallucinations, enables up-to-date "
            "knowledge without retraining, and provides citation-backed responses."
        ),
    },
    {
        "name": "hybrid_search.txt",
        "content": (
            "Hybrid search combines dense vector retrieval with sparse BM25 keyword matching. "
            "Dense retrieval uses embedding models to find semantically similar passages even "
            "when query terms differ from document terms. BM25 (Best Matching 25) uses term "
            "frequency and inverse document frequency to score exact keyword matches. "
            "Reciprocal Rank Fusion (RRF) merges ranked lists from both methods without "
            "requiring score normalisation, consistently outperforming either method alone. "
            "RRF score = Σ 1/(k + rank) where k=60 per the original paper."
        ),
    },
    {
        "name": "reranking.txt",
        "content": (
            "Cross-encoder rerankers improve retrieval precision by reading the query and "
            "document jointly in a single forward pass. Unlike bi-encoders that embed query "
            "and document independently and compare via dot product, cross-encoders attend "
            "across both texts simultaneously, capturing nuanced relevance signals. "
            "The trade-off is cost: cross-encoders are 100x slower per comparison, so they "
            "are applied only to a short-list of candidates retrieved by faster methods."
        ),
    },
    {
        "name": "eval_methodology.txt",
        "content": (
            "Evaluation is the most underinvested part of most RAG systems. Without automated "
            "evaluation, retrieval quality and generation faithfulness degrade silently. "
            "Key retrieval metrics include Recall@k (what fraction of relevant chunks appear "
            "in top-k results), MRR (position of the first relevant result), and NDCG@k "
            "(normalised discounted cumulative gain). Generation is evaluated with LLM-as-judge: "
            "faithfulness measures whether every claim is grounded in the retrieved context, "
            "and relevance measures whether the answer addresses the question. CI gates prevent "
            "merges that cause metric regressions."
        ),
    },
    {
        "name": "chunking_strategy.txt",
        "content": (
            "Chunking strategy significantly affects RAG quality. Chunk size trades off "
            "precision (smaller chunks, more targeted) against context (larger chunks, more "
            "coherent). Recursive character splitting tries paragraph breaks first, then "
            "sentence breaks, then word breaks — preserving semantic units where possible. "
            "Overlap between chunks ensures context is not lost at boundaries. Typical "
            "production settings: 512 token chunks with 64 token overlap for general corpora. "
            "Semantic chunking using embedding similarity between sentences is an alternative "
            "that produces more coherent chunks but is slower to compute."
        ),
    },
]


def main():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    from pipeline import RAGPipeline

    pipeline = RAGPipeline(dsn=dsn)
    pipeline.startup()

    print(f"Seeding {len(SAMPLE_DOCUMENTS)} documents...")
    for doc in SAMPLE_DOCUMENTS:
        doc_id, chunk_count = pipeline.ingest_text(
            text=doc["content"], name=doc["name"]
        )
        print(f"  [{doc_id}] {doc['name']} → {chunk_count} chunks")

    print("\nDone! You can now query via:")
    print('  curl -X POST http://localhost:8000/query \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"question": "What is RAG and how does it work?"}\'')

    pipeline.shutdown()


if __name__ == "__main__":
    main()
