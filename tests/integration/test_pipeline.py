"""
Integration tests — require a running PostgreSQL + pgvector database.
Set DATABASE_URL env var before running:

  DATABASE_URL=postgresql://docsrag:docsrag@localhost:5432/docsrag pytest tests/integration/
"""

import os
import pytest

# Skip entire module if no DATABASE_URL is set (CI without DB service)
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping integration tests",
)


@pytest.fixture(scope="module")
def pipeline():
    from pipeline import RAGPipeline
    dsn = os.environ["DATABASE_URL"]
    p = RAGPipeline(dsn=dsn)
    p.startup()
    yield p
    p.shutdown()


def test_ingest_and_retrieve(pipeline):
    doc_id, chunk_count = pipeline.ingest_text(
        text=(
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with large language model generation to produce "
            "grounded, citation-backed answers from a private document corpus."
        ),
        name="rag_intro.txt",
    )
    assert doc_id > 0
    assert chunk_count >= 1

    chunks = pipeline.retrieve("What is RAG?", rewrite=False, rerank=False)
    assert len(chunks) >= 1
    assert any("retrieval" in c["content"].lower() or "rag" in c["content"].lower() for c in chunks)

    # Cleanup
    pipeline.vector_store.delete_document(doc_id)


def test_full_query_pipeline(pipeline):
    doc_id, _ = pipeline.ingest_text(
        text=(
            "Hybrid search combines dense vector similarity with BM25 sparse retrieval. "
            "Reciprocal Rank Fusion merges the two ranked lists without requiring "
            "score normalisation, consistently outperforming either method alone."
        ),
        name="hybrid_search.txt",
    )

    result = pipeline.query("How does hybrid search work?", rewrite=False, rerank=False)
    assert "answer" in result
    assert len(result["chunks"]) >= 1

    pipeline.vector_store.delete_document(doc_id)


def test_eval_metrics_pass_threshold(pipeline):
    """Regression gate: composite score must stay above 0.6 on sample dataset."""
    from eval.harness import EvalHarness
    from eval.judge import LLMJudge

    # Seed docs matching the sample QA dataset questions
    doc_texts = [
        "RAG combines retrieval with generation to produce grounded answers from documents.",
        "Hybrid search uses BM25 and vector similarity with RRF fusion for better recall.",
        "Cross-encoders read query and document jointly, giving higher precision than bi-encoders.",
        "Evaluation harnesses measure recall, MRR, faithfulness, and relevance automatically.",
        "Recall@k, MRR, and NDCG@k are the primary retrieval quality metrics.",
    ]
    doc_ids = []
    for i, text in enumerate(doc_texts):
        doc_id, _ = pipeline.ingest_text(text, name=f"seed_{i}.txt")
        doc_ids.append(doc_id)

    harness = EvalHarness(pipeline=pipeline, judge=LLMJudge())
    cases = harness.load_dataset("eval/datasets/sample_qa.json")
    report = harness.run(cases, run_name="regression")

    for doc_id in doc_ids:
        pipeline.vector_store.delete_document(doc_id)

    assert report.avg_composite >= 0.60, (
        f"Composite score {report.avg_composite:.2f} below threshold 0.60 — "
        "retrieval or generation quality has regressed."
    )
