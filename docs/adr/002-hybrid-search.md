# ADR 002: Hybrid Search with Reciprocal Rank Fusion

**Date:** 2026-06-20
**Status:** Accepted

## Context

Pure vector search is strong on semantic paraphrase but misses exact keyword matches (product names, version numbers, acronyms). Pure BM25 misses conceptual synonyms. We need both.

## Decision

Run **both** dense (pgvector cosine) and sparse (BM25) retrievers, fuse rankings with **Reciprocal Rank Fusion (RRF)** using k=60 per the original Cormack et al. (2009) paper.

Weights: `vector=0.7, bm25=0.3`. The higher vector weight reflects better baseline performance on our eval set.

## Why RRF Over Score Normalisation?

Score distributions differ across retrievers (cosine similarity is bounded [-1,1]; BM25 is unbounded). Normalisation requires knowing the max score per query, which is unstable. RRF uses only rank positions, which are stable and comparable.

## Alternatives Considered

| Approach | Rejected Because |
|---|---|
| Vector only | Misses exact-match queries (version numbers, model names) |
| BM25 only | Misses semantic paraphrase; too brittle for natural language |
| Linear score fusion | Requires empirical normalisation per dataset; fragile |
| Learned sparse (SPLADE) | High complexity; marginal gain vs. BM25 for our use case |

## Consequences

- Recall@5 improved ~15% over vector-only on our eval set.
- BM25 index is rebuilt in-memory on each ingest; at >1M chunks, this must move to Elasticsearch or PostgreSQL FTS.
- Follow-up: tune `vector_weight` and `bm25_weight` with a grid search on held-out eval queries.
