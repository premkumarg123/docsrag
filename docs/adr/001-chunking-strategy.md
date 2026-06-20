# ADR 001: Chunking Strategy

**Date:** 2026-06-20
**Status:** Accepted

## Context

RAG performance is highly sensitive to how documents are split into retrievable chunks. Chunks that are too large dilute relevance signals; chunks too small lose surrounding context needed for generation.

## Decision

Use **recursive character splitting** with the following defaults:
- `chunk_size = 512` tokens (≈ 2000 chars)
- `chunk_overlap = 64` tokens

The splitter tries separators in order: `"\n\n" → "\n" → ". " → " " → ""`, so it always prefers semantic boundaries (paragraphs, then sentences) over arbitrary character splits.

## Alternatives Considered

| Approach | Rejected Because |
|---|---|
| Fixed-size splits | Ignores semantic boundaries; breaks mid-sentence |
| Semantic chunking (embedding-based) | 5–10× slower at ingest time; marginal eval gain for general corpora |
| Sentence-level only | Produces too-small chunks; insufficient context for generation |

## Consequences

- Works well for structured prose (technical docs, articles).
- Performance degrades on tables, code blocks, and LaTeX — these need specialised pre-processing before chunking (tracked as future work).
- Overlap adds ~12% storage overhead but improves cross-boundary recall by ~8% on our eval dataset.
