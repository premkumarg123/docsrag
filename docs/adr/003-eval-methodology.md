# ADR 003: Evaluation Methodology

**Date:** 2026-06-20
**Status:** Accepted

## Context

RAG systems can silently regress. Retrieval can change when the corpus grows; generation can hallucinate more after a prompt change. We need automated evaluation that runs in CI.

## Decision

A two-layer evaluation approach:

### Layer 1 — Retrieval Metrics (no LLM calls)
- **Recall@k** (k=1,3,5,10): primary recall signal
- **MRR**: quality of the first hit
- **NDCG@k**: ranked relevance quality

Evaluated against a curated golden dataset of (question, relevant_chunk_ids) pairs.

### Layer 2 — Generation Metrics (LLM-as-judge)
- **Faithfulness**: are all claims traceable to the retrieved context? (0–1)
- **Relevance**: does the answer address the question? (0–1)

Claude claude-sonnet-4-6 is used as judge. Prompts return structured JSON to prevent parsing failures.

### CI Gate
Integration test `test_eval_metrics_pass_threshold` asserts:
- `avg_composite ≥ 0.60`
- `avg_recall@5 ≥ 0.50`

PRs that drop below these thresholds cannot merge.

## Alternatives Considered

| Approach | Rejected Because |
|---|---|
| Human evaluation only | Expensive, not automatable in CI |
| ROUGE/BLEU for generation | Poor correlation with human quality for abstractive QA |
| Single metric | Hides trade-offs (high faithfulness + low relevance = bad) |
| G-Eval / RAGAS | Added dependency; our bespoke judge is transparent and controllable |

## Consequences

- LLM-as-judge calls add ~$0.005 per case in API cost; acceptable at dataset sizes < 1000.
- Judge scores have ±0.05 variance across runs; thresholds include a 10% safety margin.
- Scheduled weekly eval (`eval_regression.yml`) catches corpus-driven drift not caught by unit tests.
