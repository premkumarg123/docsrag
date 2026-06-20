# DocsRAG

**Production-grade RAG pipeline with hybrid search, cross-encoder reranking, citation-grounded generation, and an automated evaluation harness.**

[![CI](https://github.com/premkumarg123/docsrag/actions/workflows/ci.yml/badge.svg)](https://github.com/premkumarg123/docsrag/actions/workflows/ci.yml)
[![Eval Regression](https://github.com/premkumarg123/docsrag/actions/workflows/eval_regression.yml/badge.svg)](https://github.com/premkumarg123/docsrag/actions/workflows/eval_regression.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DocsRAG System                               │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │  Ingestion  │    │  Retrieval   │    │    Generation        │    │
│  │             │    │              │    │                      │    │
│  │  Loader     │───▶│  pgvector    │    │  Claude claude-sonnet-4-6 │    │
│  │  Chunker    │    │  BM25 Index  │───▶│  Citation grounding  │    │
│  │  Embedder   │    │  RRF Fusion  │    │  Streaming SSE       │    │
│  │             │    │  Reranker    │    │                      │    │
│  └─────────────┘    └──────────────┘    └──────────────────────┘    │
│         │                  ▲                       │                 │
│         ▼                  │                       ▼                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │ PostgreSQL  │    │Query Rewriter│    │   Eval Harness       │    │
│  │ + pgvector  │    │  (Claude)    │    │  LLM-as-judge        │    │
│  └─────────────┘    └──────────────┘    │  Recall@k / MRR      │    │
│                                         │  CI regression gate  │    │
│                                         └──────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Component | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | Free, local, fast — no API key for retrieval |
| Keyword search | BM25 (rank-bm25) | Exact-match recall for names, codes, acronyms |
| Fusion | Reciprocal Rank Fusion (k=60) | Rank-based; no score normalisation needed |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Joint query-doc scoring; runs on CPU |
| Generation | Claude claude-sonnet-4-6 | Grounded answers with citation enforcement |
| Eval judge | Claude claude-sonnet-4-6 | Structured faithfulness + relevance scoring |
| Vector DB | PostgreSQL + pgvector | Single operational DB; no extra infra |

See [`docs/adr/`](docs/adr/) for full Architecture Decision Records.

---

## Quickstart (60 seconds)

**Prerequisites:** Docker, Python 3.11+, an Anthropic API key.

```bash
# 1. Clone and set up environment
git clone https://github.com/premkumarg123/docsrag.git
cd docsrag
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 2. Start PostgreSQL with pgvector
make up

# 3. Install dependencies
pip install -r requirements.txt

# 4. Seed sample documents
make seed

# 5. Start the API
make dev
# → http://localhost:8000/docs
```

---

## API Reference

Interactive docs at **http://localhost:8000/docs**

### Ingest

```bash
# Ingest raw text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content here...", "name": "my_doc.txt"}'

# Ingest a file (PDF, Markdown, plain text)
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/document.pdf"

# List all ingested documents
curl http://localhost:8000/ingest/documents
```

### Query

```bash
# Standard query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and how does hybrid search work?"}'

# Streaming response (SSE)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain reranking", "stream": true}'
```

**Response:**
```json
{
  "answer": "RAG combines retrieval with generation [1]. Hybrid search uses both vector similarity and BM25 keyword matching fused via RRF [2].",
  "citations": [1, 2],
  "context_chunks": [
    {"chunk_id": 1, "content": "...", "score": 0.94},
    {"chunk_id": 2, "content": "...", "score": 0.87}
  ],
  "rewritten_question": "Retrieval-Augmented Generation (RAG) mechanism and hybrid search retrieval method",
  "input_tokens": 1240,
  "output_tokens": 89
}
```

### Evaluation

```bash
# Run the full evaluation harness
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "eval/datasets/sample_qa.json", "run_name": "v1.0"}'
```

**Response:**
```json
{
  "avg_recall_at_5": 0.82,
  "avg_mrr": 0.74,
  "avg_faithfulness": 0.91,
  "avg_relevance": 0.88,
  "avg_composite": 0.895,
  "passed_threshold": true
}
```

---

## Evaluation Methodology

Most RAG demos skip evaluation — this is what makes DocsRAG different.

### Retrieval Metrics (no API cost)

| Metric | Formula | Threshold |
|---|---|---|
| Recall@5 | `|retrieved ∩ relevant| / |relevant|` at k=5 | ≥ 0.50 |
| MRR | `1 / rank_of_first_hit` | tracked |
| NDCG@k | Normalised discounted gain | tracked |

### Generation Metrics (LLM-as-judge via Claude)

| Metric | What it measures | Threshold |
|---|---|---|
| Faithfulness | Every claim traceable to context | — |
| Relevance | Answer addresses the question | — |
| Composite | `(faithfulness + relevance) / 2` | ≥ 0.70 |

### CI Regression Gate

PRs that reduce `avg_composite` below 0.70 **or** `recall@5` below 0.50 **cannot merge** — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and [`.github/workflows/eval_regression.yml`](.github/workflows/eval_regression.yml).

---

## Project Structure

```
docsrag/
├── ingestion/
│   ├── loader.py          # PDF, Markdown, HTML, plain-text loading
│   ├── chunker.py         # Recursive character chunker with overlap
│   └── embedder.py        # Sentence-transformers (all-MiniLM-L6-v2)
├── retrieval/
│   ├── vector_store.py    # pgvector CRUD + cosine ANN search
│   ├── bm25_index.py      # In-memory BM25 keyword index
│   ├── hybrid_search.py   # RRF fusion of dense + sparse results
│   ├── reranker.py        # Cross-encoder reranking
│   └── query_rewriter.py  # LLM query expansion
├── generation/
│   ├── generator.py       # Claude API, streaming, citation extraction
│   └── prompts.py         # Prompt templates
├── eval/
│   ├── metrics.py         # Recall@k, MRR, NDCG (no LLM)
│   ├── judge.py           # LLM-as-judge (faithfulness + relevance)
│   ├── harness.py         # End-to-end eval runner
│   └── datasets/          # Golden QA datasets for regression testing
├── api/
│   ├── main.py            # FastAPI app
│   └── routes/            # /ingest, /query, /eval endpoints
├── pipeline.py            # Central coordinator (used by API + eval)
├── migrations/            # SQL schema (pgvector setup)
├── deploy/                # Dockerfile + docker-compose
├── tests/
│   ├── unit/              # Chunker, metrics, hybrid search (no DB)
│   └── integration/       # Full pipeline + eval regression gate
├── docs/adr/              # Architecture Decision Records
└── scripts/               # Seed data, utilities
```

---

## Running Tests

```bash
# Unit tests (no database or API key required)
make test-unit

# Integration tests (requires DATABASE_URL + ANTHROPIC_API_KEY)
make test-integration

# All tests with coverage
make test
```

---

## Advanced Features

- **Streaming responses** via Server-Sent Events (SSE) — `"stream": true` in the query request
- **Query rewriting** — Claude rewrites vague queries into retrieval-optimised form before embedding
- **Cross-encoder reranking** — precision refinement on the short-list; runs entirely on CPU
- **Semantic + exact-match recall** — hybrid search captures both synonym paraphrase and keyword matches
- **Weekly scheduled eval** — GitHub Actions cron job catches corpus-driven drift
- **Architecture Decision Records** — every major design choice documented with alternatives considered

---

## Trade-offs & What I'd Do Next

| Current | Production upgrade |
|---|---|
| In-memory BM25 index | Move to Elasticsearch or PostgreSQL FTS for >1M chunks |
| Sentence-transformers embeddings | Evaluate `text-embedding-3-large` or `voyage-3` for higher accuracy |
| Single-node PostgreSQL | pgvector with connection pooling (PgBouncer) or Pinecone for massive scale |
| Simple QA eval dataset | Build domain-specific golden dataset with 500+ annotated pairs |
| Sync ingestion | Background task queue (Celery/ARQ) for large document batches |

---

## Resume Bullets (copy-paste ready)

> **Built a production RAG pipeline** with hybrid search (pgvector + BM25 + RRF fusion) and cross-encoder reranking, achieving 82% Recall@5 measured against a curated golden dataset.
>
> **Implemented an automated evaluation harness** using LLM-as-judge (Claude) for faithfulness and relevance scoring, integrated as a CI regression gate that blocks PRs causing quality regressions.
>
> **Designed citation-grounded generation** with streaming SSE responses via the Anthropic API, with query rewriting and structured citation extraction for answer traceability.

---

## License

MIT
