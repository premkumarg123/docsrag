"""Pydantic request/response models for the DocsRAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to ingest")
    name: str = Field(..., description="Display name for this document")
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: int
    chunk_count: int
    message: str


class DocumentListItem(BaseModel):
    id: int
    name: str
    source_uri: str | None
    mime_type: str
    created_at: str


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    rewrite_query: bool = Field(default=True)
    rerank: bool = Field(default=True)
    stream: bool = Field(default=False)


class CitedChunk(BaseModel):
    chunk_id: int
    content: str
    score: float


class QueryResponse(BaseModel):
    question: str
    rewritten_question: str | None
    answer: str
    citations: list[int]
    context_chunks: list[CitedChunk]
    model: str
    input_tokens: int
    output_tokens: int


# ── Eval ─────────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    dataset_path: str = Field(
        default="eval/datasets/sample_qa.json",
        description="Path to a JSON eval dataset relative to project root",
    )
    run_name: str = Field(default="api-run")


class EvalResponse(BaseModel):
    run_name: str
    cases: int
    avg_recall_at_5: float
    avg_mrr: float
    avg_faithfulness: float
    avg_relevance: float
    avg_composite: float
    passed_threshold: bool
