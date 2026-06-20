"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import ingest, query, eval as eval_route

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DocsRAG",
    description=(
        "Production RAG pipeline with hybrid search, cross-encoder reranking, "
        "citation-grounded generation, and an automated evaluation harness."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval_route.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
