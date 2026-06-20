"""Ingest endpoints: file upload and raw text."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.models import IngestResponse, DocumentListItem, IngestTextRequest
from api.deps import get_pipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/text", response_model=IngestResponse)
def ingest_text(body: IngestTextRequest, pipeline=Depends(get_pipeline)):
    """Ingest raw text directly."""
    doc_id, chunk_count = pipeline.ingest_text(
        text=body.text, name=body.name, metadata=body.metadata
    )
    return IngestResponse(
        document_id=doc_id,
        chunk_count=chunk_count,
        message=f"Ingested {chunk_count} chunks from '{body.name}'",
    )


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    pipeline=Depends(get_pipeline),
):
    """Ingest a file (PDF, Markdown, plain text)."""
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in (".pdf", ".txt", ".md", ".markdown", ".html"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Use .pdf, .txt, .md, or .html",
        )

    raw_bytes = await file.read()

    # Write to a temp file so loaders can use path-based APIs
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        doc_id, chunk_count = pipeline.ingest_file(tmp_path, name=file.filename)
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        document_id=doc_id,
        chunk_count=chunk_count,
        message=f"Ingested '{file.filename}' → {chunk_count} chunks",
    )


@router.get("/documents", response_model=list[DocumentListItem])
def list_documents(pipeline=Depends(get_pipeline)):
    docs = pipeline.vector_store.list_documents()
    return [
        DocumentListItem(
            id=d["id"],
            name=d["name"],
            source_uri=d.get("source_uri"),
            mime_type=d["mime_type"],
            created_at=str(d["created_at"]),
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int, pipeline=Depends(get_pipeline)):
    pipeline.vector_store.delete_document(doc_id)
    pipeline.rebuild_bm25()
