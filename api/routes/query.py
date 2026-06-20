"""Query endpoint with optional streaming."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.models import QueryRequest, QueryResponse, CitedChunk
from api.deps import get_pipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(body: QueryRequest, pipeline=Depends(get_pipeline)):
    if body.stream:
        # Return SSE stream of text deltas
        def event_stream():
            for chunk in pipeline.stream_answer(
                body.question,
                top_k=body.top_k,
                rewrite=body.rewrite_query,
                rerank=body.rerank,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = pipeline.query(
        question=body.question,
        top_k=body.top_k,
        rewrite=body.rewrite_query,
        rerank=body.rerank,
    )
    return QueryResponse(
        question=body.question,
        rewritten_question=result.get("rewritten_question"),
        answer=result["answer"].answer,
        citations=result["answer"].citations,
        context_chunks=[
            CitedChunk(chunk_id=c["chunk_id"], content=c["content"], score=c.get("score", 0.0))
            for c in result["chunks"]
        ],
        model=result["answer"].model,
        input_tokens=result["answer"].input_tokens,
        output_tokens=result["answer"].output_tokens,
    )
