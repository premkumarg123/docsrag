"""FastAPI dependency injection — shared pipeline singleton."""

from __future__ import annotations

import os
from functools import lru_cache

from pipeline import RAGPipeline


@lru_cache(maxsize=1)
def _build_pipeline() -> RAGPipeline:
    dsn = os.environ["DATABASE_URL"]
    pipeline = RAGPipeline(dsn=dsn)
    pipeline.startup()
    return pipeline


def get_pipeline() -> RAGPipeline:
    return _build_pipeline()
