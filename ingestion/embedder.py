"""Sentence-transformer embeddings (local, no API key required)."""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np


class Embedder:
    """
    Wraps sentence-transformers' all-MiniLM-L6-v2 (384-dim).

    Model is downloaded once and cached in ~/.cache/huggingface.
    Batching is used to saturate CPU/GPU during ingestion.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # lazy-load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return L2-normalised embeddings as plain Python lists."""
        if not texts:
            return []
        vecs: np.ndarray = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vecs.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
