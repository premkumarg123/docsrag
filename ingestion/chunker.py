"""Recursive character-based text chunker with overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    content: str
    chunk_index: int
    token_count: int
    metadata: dict


class RecursiveChunker:
    """
    Splits text hierarchically on separators (paragraph → sentence → word)
    until each piece is under `chunk_size` tokens.

    Overlap keeps context across boundaries without re-embedding full pages.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        metadata = metadata or {}
        raw_chunks = self._split(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        return [
            Chunk(
                content=c,
                chunk_index=i,
                token_count=self._approx_tokens(c),
                metadata=metadata,
            )
            for i, c in enumerate(merged)
        ]

    # ------------------------------------------------------------------

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        sep, *rest = separators
        if sep == "":
            # Last resort: split on every character then re-join words
            words = text.split()
            return words if words else [text]

        parts = [p for p in text.split(sep) if p.strip()]

        final: List[str] = []
        for part in parts:
            if self._approx_tokens(part) <= self.chunk_size:
                final.append(part.strip())
            else:
                final.extend(self._split(part, rest))
        return final

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Greedily merge small pieces; add overlap window between chunks."""
        chunks: List[str] = []
        current_tokens = 0
        current_parts: List[str] = []

        for piece in pieces:
            piece_tokens = self._approx_tokens(piece)
            if current_tokens + piece_tokens > self.chunk_size and current_parts:
                chunks.append(" ".join(current_parts))
                # Retain tail of previous chunk as overlap
                overlap_parts = self._tail_overlap(current_parts)
                current_parts = overlap_parts
                current_tokens = sum(self._approx_tokens(p) for p in current_parts)
            current_parts.append(piece)
            current_tokens += piece_tokens

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks

    def _tail_overlap(self, parts: List[str]) -> List[str]:
        """Return the last N tokens worth of parts for overlap."""
        kept: List[str] = []
        tokens = 0
        for part in reversed(parts):
            t = self._approx_tokens(part)
            if tokens + t > self.chunk_overlap:
                break
            kept.insert(0, part)
            tokens += t
        return kept

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # Rough heuristic: 1 token ≈ 4 chars
        return max(1, len(text) // 4)
