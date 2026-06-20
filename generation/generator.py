"""Citation-grounded RAG generator with streaming support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Generator, List

import anthropic

from generation.prompts import SYSTEM_PROMPT, USER_PROMPT, build_context_block


@dataclass
class GeneratedAnswer:
    answer: str
    citations: List[int]          # chunk_ids cited
    model: str
    input_tokens: int
    output_tokens: int
    context_chunks: List[dict] = field(default_factory=list)


class RAGGenerator:
    """
    Generates citation-grounded answers from retrieved chunks.

    Supports both blocking and streaming response modes.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024) -> None:
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, chunks: List[dict]) -> GeneratedAnswer:
        """Blocking generation. chunks: [{chunk_id, content}, ...]"""
        context_block = build_context_block(chunks)
        user_msg = USER_PROMPT.format(
            context_block=context_block,
            question=question,
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        answer = response.content[0].text
        citations = self._extract_citations(answer)

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            context_chunks=chunks,
        )

    def stream(
        self,
        question: str,
        chunks: List[dict],
    ) -> Generator[str, None, None]:
        """Yield text deltas for streaming responses."""
        context_block = build_context_block(chunks)
        user_msg = USER_PROMPT.format(
            context_block=context_block,
            question=question,
        )

        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    @staticmethod
    def _extract_citations(text: str) -> List[int]:
        """Pull all [N] style citation markers from the answer text."""
        return [int(m) for m in re.findall(r"\[(\d+)\]", text)]
