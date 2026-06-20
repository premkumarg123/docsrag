"""LLM-based query rewriting and expansion."""

from __future__ import annotations

import anthropic

REWRITE_PROMPT = """\
You are a search query optimizer. Given a user question, rewrite it into a \
clear, dense retrieval query that maximises recall from a text corpus.

Rules:
- Expand abbreviations and resolve pronouns
- Add relevant synonyms in parentheses
- Keep it under 30 words
- Return ONLY the rewritten query, no explanation

Question: {question}
Rewritten query:"""


class QueryRewriter:
    """
    Uses Claude to produce a retrieval-optimised version of the user's question.

    This improves recall for conversational or vague queries by making
    keywords more explicit before embedding and BM25 indexing.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    def rewrite(self, question: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(question=question),
                }
            ],
        )
        block = message.content[0]
        rewritten = (block.text if hasattr(block, "text") else "").strip()  # type: ignore[union-attr]
        return rewritten if rewritten else question
