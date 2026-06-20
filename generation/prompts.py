"""Prompt templates for generation and evaluation."""

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer using ONLY the provided \
context passages. Cite each claim with [chunk_id] immediately after it.

Rules:
- If the context is insufficient, say: "The provided documents do not contain \
  enough information to answer this question."
- Never fabricate facts outside the context.
- Be concise and direct.
- Cite every factual sentence."""

USER_PROMPT = """\
Context passages:
{context_block}

---
Question: {question}

Answer (with citations):"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks for injection into the prompt."""
    parts = []
    for c in chunks:
        parts.append(f"[{c['chunk_id']}] {c['content'].strip()}")
    return "\n\n".join(parts)
