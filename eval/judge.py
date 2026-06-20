"""LLM-as-judge: faithfulness and answer relevance scoring via Claude."""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic


FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Score whether the ANSWER is faithfully grounded \
in the CONTEXT (no hallucinated facts outside the context).

CONTEXT:
{context}

ANSWER:
{answer}

Respond with JSON only:
{{"score": <0.0 to 1.0>, "reason": "<one sentence>"}}

score=1.0 means every claim is traceable to the context.
score=0.0 means the answer contains fabricated information."""

RELEVANCE_PROMPT = """\
You are an evaluation judge. Score whether the ANSWER actually addresses the QUESTION.

QUESTION: {question}
ANSWER: {answer}

Respond with JSON only:
{{"score": <0.0 to 1.0>, "reason": "<one sentence>"}}

score=1.0 means the answer fully and directly answers the question.
score=0.0 means the answer is completely off-topic."""


@dataclass
class JudgeScore:
    score: float
    reason: str


class LLMJudge:
    """
    Uses Claude as a judge to score two axes:
      faithfulness  — are all claims grounded in the retrieved context?
      relevance     — does the answer address the question?

    Both are scored 0–1 and returned as JudgeScore objects.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic()
        self.model = model

    def score_faithfulness(self, answer: str, context: str) -> JudgeScore:
        return self._judge(FAITHFULNESS_PROMPT.format(context=context, answer=answer))

    def score_relevance(self, question: str, answer: str) -> JudgeScore:
        return self._judge(RELEVANCE_PROMPT.format(question=question, answer=answer))

    def evaluate(
        self, question: str, answer: str, context: str
    ) -> dict:
        faith = self.score_faithfulness(answer, context)
        relev = self.score_relevance(question, answer)
        return {
            "faithfulness": {"score": faith.score, "reason": faith.reason},
            "relevance": {"score": relev.score, "reason": relev.reason},
            "composite": round((faith.score + relev.score) / 2, 4),
        }

    def _judge(self, prompt: str) -> JudgeScore:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
            return JudgeScore(score=float(data["score"]), reason=data.get("reason", ""))
        except (json.JSONDecodeError, KeyError, ValueError):
            # Best-effort extraction if model wraps JSON in markdown
            import re
            match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            score = float(match.group(1)) if match else 0.0
            return JudgeScore(score=score, reason=raw[:200])
