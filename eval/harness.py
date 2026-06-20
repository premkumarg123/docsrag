"""End-to-end evaluation harness: retrieval metrics + LLM-as-judge."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from eval.metrics import retrieval_metrics
from eval.judge import LLMJudge


@dataclass
class EvalCase:
    question: str
    relevant_chunk_ids: List[int]
    reference_answer: str = ""


@dataclass
class EvalResult:
    question: str
    retrieval: dict
    generation: dict
    composite: float


@dataclass
class HarnessReport:
    run_name: str
    cases: int
    avg_recall_at_5: float
    avg_mrr: float
    avg_faithfulness: float
    avg_relevance: float
    avg_composite: float
    per_case: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "cases": self.cases,
            "avg_recall@5": self.avg_recall_at_5,
            "avg_mrr": self.avg_mrr,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_relevance": self.avg_relevance,
            "avg_composite": self.avg_composite,
        }


class EvalHarness:
    """
    Runs a dataset of (question, relevant_chunk_ids, optional_reference) through
    the full retrieval → generation → judge pipeline and aggregates results.

    Designed to slot into CI: `pytest tests/integration/test_pipeline.py`
    will call harness.run() and assert score thresholds.
    """

    def __init__(self, pipeline, judge: LLMJudge | None = None) -> None:
        """
        pipeline: an object with .retrieve(question) -> [{chunk_id, content}]
                  and .answer(question, chunks) -> str
        """
        self.pipeline = pipeline
        self.judge = judge or LLMJudge()

    def load_dataset(self, path: str | Path) -> List[EvalCase]:
        data = json.loads(Path(path).read_text())
        return [
            EvalCase(
                question=item["question"],
                relevant_chunk_ids=item["relevant_chunk_ids"],
                reference_answer=item.get("reference_answer", ""),
            )
            for item in data
        ]

    def run(self, cases: List[EvalCase], run_name: str = "default") -> HarnessReport:
        results: List[EvalResult] = []

        for case in cases:
            chunks = self.pipeline.retrieve(case.question)
            retrieved_ids = [c["chunk_id"] for c in chunks]
            answer = self.pipeline.answer(case.question, chunks)
            context = "\n\n".join(c["content"] for c in chunks)

            ret_metrics = retrieval_metrics(
                retrieved_ids, set(case.relevant_chunk_ids)
            )
            gen_metrics = self.judge.evaluate(case.question, answer, context)

            results.append(
                EvalResult(
                    question=case.question,
                    retrieval=ret_metrics,
                    generation=gen_metrics,
                    composite=gen_metrics["composite"],
                )
            )

        def avg(vals):
            return round(statistics.mean(vals), 4) if vals else 0.0

        return HarnessReport(
            run_name=run_name,
            cases=len(results),
            avg_recall_at_5=avg([r.retrieval.get("recall@5", 0.0) for r in results]),
            avg_mrr=avg([r.retrieval.get("mrr", 0.0) for r in results]),
            avg_faithfulness=avg([r.generation["faithfulness"]["score"] for r in results]),
            avg_relevance=avg([r.generation["relevance"]["score"] for r in results]),
            avg_composite=avg([r.composite for r in results]),
            per_case=results,
        )
