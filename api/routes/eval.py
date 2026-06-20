"""Evaluation endpoint — runs the full harness and returns metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_pipeline
from api.models import EvalRequest, EvalResponse
from eval.harness import EvalHarness
from eval.judge import LLMJudge

router = APIRouter(prefix="/eval", tags=["eval"])

# Thresholds that must hold for CI to pass
COMPOSITE_THRESHOLD = 0.70
RECALL_THRESHOLD = 0.50


@router.post("", response_model=EvalResponse)
def run_eval(body: EvalRequest, pipeline=Depends(get_pipeline)):
    harness = EvalHarness(pipeline=pipeline, judge=LLMJudge())

    try:
        cases = harness.load_dataset(body.dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {body.dataset_path}")

    report = harness.run(cases, run_name=body.run_name)

    return EvalResponse(
        run_name=report.run_name,
        cases=report.cases,
        avg_recall_at_5=report.avg_recall_at_5,
        avg_mrr=report.avg_mrr,
        avg_faithfulness=report.avg_faithfulness,
        avg_relevance=report.avg_relevance,
        avg_composite=report.avg_composite,
        passed_threshold=(
            report.avg_composite >= COMPOSITE_THRESHOLD
            and report.avg_recall_at_5 >= RECALL_THRESHOLD
        ),
    )
