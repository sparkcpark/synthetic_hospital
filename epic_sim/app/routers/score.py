"""Scoring (reward) endpoint.

    GET  /score/tasks                      tasks and the metric each reward is taken from
    GET  /score/instances?task=&split=     benchmark instances without their labels
    POST /score                            score one submission -> reward in [0, 1]
    POST /score/batch                      score up to 256 submissions in one call

Authentication is a shared secret, not a user JWT: the trainer holds
EPIC_SIM_SCORER_TOKEN and sends it as `X-Scorer-Token`; the policy under
training authenticates to the EHR tools with an ordinary user token and cannot
reach its own reward or the labels. Setting EPIC_SIM_SCORER_TOKEN to an empty
string disables the endpoint.

Scoring is synchronous CPU work over a psycopg connection, so the handlers are
plain `def` functions and FastAPI runs them in its threadpool.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import contextmanager

import psycopg
from fastapi import APIRouter, Header, HTTPException, Query

from epic_sim.app.config import settings
from epic_sim.app.schemas.score import (
    InstanceSummary,
    ScoreBatchItem,
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoreRequest,
    ScoreResponse,
    TaskInfo,
)
from eval.config import EVAL_TASKS
from eval.score_one import (
    PRIMARY_METRIC,
    InstanceNotFound,
    TaskMismatch,
    list_instances,
    score_submission,
)

log = logging.getLogger(__name__)
router = APIRouter()

_TASK_NOTES = {
    "patient_diagnosis": "severity-weighted F1 under the chart-neutral rule",
    "context_summarization": "must-include finding recall; specialty variant: conditioned_f1 "
                             "(involved) or abstention_accuracy (absent)",
    "evidence_retrieval": "precision at 5 over chart sections; ndcg_10 also returned",
    "imaging_indication": "ontology-grounded concept F1 of the inferred clinical question "
                          "(clinical_question_f1, token-level, also returned)",
}


def _require_token(x_scorer_token: str | None) -> None:
    if not settings.scorer_token:
        raise HTTPException(503, "scoring endpoint disabled (EPIC_SIM_SCORER_TOKEN is empty)")
    if not x_scorer_token or not secrets.compare_digest(x_scorer_token, settings.scorer_token):
        raise HTTPException(401, "missing or invalid X-Scorer-Token")


@contextmanager
def _pg():
    dsn = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")
    conn = psycopg.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


@router.get("/tasks", response_model=list[TaskInfo])
def scoring_tasks(x_scorer_token: str | None = Header(default=None)):
    _require_token(x_scorer_token)
    return [TaskInfo(task=t, reward_metric=PRIMARY_METRIC[t], note=_TASK_NOTES.get(t)) for t in EVAL_TASKS]


@router.get("/instances", response_model=list[InstanceSummary])
def scoring_instances(
    task: str | None = Query(default=None),
    split: str | None = Query(default=None, pattern="^(public|heldout|train)$"),
    variant: str | None = Query(default=None, description="context_summarization only"),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    x_scorer_token: str | None = Header(default=None),
):
    _require_token(x_scorer_token)
    if task is not None and task not in EVAL_TASKS:
        raise HTTPException(422, f"unknown task '{task}'; one of {EVAL_TASKS}")
    with _pg() as conn:
        return list_instances(conn, task=task, split=split, variant=variant, limit=limit, offset=offset)


def _score_or_raise(conn, req: ScoreRequest) -> ScoreResponse:
    try:
        result = score_submission(conn, req.gt_id, req.prediction, task=req.task,
                                  reward_metric=req.reward_metric)
    except InstanceNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except TaskMismatch as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return ScoreResponse(**result)


@router.post("", response_model=ScoreResponse)
def score(req: ScoreRequest, x_scorer_token: str | None = Header(default=None)):
    """Score one submission. Malformed submissions score 0 rather than erroring."""
    _require_token(x_scorer_token)
    with _pg() as conn:
        return _score_or_raise(conn, req)


@router.post("/batch", response_model=ScoreBatchResponse)
def score_batch(req: ScoreBatchRequest, x_scorer_token: str | None = Header(default=None)):
    """Score several submissions; per-item failures are reported, not raised."""
    _require_token(x_scorer_token)
    items: list[ScoreBatchItem] = []
    with _pg() as conn:
        for item in req.items:
            try:
                items.append(ScoreBatchItem(gt_id=item.gt_id, ok=True, result=_score_or_raise(conn, item)))
            except HTTPException as exc:
                items.append(ScoreBatchItem(gt_id=item.gt_id, ok=False, error=str(exc.detail)))
    return ScoreBatchResponse(items=items)
