"""Pydantic models for the scoring (reward) endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    gt_id: int = Field(..., description="Benchmark instance id (benchmark_ground_truth.gt_id)")
    prediction: dict[str, Any] = Field(
        default_factory=dict,
        description="Submission in the task's output schema (same JSON the eval harness stores)",
    )
    task: str | None = Field(
        default=None,
        description="Optional; if given, must match the instance's task (guards against mixed-up ids)",
    )
    reward_metric: str | None = Field(
        default=None,
        description="Optional; take the reward from this metric instead of the task's primary metric",
    )


class ScoreResponse(BaseModel):
    gt_id: int
    task: str
    split: str
    patient_id: int | None = None
    encounter_id: int | None = None
    variant: str | None = None
    reward: float = Field(..., ge=0.0, le=1.0)
    reward_metric: str
    metrics: dict[str, Any]


class ScoreBatchRequest(BaseModel):
    items: list[ScoreRequest] = Field(..., max_length=256)


class ScoreBatchItem(BaseModel):
    gt_id: int
    ok: bool
    result: ScoreResponse | None = None
    error: str | None = None


class ScoreBatchResponse(BaseModel):
    items: list[ScoreBatchItem]


class InstanceSummary(BaseModel):
    gt_id: int
    task: str
    granularity: str
    split: str
    patient_id: int | None = None
    encounter_id: int | None = None
    difficulty: str | None = None
    variant: str | None = None
    clinical_question: str | None = None
    specialty: str | None = None


class TaskInfo(BaseModel):
    task: str
    reward_metric: str
    note: str | None = None
