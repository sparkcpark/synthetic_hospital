"""Pydantic models for the reset-and-step environment endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    gt_id: int | None = Field(default=None, description="Play this instance; omit to sample one")
    task: str | None = Field(default=None, description="Required when sampling; checked when gt_id is given")
    split: str | None = Field(default=None, pattern="^(public|heldout|train)$",
                              description="Sampling pool (default train)")
    seed: int | None = Field(default=None, description="Seed for sampling")
    budget: int | None = Field(default=None, ge=1, le=500, description="Action budget (default 40)")


class ResetResponse(BaseModel):
    episode_id: str
    agent_token: str = Field(..., description="Per-episode token for step/state without the scorer token; "
                                              "callers using it never see the reward")
    gt_id: int
    task: str
    split: str
    variant: str | None = None
    patient_id: int | None = None
    encounter_id: int | None = None
    budget: int
    remaining: int
    submit_tool: str
    instructions: str = Field(..., description="System prompt used by the paper's agents")
    intro: str = Field(..., description="First user message: the patient assignment")
    task_inputs: dict[str, Any]
    tools: list[dict[str, Any]] = Field(..., description="OpenAI-style function schemas incl. the submit tool")


class StepRequest(BaseModel):
    episode_id: str
    name: str = Field(..., description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    episode_id: str
    step: int
    tool_name: str
    observation: Any
    observation_text: str
    remaining: int
    reward: float | None = Field(..., description="None for agent-token callers at submission")
    done: bool
    info: dict[str, Any]


class EpisodeRef(BaseModel):
    episode_id: str


class OracleResponse(BaseModel):
    gt_id: int
    task: str
    submit_tool: str
    arguments: dict[str, Any]


class EpisodeState(BaseModel):
    episode_id: str
    gt_id: int
    task: str
    split: str
    patient_id: int | None = None
    encounter_id: int | None = None
    budget: int
    steps: int
    done: bool
    submitted: bool
    reward: float | None = None
    trace: list[dict[str, Any]]
    created_at: str
