"""Reset-and-step environment endpoints.

    POST /env/reset            start an episode (given gt_id, or sampled from task/split)   [scorer token]
    GET  /env/current          the episode this container was started for (autostart)      [no auth]
    POST /env/step             execute one tool call; the task's submit tool ends the episode
    GET  /env/state/{id}       budget, trace, outcome
    POST /env/close            abandon an episode (reward 0 if it was not submitted)        [scorer token]
    GET  /env/oracle/{id}      a label-derived submission for the episode's instance        [scorer token]

Two credentials:
- the scorer token (X-Scorer-Token, EPIC_SIM_SCORER_TOKEN) is held by the training
  harness or the verifier; it can do everything and sees rewards and metrics;
- the per-episode agent token (X-Episode-Token, returned by reset and by
  /env/current) lets the policy itself call step and state for that one episode,
  but the reward is withheld from it: submission only reports that the episode
  is over. This is what a Harbor task's agent container uses.

Episodes live in Redis, which the compose stack provides. Python client:
eval.env_client.SyntheticHospitalEnv.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from epic_sim.app.config import settings
from epic_sim.app.models.base import get_db
from epic_sim.app.routers.agent import TOOL_DEFINITIONS, _dispatch_tool
from epic_sim.app.schemas.env import (
    EpisodeRef,
    EpisodeState,
    OracleResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from epic_sim.app.services import env_service
from epic_sim.app.services.env_service import EpisodeError, submit_tool_for
from epic_sim.app.services.session_service import get_redis
from eval.config import EVAL_TASKS

router = APIRouter()


def _scorer_ok(x_scorer_token: str | None) -> bool:
    return bool(settings.scorer_token and x_scorer_token
                and secrets.compare_digest(x_scorer_token, settings.scorer_token))


def _require_scorer(x_scorer_token: str | None) -> None:
    if not settings.scorer_token:
        raise HTTPException(503, "environment disabled (EPIC_SIM_SCORER_TOKEN is empty)")
    if not _scorer_ok(x_scorer_token):
        raise HTTPException(401, "missing or invalid X-Scorer-Token")


def _require_redis(redis) -> None:
    if redis is None:
        raise HTTPException(503, "environment requires Redis for episode state")


async def _authorize(redis, episode_id: str, x_scorer_token: str | None, x_episode_token: str | None) -> bool:
    """Return True for a privileged (scorer) caller, False for a valid agent-token caller."""
    if _scorer_ok(x_scorer_token):
        return True
    if x_episode_token:
        try:
            ep = await env_service.load_episode(redis, episode_id)
        except EpisodeError as exc:
            raise HTTPException(exc.status, exc.detail) from exc
        if secrets.compare_digest(x_episode_token, ep.get("agent_token", "")):
            return False
    raise HTTPException(401, "missing or invalid X-Scorer-Token / X-Episode-Token")


def _raise(exc: EpisodeError):
    raise HTTPException(exc.status, exc.detail)


@router.post("/reset", response_model=ResetResponse)
async def env_reset(req: ResetRequest, redis=Depends(get_redis),
                    x_scorer_token: str | None = Header(default=None)):
    _require_scorer(x_scorer_token)
    _require_redis(redis)
    if req.gt_id is None and not req.task:
        raise HTTPException(422, "give gt_id, or task (with optional split/seed) to sample an instance")
    if req.task and req.task not in EVAL_TASKS:
        raise HTTPException(422, f"unknown task '{req.task}'; one of {EVAL_TASKS}")
    try:
        return await env_service.reset(
            redis, [t.model_dump() for t in TOOL_DEFINITIONS], gt_id=req.gt_id, task=req.task,
            split=req.split, seed=req.seed, budget=req.budget,
        )
    except EpisodeError as exc:
        _raise(exc)


@router.get("/current", response_model=ResetResponse)
async def env_current(redis=Depends(get_redis)):
    """The autostart episode's brief, including its agent token (no authentication:
    a container started for one episode is that episode's agent)."""
    _require_redis(redis)
    try:
        return await env_service.current_episode(redis)
    except EpisodeError as exc:
        _raise(exc)


@router.post("/step", response_model=StepResponse)
async def env_step(req: StepRequest, db: AsyncSession = Depends(get_db), redis=Depends(get_redis),
                   x_scorer_token: str | None = Header(default=None),
                   x_episode_token: str | None = Header(default=None)):
    _require_redis(redis)
    privileged = await _authorize(redis, req.episode_id, x_scorer_token, x_episode_token)
    try:
        return await env_service.step(redis, db, _dispatch_tool, req.episode_id, req.name, req.arguments,
                                      privileged=privileged)
    except EpisodeError as exc:
        _raise(exc)


@router.get("/state/{episode_id}", response_model=EpisodeState)
async def env_state(episode_id: str, redis=Depends(get_redis),
                    x_scorer_token: str | None = Header(default=None),
                    x_episode_token: str | None = Header(default=None)):
    _require_redis(redis)
    privileged = await _authorize(redis, episode_id, x_scorer_token, x_episode_token)
    try:
        return env_service.state_view(await env_service.load_episode(redis, episode_id), privileged=privileged)
    except EpisodeError as exc:
        _raise(exc)


@router.post("/close", response_model=EpisodeState)
async def env_close(req: EpisodeRef, redis=Depends(get_redis),
                    x_scorer_token: str | None = Header(default=None)):
    _require_scorer(x_scorer_token)
    _require_redis(redis)
    try:
        return await env_service.close(redis, req.episode_id)
    except EpisodeError as exc:
        _raise(exc)


@router.get("/oracle/{episode_id}", response_model=OracleResponse)
async def env_oracle(episode_id: str, redis=Depends(get_redis),
                     x_scorer_token: str | None = Header(default=None)):
    """A submission built from the labels, for Harbor's oracle agent and for smoke tests."""
    _require_scorer(x_scorer_token)
    _require_redis(redis)
    try:
        ep = await env_service.load_episode(redis, episode_id)
        args = await run_in_threadpool(env_service.oracle_submission, ep["gt_id"])
    except EpisodeError as exc:
        _raise(exc)
    return OracleResponse(gt_id=ep["gt_id"], task=ep["task"], submit_tool=submit_tool_for(ep["task"]), arguments=args)
