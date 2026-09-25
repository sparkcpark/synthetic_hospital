"""Session lifecycle endpoints for agent evaluation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.auth.oauth2 import CurrentUser, get_current_user
from epic_sim.app.models.base import get_db
from epic_sim.app.models.benchmark import BenchmarkGroundTruth
from epic_sim.app.schemas.epic import SessionCreateRequest, SessionResponse
from epic_sim.app.services import session_service
from epic_sim.app.services.session_service import get_redis

router = APIRouter()


def _to_response(s: session_service.SessionData) -> SessionResponse:
    return SessionResponse(
        session_id=s.session_id,
        patient_id=s.patient_id,
        role=s.role,
        department=s.department,
        api_call_count=s.api_call_count,
        max_api_calls=s.max_api_calls,
        status=s.status,
        created_at=s.created_at,
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    redis=Depends(get_redis),
    db: AsyncSession = Depends(get_db),
):
    """Create an evaluation session."""
    if redis is None:
        raise HTTPException(503, "Session management requires Redis")

    # Resolve patient_id from gt_id if needed
    patient_id = body.patient_id
    if body.gt_id and not patient_id:
        result = await db.execute(
            select(BenchmarkGroundTruth.patient_id).where(
                BenchmarkGroundTruth.gt_id == body.gt_id
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise HTTPException(404, f"Ground truth {body.gt_id} not found")
        patient_id = row

    if patient_id is None:
        raise HTTPException(400, "Either patient_id or gt_id must be provided")

    session = await session_service.create_session(
        redis=redis,
        patient_id=patient_id,
        role=body.role,
        department=body.department,
        user_id=str(user.user_id),
        gt_id=body.gt_id,
        max_api_calls=body.max_api_calls,
    )
    return _to_response(session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Get session state."""
    if redis is None:
        raise HTTPException(503, "Session management requires Redis")
    session = await session_service.get_session(redis, session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired")
    return _to_response(session)


@router.delete("/{session_id}", response_model=SessionResponse)
async def end_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """End a session and return final state."""
    if redis is None:
        raise HTTPException(503, "Session management requires Redis")
    session = await session_service.finalize_session(redis, session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired")
    return _to_response(session)
