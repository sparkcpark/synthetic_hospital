"""Simulated Epic clinical workflow endpoints.

Higher-level REST endpoints that compose database queries into
clinical workflows: chart review, results, orders, radiology.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.auth.oauth2 import CurrentUser, get_current_user
from epic_sim.app.auth.scopes import has_scope
from epic_sim.app.models.base import get_db
from epic_sim.app.schemas.epic import (
    AddProblemRequest,
    ImagingOrderRequest,
    LabOrderRequest,
    PreReadSubmission,
)
from epic_sim.app.services import epic_service

router = APIRouter()


def _require_scope(user: CurrentUser, scope: str) -> None:
    if not has_scope(user.scopes, scope):
        raise HTTPException(403, f"Missing scope: {scope}")


# ---------------------------------------------------------------------------
# Chart Review
# ---------------------------------------------------------------------------

@router.get("/chart/{patient_id}/summary")
async def chart_summary(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/Patient.read")
    summary = await epic_service.get_patient_summary(db, patient_id, user.role)
    return summary.model_dump()


@router.get("/chart/{patient_id}/encounters")
async def chart_encounters(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/Encounter.read")
    encounters = await epic_service.get_encounters(db, patient_id, user.role)
    return [e.model_dump() for e in encounters]


@router.get("/chart/{patient_id}/encounters/{encounter_id}")
async def chart_encounter_detail(
    patient_id: int,
    encounter_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/Encounter.read")
    detail = await epic_service.get_encounter_detail(db, encounter_id, user.role)
    if detail is None or detail.patient_id != patient_id:
        raise HTTPException(404, "Encounter not found")
    return detail.model_dump()


@router.get("/chart/{patient_id}/problems")
async def chart_problems(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/Condition.read")
    problems = await epic_service.get_problem_list(db, patient_id)
    return [p.model_dump() for p in problems]


@router.post("/chart/{patient_id}/problems", status_code=201)
async def add_problem(
    patient_id: int,
    body: AddProblemRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/Condition.read")
    problem = await epic_service.add_problem(db, patient_id, body)
    return problem.model_dump()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@router.get("/results/{patient_id}/labs")
async def lab_results(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/Observation.read")
    results = await epic_service.get_lab_results(db, patient_id)
    return [r.model_dump() for r in results]


@router.get("/results/{patient_id}/imaging")
async def imaging_results(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/DiagnosticReport.read")
    results = await epic_service.get_imaging_results(db, patient_id)
    return [r.model_dump() for r in results]


@router.get("/results/{patient_id}/pathology")
async def pathology_results(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/DiagnosticReport.read")
    results = await epic_service.get_pathology_results(db, patient_id)
    return [r.model_dump() for r in results]


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

@router.post("/orders/imaging", status_code=201)
async def order_imaging(
    body: ImagingOrderRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/ServiceRequest.write")
    try:
        order = await epic_service.place_imaging_order(db, body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return order.model_dump()


@router.post("/orders/lab", status_code=201)
async def order_lab(
    body: LabOrderRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/ServiceRequest.write")
    order = await epic_service.place_lab_order(db, body)
    return order.model_dump()


# ---------------------------------------------------------------------------
# Radiology
# ---------------------------------------------------------------------------

@router.get("/radiology/worklist")
async def radiology_worklist(
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/ServiceRequest.read")
    items = await epic_service.get_radiology_worklist(db)
    return [i.model_dump() for i in items]


@router.get("/radiology/pre-read/{order_id}")
async def pre_read_context(
    order_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/DiagnosticReport.read")
    ctx = await epic_service.get_pre_read_context(db, order_id, user.role)
    if ctx is None:
        raise HTTPException(404, "Order not found")
    return ctx.model_dump()


@router.post("/radiology/pre-read/{order_id}", status_code=201)
async def submit_pre_read(
    order_id: int,
    body: PreReadSubmission,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "patient/ServiceRequest.read")
    from epic_sim.app.models.auth import AgentSubmission
    import uuid

    submission = AgentSubmission(
        submission_id=str(uuid.uuid4()),
        session_id="",
        submission_type="pre_read",
        payload={
            "order_id": order_id,
            "summary": body.summary,
            "impression": body.impression,
            "findings": body.findings,
        },
    )
    db.add(submission)
    await db.commit()
    return {"status": "recorded", "order_id": order_id}
