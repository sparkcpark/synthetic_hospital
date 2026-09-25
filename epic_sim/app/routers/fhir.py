"""FHIR R4 endpoints — read and search for all clinical resources.

Implements Epic-style FHIR R4 patterns:
- Bearer token auth on all endpoints
- Role-based section filtering (partial observability)
- Pagination via Bundle.link
- _count, _sort, date range search
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from epic_sim.app.auth.oauth2 import CurrentUser, get_current_user
from epic_sim.app.auth.rbac import (
    ROLE_SECTION_ACCESS,
    can_access_resource,
    get_allowed_sections,
)
from epic_sim.app.fhir.capability import CAPABILITY_STATEMENT
from epic_sim.app.fhir.resources.allergy_intolerance import AllergyIntolerance
from epic_sim.app.fhir.resources.condition import Condition
from epic_sim.app.fhir.resources.diagnostic_report import DiagnosticReport
from epic_sim.app.fhir.resources.document_reference import DocumentReference
from epic_sim.app.fhir.resources.encounter import Encounter
from epic_sim.app.fhir.resources.medication_request import MedicationRequest
from epic_sim.app.fhir.resources.observation import Observation
from epic_sim.app.fhir.resources.patient import Patient
from epic_sim.app.fhir.resources.service_request import ServiceRequest
from epic_sim.app.fhir.types import Bundle, BundleEntry, BundleEntrySearch, BundleLink
from epic_sim.app.models.base import get_db
from epic_sim.app.models.benchmark import ImagingOrder
from epic_sim.app.models.longitudinal import (
    EncounterEhrSection,
    LongitudinalEncounter,
    LongitudinalPatient,
)
from epic_sim.app.models.ontology import ClinicalFinding, Diagnosis
from epic_sim.app.models.relationships import QuestionDiagnosis, QuestionFinding

router = APIRouter()


def _require_scope(user: CurrentUser, resource_type: str) -> None:
    """Raise 403 if the user lacks the required scope for a resource type."""
    if not can_access_resource(user.role, resource_type, user.scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient scope for {resource_type}",
        )


def _make_bundle(
    entries: list[dict],
    total: int,
    base_url: str = "",
    offset: int = 0,
    count: int = 10,
) -> dict:
    """Build a FHIR Bundle with pagination links."""
    links = [BundleLink(relation="self", url=base_url)]
    if offset + count < total:
        links.append(BundleLink(relation="next", url=f"{base_url}&_offset={offset + count}"))
    if offset > 0:
        prev_offset = max(0, offset - count)
        links.append(BundleLink(relation="previous", url=f"{base_url}&_offset={prev_offset}"))

    bundle = Bundle(
        type="searchset",
        total=total,
        link=links,
        entry=[
            BundleEntry(
                resource=e,
                search=BundleEntrySearch(mode="match"),
            )
            for e in entries
        ],
    )
    return bundle.model_dump(by_alias=True, exclude_none=True)


# ── Metadata ─────────────────────────────────────────────────

@router.get("/metadata")
async def capability_statement():
    """Return the FHIR CapabilityStatement (no auth required)."""
    return CAPABILITY_STATEMENT


# ── Patient ──────────────────────────────────────────────────

@router.get("/Patient/{patient_id}")
async def read_patient(
    patient_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Patient")
    result = await db.execute(
        select(LongitudinalPatient).where(LongitudinalPatient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return Patient.from_db(patient).model_dump(by_alias=True, exclude_none=True)


@router.get("/Patient")
async def search_patient(
    name: str | None = None,
    gender: str | None = None,
    _id: str | None = Query(None, alias="_id"),
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Patient")
    q = select(LongitudinalPatient)

    if _id:
        q = q.where(LongitudinalPatient.patient_id == int(_id))
    if gender:
        sex_map = {"male": "M", "female": "F"}
        q = q.where(LongitudinalPatient.sex == sex_map.get(gender, gender))
    if name:
        # Search in profile JSONB for name field
        q = q.where(
            func.cast(LongitudinalPatient.profile["name"], type_=func.text()).ilike(f"%{name}%")
        )

    # Count
    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    # Paginate
    q = q.order_by(LongitudinalPatient.patient_id).offset(_offset).limit(_count)
    result = await db.execute(q)
    patients = result.scalars().all()

    entries = [Patient.from_db(p).model_dump(by_alias=True, exclude_none=True) for p in patients]
    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── Encounter ────────────────────────────────────────────────

@router.get("/Encounter/{encounter_id}")
async def read_encounter(
    encounter_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Encounter")
    result = await db.execute(
        select(LongitudinalEncounter).where(LongitudinalEncounter.encounter_id == encounter_id)
    )
    enc = result.scalar_one_or_none()
    if enc is None:
        raise HTTPException(status_code=404, detail="Encounter not found")
    return Encounter.from_db(enc).model_dump(by_alias=True, exclude_none=True)


@router.get("/Encounter")
async def search_encounter(
    patient: int | None = None,
    date: str | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    _sort: str | None = Query(None, alias="_sort"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Encounter")
    q = select(LongitudinalEncounter)

    if patient:
        q = q.where(LongitudinalEncounter.patient_id == patient)
    if date:
        # Support ge/le prefixes: date=ge2020-01-01
        if date.startswith("ge"):
            q = q.where(LongitudinalEncounter.encounter_date >= date[2:])
        elif date.startswith("le"):
            q = q.where(LongitudinalEncounter.encounter_date <= date[2:])
        else:
            q = q.where(LongitudinalEncounter.encounter_date == date)

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    # Sort
    if _sort == "-date":
        q = q.order_by(LongitudinalEncounter.encounter_date.desc())
    elif _sort == "date":
        q = q.order_by(LongitudinalEncounter.encounter_date.asc())
    else:
        q = q.order_by(LongitudinalEncounter.encounter_id)

    q = q.offset(_offset).limit(_count)
    result = await db.execute(q)
    encounters = result.scalars().all()

    entries = [Encounter.from_db(e).model_dump(by_alias=True, exclude_none=True) for e in encounters]
    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── Condition ────────────────────────────────────────────────

@router.get("/Condition/{condition_id}")
async def read_condition(
    condition_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Condition")
    result = await db.execute(
        select(Diagnosis).where(Diagnosis.diagnosis_id == condition_id)
    )
    dx = result.scalar_one_or_none()
    if dx is None:
        raise HTTPException(status_code=404, detail="Condition not found")
    return Condition.from_db(dx).model_dump(by_alias=True, exclude_none=True)


@router.get("/Condition")
async def search_condition(
    patient: int | None = None,
    category: str | None = None,
    code: str | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search conditions for a patient.

    Conditions are derived from question_diagnoses for encounters belonging to the patient.
    """
    _require_scope(user, "Condition")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    # Get encounters for this patient
    enc_q = select(LongitudinalEncounter.source_question_ids).where(
        LongitudinalEncounter.patient_id == patient
    )
    enc_result = await db.execute(enc_q)
    rows = enc_result.scalars().all()

    # Collect all question IDs (stored as JSON arrays like "[6235]" or "[6235, 1280]")
    question_ids = set()
    for sq_ids in rows:
        if sq_ids:
            try:
                parsed = json.loads(sq_ids)
                if isinstance(parsed, list):
                    question_ids.update(int(x) for x in parsed)
                else:
                    question_ids.add(int(parsed))
            except (json.JSONDecodeError, ValueError):
                # Fallback: try comma-separated
                for qid_str in sq_ids.split(","):
                    qid_str = qid_str.strip().strip("[]")
                    if qid_str.isdigit():
                        question_ids.add(int(qid_str))

    if not question_ids:
        return _make_bundle([], 0, offset=_offset, count=_count)

    # Get diagnoses via question_diagnoses (correct role only for problem list)
    q = (
        select(Diagnosis, QuestionDiagnosis.role)
        .join(QuestionDiagnosis, Diagnosis.diagnosis_id == QuestionDiagnosis.diagnosis_id)
        .where(QuestionDiagnosis.question_id.in_(question_ids))
    )

    # Category filter
    cat_code = category or "encounter-diagnosis"
    if cat_code == "problem-list-item":
        q = q.where(QuestionDiagnosis.role == "correct")
    # encounter-diagnosis = all roles

    # Code filter (ICD-10 or SNOMED)
    if code:
        # Support token search: system|code
        if "|" in code:
            system, code_val = code.split("|", 1)
            if "icd" in system.lower():
                q = q.where(Diagnosis.icd10_code == code_val)
            elif "snomed" in system.lower():
                q = q.where(Diagnosis.snomed_id == code_val)
        else:
            q = q.where((Diagnosis.icd10_code == code) | (Diagnosis.snomed_id == code))

    # Count distinct diagnoses
    count_q = (
        select(func.count(func.distinct(Diagnosis.diagnosis_id)))
        .select_from(Diagnosis)
        .join(QuestionDiagnosis, Diagnosis.diagnosis_id == QuestionDiagnosis.diagnosis_id)
        .where(QuestionDiagnosis.question_id.in_(question_ids))
    )
    if cat_code == "problem-list-item":
        count_q = count_q.where(QuestionDiagnosis.role == "correct")
    if code:
        if "|" in code:
            system_str, code_val = code.split("|", 1)
            if "icd" in system_str.lower():
                count_q = count_q.where(Diagnosis.icd10_code == code_val)
            elif "snomed" in system_str.lower():
                count_q = count_q.where(Diagnosis.snomed_id == code_val)
        else:
            count_q = count_q.where((Diagnosis.icd10_code == code) | (Diagnosis.snomed_id == code))
    total = (await db.execute(count_q)).scalar() or 0

    # Deduplicate by diagnosis_id, paginate
    q = q.distinct(Diagnosis.diagnosis_id).offset(_offset).limit(_count)
    result = await db.execute(q)

    entries = []
    for dx, role in result.all():
        cond = Condition.from_db(dx, patient_id=patient, category_code=cat_code)
        entries.append(cond.model_dump(by_alias=True, exclude_none=True))

    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── Observation ──────────────────────────────────────────────

@router.get("/Observation/{observation_id}")
async def read_observation(
    observation_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Observation")
    result = await db.execute(
        select(ClinicalFinding).where(ClinicalFinding.finding_id == observation_id)
    )
    cf = result.scalar_one_or_none()
    if cf is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return Observation.from_db(cf).model_dump(by_alias=True, exclude_none=True)


@router.get("/Observation")
async def search_observation(
    patient: int | None = None,
    category: str | None = None,
    code: str | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "Observation")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    # Get question IDs for this patient
    enc_q = select(LongitudinalEncounter.source_question_ids).where(
        LongitudinalEncounter.patient_id == patient
    )
    enc_result = await db.execute(enc_q)

    # Collect all question IDs (stored as JSON arrays like "[6235]" or "[6235, 1280]")
    question_ids = set()
    for sq_ids in enc_result.scalars().all():
        if sq_ids:
            try:
                parsed = json.loads(sq_ids)
                if isinstance(parsed, list):
                    question_ids.update(int(x) for x in parsed)
                else:
                    question_ids.add(int(parsed))
            except (json.JSONDecodeError, ValueError):
                # Fallback: try comma-separated
                for qid_str in sq_ids.split(","):
                    qid_str = qid_str.strip().strip("[]")
                    if qid_str.isdigit():
                        question_ids.add(int(qid_str))

    if not question_ids:
        return _make_bundle([], 0, offset=_offset, count=_count)

    q = (
        select(ClinicalFinding, QuestionFinding)
        .join(QuestionFinding, ClinicalFinding.finding_id == QuestionFinding.finding_id)
        .where(QuestionFinding.question_id.in_(question_ids))
    )

    # Category filter maps to finding_type
    if category:
        cat_to_types = {
            "laboratory": ["lab_value"],
            "vital-signs": ["vital_sign"],
            "imaging": ["imaging_finding"],
            "exam": ["symptom", "sign"],
            "social-history": ["history_item", "demographic"],
        }
        types = cat_to_types.get(category, [])
        if types:
            q = q.where(ClinicalFinding.finding_type.in_(types))

    # Count distinct findings
    count_q = (
        select(func.count(func.distinct(ClinicalFinding.finding_id)))
        .select_from(ClinicalFinding)
        .join(QuestionFinding, ClinicalFinding.finding_id == QuestionFinding.finding_id)
        .where(QuestionFinding.question_id.in_(question_ids))
    )
    if category:
        cat_to_types_count = {
            "laboratory": ["lab_value"],
            "vital-signs": ["vital_sign"],
            "imaging": ["imaging_finding"],
            "exam": ["symptom", "sign"],
            "social-history": ["history_item", "demographic"],
        }
        types_count = cat_to_types_count.get(category, [])
        if types_count:
            count_q = count_q.where(ClinicalFinding.finding_type.in_(types_count))
    total = (await db.execute(count_q)).scalar() or 0

    q = q.distinct(ClinicalFinding.finding_id).offset(_offset).limit(_count)
    result = await db.execute(q)

    entries = []
    for cf, qf in result.all():
        obs = Observation.from_db(cf, qf=qf, patient_id=patient)
        entries.append(obs.model_dump(by_alias=True, exclude_none=True))

    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── DiagnosticReport ─────────────────────────────────────────

@router.get("/DiagnosticReport")
async def search_diagnostic_report(
    patient: int | None = None,
    category: str | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "DiagnosticReport")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    # Filter sections by allowed section types AND role-based access
    allowed = get_allowed_sections(user.role)
    report_types = {"imaging", "pathology", "labs"}
    if allowed is not None:
        report_types = report_types & allowed

    if not report_types:
        return _make_bundle([], 0, offset=_offset, count=_count)

    # Category filter
    if category:
        cat_map = {"RAD": "imaging", "SP": "pathology", "LAB": "labs"}
        requested_type = cat_map.get(category, category)
        report_types = report_types & {requested_type}

    q = (
        select(EncounterEhrSection, LongitudinalEncounter.encounter_date)
        .join(LongitudinalEncounter, EncounterEhrSection.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
        .where(EncounterEhrSection.section_type.in_(report_types))
    )

    count_subq = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_subq)).scalar() or 0

    q = q.order_by(EncounterEhrSection.id).offset(_offset).limit(_count)
    result = await db.execute(q)

    entries = []
    for section, enc_date in result.all():
        report = DiagnosticReport.from_ehr_section(
            section, patient_id=patient, encounter_date=enc_date, section_type=section.section_type
        )
        entries.append(report.model_dump(by_alias=True, exclude_none=True))

    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── ServiceRequest ───────────────────────────────────────────

@router.get("/ServiceRequest/{request_id}")
async def read_service_request(
    request_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "ServiceRequest")
    result = await db.execute(
        select(ImagingOrder).where(ImagingOrder.order_id == request_id)
    )
    order = result.scalar_one_or_none()
    if order is None:
        raise HTTPException(status_code=404, detail="ServiceRequest not found")

    # Need patient_id from the encounter
    enc = await db.execute(
        select(LongitudinalEncounter.patient_id).where(
            LongitudinalEncounter.encounter_id == order.encounter_id
        )
    )
    patient_id = enc.scalar_one()
    return ServiceRequest.from_db(order, patient_id=patient_id).model_dump(by_alias=True, exclude_none=True)


@router.get("/ServiceRequest")
async def search_service_request(
    patient: int | None = None,
    status_param: str | None = Query(None, alias="status"),
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "ServiceRequest")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    q = (
        select(ImagingOrder)
        .join(LongitudinalEncounter, ImagingOrder.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
    )

    count_subq = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_subq)).scalar() or 0

    q = q.order_by(ImagingOrder.order_id).offset(_offset).limit(_count)
    result = await db.execute(q)
    orders = result.scalars().all()

    entries = [
        ServiceRequest.from_db(o, patient_id=patient).model_dump(by_alias=True, exclude_none=True)
        for o in orders
    ]
    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── DocumentReference ────────────────────────────────────────

@router.get("/DocumentReference")
async def search_document_reference(
    patient: int | None = None,
    type_param: str | None = Query(None, alias="type"),
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search DocumentReferences (EHR sections) for a patient.

    Applies role-based section filtering (partial observability).
    """
    _require_scope(user, "DocumentReference")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    # Role-based section filtering
    allowed = get_allowed_sections(user.role)

    q = (
        select(EncounterEhrSection, LongitudinalEncounter.encounter_date)
        .join(LongitudinalEncounter, EncounterEhrSection.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
    )

    if allowed is not None:
        q = q.where(EncounterEhrSection.section_type.in_(allowed))

    if type_param:
        q = q.where(EncounterEhrSection.section_type == type_param)

    count_subq = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_subq)).scalar() or 0

    q = q.order_by(EncounterEhrSection.encounter_id, EncounterEhrSection.section_order)
    q = q.offset(_offset).limit(_count)
    result = await db.execute(q)

    entries = []
    for section, enc_date in result.all():
        doc = DocumentReference.from_ehr_section(section, patient_id=patient, encounter_date=enc_date)
        entries.append(doc.model_dump(by_alias=True, exclude_none=True))

    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── MedicationRequest ────────────────────────────────────────

@router.get("/MedicationRequest")
async def search_medication_request(
    patient: int | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "MedicationRequest")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    q = (
        select(EncounterEhrSection, LongitudinalEncounter.encounter_date)
        .join(LongitudinalEncounter, EncounterEhrSection.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
        .where(EncounterEhrSection.section_type == "medications")
    )

    count_subq = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_subq)).scalar() or 0

    q = q.order_by(EncounterEhrSection.encounter_id).offset(_offset).limit(_count)
    result = await db.execute(q)

    entries = []
    for section, enc_date in result.all():
        med = MedicationRequest.from_ehr_section(section, patient_id=patient, encounter_date=enc_date)
        entries.append(med.model_dump(by_alias=True, exclude_none=True))

    return _make_bundle(entries, total, offset=_offset, count=_count)


# ── AllergyIntolerance ───────────────────────────────────────

@router.get("/AllergyIntolerance")
async def search_allergy_intolerance(
    patient: int | None = None,
    _count: int = Query(10, le=100, alias="_count"),
    _offset: int = Query(0, alias="_offset"),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_scope(user, "AllergyIntolerance")

    if patient is None:
        raise HTTPException(status_code=400, detail="'patient' parameter is required")

    q = (
        select(EncounterEhrSection, LongitudinalEncounter.encounter_date)
        .join(LongitudinalEncounter, EncounterEhrSection.encounter_id == LongitudinalEncounter.encounter_id)
        .where(LongitudinalEncounter.patient_id == patient)
        .where(EncounterEhrSection.section_type == "allergies")
    )

    count_subq = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_subq)).scalar() or 0

    q = q.order_by(EncounterEhrSection.encounter_id).offset(_offset).limit(_count)
    result = await db.execute(q)

    entries = []
    for section, enc_date in result.all():
        allergy = AllergyIntolerance.from_ehr_section(section, patient_id=patient, encounter_date=enc_date)
        entries.append(allergy.model_dump(by_alias=True, exclude_none=True))

    return _make_bundle(entries, total, offset=_offset, count=_count)
