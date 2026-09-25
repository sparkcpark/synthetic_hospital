"""Shared data-access logic for Epic simulation endpoints.

Both the Epic router and Agent router call these functions directly,
avoiding internal HTTP calls and ensuring consistent RBAC.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.auth.rbac import get_allowed_sections
from epic_sim.app.models.auth import AgentSubmission
from epic_sim.app.models.benchmark import ImagingOrder
from epic_sim.app.models.longitudinal import (
    EncounterEhrSection,
    LongitudinalEncounter,
    LongitudinalPatient,
)
from epic_sim.app.models.ontology import Diagnosis
from epic_sim.app.models.relationships import QuestionDiagnosis
from epic_sim.app.schemas.epic import (
    AddProblemRequest,
    EncounterDetail,
    EncounterSummary,
    ImagingOrderRequest,
    ImagingResult,
    LabOrderRequest,
    LabResult,
    OrderResponse,
    PathologyResult,
    PatientSummary,
    PreReadContext,
    ProblemEntry,
    SectionEntry,
    WorklistItem,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_question_ids(source_question_ids_list: list[str | None]) -> set[int]:
    """Parse source_question_ids JSON arrays into a flat set of question IDs.

    Handles both JSON array format ("[6235]", "[6235, 1280]") and
    comma-separated fallback.
    """
    question_ids: set[int] = set()
    for sq_ids in source_question_ids_list:
        if not sq_ids:
            continue
        try:
            parsed = json.loads(sq_ids)
            if isinstance(parsed, list):
                question_ids.update(int(x) for x in parsed)
            else:
                question_ids.add(int(parsed))
        except (json.JSONDecodeError, ValueError):
            for qid_str in sq_ids.split(","):
                qid_str = qid_str.strip().strip("[]")
                if qid_str.isdigit():
                    question_ids.add(int(qid_str))
    return question_ids


def _section_to_entry(s: EncounterEhrSection) -> SectionEntry:
    return SectionEntry(
        section_id=s.id,
        encounter_id=s.encounter_id,
        section_type=s.section_type,
        section_text=s.section_text,
        section_order=s.section_order,
    )


# ---------------------------------------------------------------------------
# Chart Review
# ---------------------------------------------------------------------------

async def get_patient_summary(
    db: AsyncSession, patient_id: int, role: str
) -> PatientSummary:
    """Patient snapshot: demographics, active problems, recent encounters, meds, allergies."""
    # Patient
    patient = await db.get(LongitudinalPatient, patient_id)
    if patient is None:
        return PatientSummary(patient_id=patient_id)

    profile = patient.profile or {}
    name = profile.get("name", f"Patient {patient_id}")

    # Recent encounters (latest 5)
    enc_q = (
        select(LongitudinalEncounter)
        .where(LongitudinalEncounter.patient_id == patient_id)
        .order_by(LongitudinalEncounter.encounter_date.desc())
        .limit(5)
    )
    enc_result = await db.execute(enc_q)
    encounters = enc_result.scalars().all()

    recent = [
        EncounterSummary(
            encounter_id=e.encounter_id,
            date=e.encounter_date,
            type=e.encounter_type.value if e.encounter_type else None,
            chief_complaint=e.chief_complaint,
            department=e.department,
            attending=e.attending_name,
        )
        for e in encounters
    ]

    # Medications and allergies from latest encounter sections
    medications: list[str] = []
    allergies: list[str] = []
    if encounters:
        latest_id = encounters[0].encounter_id
        allowed = get_allowed_sections(role)
        sec_q = (
            select(EncounterEhrSection)
            .where(EncounterEhrSection.encounter_id == latest_id)
            .where(EncounterEhrSection.section_type.in_(["medications", "allergies"]))
        )
        sec_result = await db.execute(sec_q)
        for sec in sec_result.scalars().all():
            if allowed is not None and sec.section_type not in allowed:
                continue
            if sec.section_type == "medications":
                medications = [line.strip() for line in sec.section_text.split("\n") if line.strip()]
            elif sec.section_type == "allergies":
                allergies = [line.strip() for line in sec.section_text.split("\n") if line.strip()]

    # Active problems (correct diagnoses)
    problems = await get_problem_list(db, patient_id)

    return PatientSummary(
        patient_id=patient_id,
        name=name,
        age=patient.age,
        sex=patient.sex,
        race_ethnicity=patient.race_ethnicity,
        insurance=patient.insurance,
        pcp=patient.pcp_name,
        active_problems=problems,
        recent_encounters=recent,
        medications=medications,
        allergies=allergies,
    )


async def get_encounters(
    db: AsyncSession, patient_id: int, role: str
) -> list[EncounterSummary]:
    """All encounters for patient, ordered chronologically."""
    q = (
        select(LongitudinalEncounter)
        .where(LongitudinalEncounter.patient_id == patient_id)
        .order_by(LongitudinalEncounter.encounter_date)
    )
    result = await db.execute(q)
    return [
        EncounterSummary(
            encounter_id=e.encounter_id,
            date=e.encounter_date,
            type=e.encounter_type.value if e.encounter_type else None,
            chief_complaint=e.chief_complaint,
            department=e.department,
            attending=e.attending_name,
        )
        for e in result.scalars().all()
    ]


async def get_encounter_detail(
    db: AsyncSession, encounter_id: int, role: str
) -> EncounterDetail | None:
    """Single encounter with all sections, role-filtered."""
    enc = await db.get(LongitudinalEncounter, encounter_id)
    if enc is None:
        return None

    allowed = get_allowed_sections(role)
    sec_q = (
        select(EncounterEhrSection)
        .where(EncounterEhrSection.encounter_id == encounter_id)
        .order_by(EncounterEhrSection.section_order)
    )
    sec_result = await db.execute(sec_q)
    sections = []
    for s in sec_result.scalars().all():
        if allowed is not None and s.section_type not in allowed:
            continue
        sections.append(_section_to_entry(s))

    return EncounterDetail(
        encounter_id=enc.encounter_id,
        patient_id=enc.patient_id,
        date=enc.encounter_date,
        type=enc.encounter_type.value if enc.encounter_type else None,
        chief_complaint=enc.chief_complaint,
        department=enc.department,
        attending=enc.attending_name,
        sections=sections,
    )


async def get_encounter_section(
    db: AsyncSession, section_id: int, role: str
) -> SectionEntry | None:
    """Single section by ID, with role check."""
    sec = await db.get(EncounterEhrSection, section_id)
    if sec is None:
        return None
    allowed = get_allowed_sections(role)
    if allowed is not None and sec.section_type not in allowed:
        return None
    return _section_to_entry(sec)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

async def get_lab_results(
    db: AsyncSession, patient_id: int
) -> list[LabResult]:
    """Lab sections for patient, chronological."""
    sql = text("""
        SELECT ees.encounter_id, le.encounter_date, ees.section_text
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON le.encounter_id = ees.encounter_id
        WHERE le.patient_id = :pid AND ees.section_type = 'labs'
        ORDER BY le.encounter_date
    """)
    result = await db.execute(sql, {"pid": patient_id})
    return [
        LabResult(encounter_id=r[0], encounter_date=r[1], section_text=r[2])
        for r in result.fetchall()
    ]


async def get_imaging_results(
    db: AsyncSession, patient_id: int
) -> list[ImagingResult]:
    """Imaging sections with order info."""
    sql = text("""
        SELECT ees.encounter_id, le.encounter_date, ees.section_text,
               CASE WHEN io.order_id IS NOT NULL THEN true ELSE false END AS has_order
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON le.encounter_id = ees.encounter_id
        LEFT JOIN imaging_orders io ON io.encounter_id = ees.encounter_id
        WHERE le.patient_id = :pid AND ees.section_type = 'imaging'
        ORDER BY le.encounter_date
    """)
    result = await db.execute(sql, {"pid": patient_id})
    return [
        ImagingResult(
            encounter_id=r[0], encounter_date=r[1],
            section_text=r[2], has_order=r[3],
        )
        for r in result.fetchall()
    ]


async def get_pathology_results(
    db: AsyncSession, patient_id: int
) -> list[PathologyResult]:
    """Pathology sections for patient."""
    sql = text("""
        SELECT ees.encounter_id, le.encounter_date, ees.section_text
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON le.encounter_id = ees.encounter_id
        WHERE le.patient_id = :pid AND ees.section_type = 'pathology'
        ORDER BY le.encounter_date
    """)
    result = await db.execute(sql, {"pid": patient_id})
    return [
        PathologyResult(encounter_id=r[0], encounter_date=r[1], section_text=r[2])
        for r in result.fetchall()
    ]


# ---------------------------------------------------------------------------
# Problem List
# ---------------------------------------------------------------------------

async def get_problem_list(
    db: AsyncSession, patient_id: int
) -> list[ProblemEntry]:
    """Active problem list: correct diagnoses from encounter questions."""
    # Get all question IDs for this patient
    enc_q = select(LongitudinalEncounter.source_question_ids).where(
        LongitudinalEncounter.patient_id == patient_id
    )
    enc_result = await db.execute(enc_q)
    sq_ids_list = enc_result.scalars().all()
    question_ids = _parse_question_ids(sq_ids_list)

    if not question_ids:
        return []

    # Get correct diagnoses (deduplicated)
    q = (
        select(Diagnosis)
        .join(QuestionDiagnosis, Diagnosis.diagnosis_id == QuestionDiagnosis.diagnosis_id)
        .where(QuestionDiagnosis.question_id.in_(question_ids))
        .where(QuestionDiagnosis.role == "correct")
        .distinct(Diagnosis.diagnosis_id)
    )
    result = await db.execute(q)
    seen: set[int] = set()
    problems = []
    for dx in result.scalars().all():
        if dx.diagnosis_id not in seen:
            seen.add(dx.diagnosis_id)
            problems.append(ProblemEntry(
                diagnosis_id=dx.diagnosis_id,
                display_name=dx.display_name or dx.icd10_desc or "Unknown",
                icd10_code=dx.icd10_code,
                snomed_id=dx.snomed_id,
            ))
    return problems


async def add_problem(
    db: AsyncSession, patient_id: int, request: AddProblemRequest, session_id: str = ""
) -> ProblemEntry:
    """Record a diagnosis submission (does NOT modify ground truth tables)."""
    submission = AgentSubmission(
        submission_id=str(uuid.uuid4()),
        session_id=session_id,
        patient_id=patient_id,
        submission_type="diagnosis",
        payload={
            "display_name": request.display_name,
            "icd10_code": request.icd10_code,
        },
    )
    db.add(submission)
    await db.commit()

    return ProblemEntry(
        diagnosis_id=0,  # Synthetic — not a real diagnosis row
        display_name=request.display_name,
        icd10_code=request.icd10_code,
    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

async def place_imaging_order(
    db: AsyncSession, request: ImagingOrderRequest
) -> OrderResponse:
    """Insert an imaging order. Validates encounter→patient relationship."""
    # Verify encounter belongs to patient
    enc = await db.get(LongitudinalEncounter, request.encounter_id)
    if enc is None or enc.patient_id != request.patient_id:
        raise ValueError(f"Encounter {request.encounter_id} does not belong to patient {request.patient_id}")

    sql = text("""
        INSERT INTO imaging_orders (encounter_id, gt_id, modality, body_region,
                                    clinical_indication, ordering_provider, order_priority, order_datetime)
        VALUES (:eid, 0, :mod, :body, :indication, 'Agent', :priority, :dt)
        RETURNING order_id
    """)
    result = await db.execute(sql, {
        "eid": request.encounter_id,
        "mod": request.modality,
        "body": request.body_region,
        "indication": request.clinical_indication,
        "priority": request.priority,
        "dt": datetime.now(timezone.utc).isoformat(),
    })
    order_id = result.scalar_one()
    await db.commit()
    return OrderResponse(order_id=order_id, status="placed")


async def place_lab_order(
    db: AsyncSession, request: LabOrderRequest, session_id: str = ""
) -> OrderResponse:
    """Record a lab order in agent_submissions (no lab_orders table). Returns synthetic ID."""
    submission = AgentSubmission(
        submission_id=str(uuid.uuid4()),
        session_id=session_id,
        patient_id=request.patient_id,
        encounter_id=request.encounter_id,
        submission_type="lab_order",
        payload={
            "test_name": request.test_name,
            "loinc_code": request.loinc_code,
            "clinical_indication": request.clinical_indication,
            "priority": request.priority,
        },
    )
    db.add(submission)
    await db.commit()
    return OrderResponse(order_id=-1, status="recorded")


# ---------------------------------------------------------------------------
# Radiology
# ---------------------------------------------------------------------------

async def get_radiology_worklist(
    db: AsyncSession,
) -> list[WorklistItem]:
    """All imaging orders with patient info."""
    sql = text("""
        SELECT io.order_id, lp.patient_id, lp.profile, io.encounter_id,
               io.modality, io.body_region, io.clinical_indication,
               COALESCE(io.order_priority, 'routine') AS priority,
               io.order_datetime
        FROM imaging_orders io
        JOIN longitudinal_encounters le ON le.encounter_id = io.encounter_id
        JOIN longitudinal_patients lp ON lp.patient_id = le.patient_id
        ORDER BY io.order_id
    """)
    result = await db.execute(sql)
    items = []
    for r in result.fetchall():
        profile = r[2] if isinstance(r[2], dict) else {}
        items.append(WorklistItem(
            order_id=r[0],
            patient_id=r[1],
            patient_name=profile.get("name"),
            encounter_id=r[3],
            modality=r[4],
            body_region=r[5],
            clinical_indication=r[6],
            priority=r[7],
            order_datetime=r[8],
        ))
    return items


async def get_pre_read_context(
    db: AsyncSession, order_id: int, role: str
) -> PreReadContext | None:
    """Radiology pre-read context: order + encounter + relevant history."""
    # Get imaging order
    order = await db.get(ImagingOrder, order_id)
    if order is None:
        return None

    # Get encounter detail (role-filtered)
    enc_detail = await get_encounter_detail(db, order.encounter_id, role)
    if enc_detail is None:
        return None

    # Build worklist item from order
    patient = await db.get(LongitudinalPatient, enc_detail.patient_id)
    profile = patient.profile if patient else {}
    worklist = WorklistItem(
        order_id=order.order_id,
        patient_id=enc_detail.patient_id,
        patient_name=profile.get("name") if isinstance(profile, dict) else None,
        encounter_id=order.encounter_id,
        modality=order.modality,
        body_region=order.body_region,
        clinical_indication=order.clinical_indication,
        priority=(order.order_priority.value if order.order_priority else "routine"),
        order_datetime=order.order_datetime,
    )

    # Get relevant history: prior imaging + HPI from earlier encounters
    allowed = get_allowed_sections(role)
    history_types = {"imaging", "hpi", "labs"}
    if allowed is not None:
        history_types = history_types & allowed

    sql = text("""
        SELECT ees.id, ees.encounter_id, ees.section_type, ees.section_text, ees.section_order
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON le.encounter_id = ees.encounter_id
        WHERE le.patient_id = :pid
          AND le.encounter_id != :current_eid
          AND ees.section_type = ANY(:types)
        ORDER BY le.encounter_date DESC
        LIMIT 10
    """)
    result = await db.execute(sql, {
        "pid": enc_detail.patient_id,
        "current_eid": order.encounter_id,
        "types": list(history_types),
    })
    history = [
        SectionEntry(
            section_id=r[0], encounter_id=r[1],
            section_type=r[2], section_text=r[3], section_order=r[4],
        )
        for r in result.fetchall()
    ]

    return PreReadContext(order=worklist, encounter=enc_detail, relevant_history=history)


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

async def get_medications(
    db: AsyncSession, patient_id: int
) -> list[SectionEntry]:
    """Medication sections across all encounters for patient."""
    sql = text("""
        SELECT ees.id, ees.encounter_id, ees.section_type, ees.section_text, ees.section_order
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON le.encounter_id = ees.encounter_id
        WHERE le.patient_id = :pid AND ees.section_type = 'medications'
        ORDER BY le.encounter_date
    """)
    result = await db.execute(sql, {"pid": patient_id})
    return [
        SectionEntry(
            section_id=r[0], encounter_id=r[1],
            section_type=r[2], section_text=r[3], section_order=r[4],
        )
        for r in result.fetchall()
    ]


# ---------------------------------------------------------------------------
# Patient search (for agent tool)
# ---------------------------------------------------------------------------

async def search_patients(
    db: AsyncSession,
    name: str | None = None,
    gender: str | None = None,
    mrn: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search patients by name, gender, or MRN (patient_id)."""
    if mrn:
        patient = await db.get(LongitudinalPatient, int(mrn))
        if patient is None:
            return []
        profile = patient.profile or {}
        return [{
            "patient_id": patient.patient_id,
            "name": profile.get("name", f"Patient {patient.patient_id}"),
            "age": patient.age,
            "sex": patient.sex,
        }]

    # Search by profile JSONB name field or demographics
    conditions = []
    params: dict = {"limit": limit}

    if name:
        conditions.append("lp.profile->>'name' ILIKE :name")
        params["name"] = f"%{name}%"
    if gender:
        g = gender[0].upper() if gender else None
        if g in ("M", "F"):
            conditions.append("lp.sex = :sex")
            params["sex"] = g

    where = " AND ".join(conditions) if conditions else "TRUE"
    sql = text(f"""
        SELECT lp.patient_id, lp.profile, lp.age, lp.sex
        FROM longitudinal_patients lp
        WHERE {where}
        ORDER BY lp.patient_id
        LIMIT :limit
    """)
    result = await db.execute(sql, params)
    patients = []
    for r in result.fetchall():
        profile = r[1] if isinstance(r[1], dict) else {}
        patients.append({
            "patient_id": r[0],
            "name": profile.get("name", f"Patient {r[0]}"),
            "age": r[2],
            "sex": r[3],
        })
    return patients
