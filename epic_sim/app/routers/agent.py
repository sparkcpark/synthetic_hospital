"""Agent tool-use protocol — Anthropic Messages / OpenAI function-calling compatible.

Provides GET /agent/tools for tool schema discovery and
POST /agent/tools for tool execution with session tracking.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.auth.oauth2 import CurrentUser, get_current_user
from epic_sim.app.models.auth import AgentSubmission
from epic_sim.app.models.base import get_db
from epic_sim.app.schemas.epic import (
    AddProblemRequest,
    ImagingOrderRequest,
    LabOrderRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from epic_sim.app.services import epic_service, session_service
from epic_sim.app.services.session_service import get_redis

router = APIRouter()


# ---------------------------------------------------------------------------
# Tool Definitions (12 tools)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="search_patients",
        description="Search for patients by name, gender, or MRN (patient ID).",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Patient name to search for"},
                "gender": {"type": "string", "enum": ["male", "female"]},
                "mrn": {"type": "string", "description": "Patient ID / MRN"},
            },
        },
    ),
    ToolDefinition(
        name="open_chart",
        description="Open a patient's chart. Returns demographics, active problems, recent encounters, medications, and allergies.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
            },
            "required": ["patient_id"],
        },
    ),
    ToolDefinition(
        name="view_encounters",
        description="List all encounters for a patient in chronological order.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
            },
            "required": ["patient_id"],
        },
    ),
    ToolDefinition(
        name="view_encounter_detail",
        description="View a specific encounter with all available EHR sections (filtered by role).",
        parameters={
            "type": "object",
            "properties": {
                "encounter_id": {"type": "integer"},
            },
            "required": ["encounter_id"],
        },
    ),
    ToolDefinition(
        name="view_section",
        description="View a single EHR section by its ID.",
        parameters={
            "type": "object",
            "properties": {
                "section_id": {"type": "integer"},
            },
            "required": ["section_id"],
        },
    ),
    ToolDefinition(
        name="view_results",
        description="View lab, imaging, or pathology results for a patient.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "result_type": {"type": "string", "enum": ["labs", "imaging", "pathology"]},
            },
            "required": ["patient_id", "result_type"],
        },
    ),
    ToolDefinition(
        name="search_chart",
        description="Search across all EHR sections for a patient using keywords or clinical terms.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "query": {"type": "string", "description": "Search terms (e.g., 'chest pain', 'hemoglobin')"},
            },
            "required": ["patient_id", "query"],
        },
    ),
    ToolDefinition(
        name="view_problem_list",
        description="View the active problem list (diagnoses) for a patient.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
            },
            "required": ["patient_id"],
        },
    ),
    ToolDefinition(
        name="view_medications",
        description="View current and historical medications for a patient.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
            },
            "required": ["patient_id"],
        },
    ),
    ToolDefinition(
        name="submit_diagnosis",
        description="Submit a diagnosis prediction for a patient.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "diagnosis_name": {"type": "string"},
                "icd10_code": {"type": "string", "description": "ICD-10-CM code (optional)"},
            },
            "required": ["patient_id", "diagnosis_name"],
        },
    ),
    ToolDefinition(
        name="submit_summary",
        description="Submit a clinical context summary for a patient.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "summary": {"type": "string"},
            },
            "required": ["patient_id", "summary"],
        },
    ),
    ToolDefinition(
        name="submit_pre_read",
        description="Submit a radiology pre-read assessment for an imaging order.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "integer"},
                "summary": {"type": "string"},
                "impression": {"type": "string"},
                "findings": {"type": "string"},
            },
            "required": ["order_id", "summary"],
        },
    ),
    ToolDefinition(
        name="submit_rankings",
        description="Submit evidence passage relevance rankings for a set of diagnoses.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "rankings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "passage_id": {"type": "string"},
                            "grade": {"type": "integer", "minimum": 0, "maximum": 3},
                        },
                        "required": ["passage_id", "grade"],
                    },
                    "description": "Relevance rankings: passage_id + grade (0-3)",
                },
            },
            "required": ["patient_id", "rankings"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/tools")
async def get_tools():
    """Return tool definitions for agent integration."""
    return [t.model_dump() for t in TOOL_DEFINITIONS]


@router.post("/tools", response_model=ToolCallResponse)
async def execute_tool(
    request: ToolCallRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """Execute a tool call. Routes to service functions, tracks in session."""
    start = time.monotonic()

    # Rate limit check
    if redis and request.session_id:
        session = await session_service.get_session(redis, request.session_id)
        if session and session.api_call_count >= session.max_api_calls:
            raise HTTPException(429, "Rate limit exceeded")

    # Dispatch
    result = await _dispatch_tool(
        request.tool_name,
        request.arguments,
        db,
        user.role,
        request.session_id,
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Record in session
    api_call_count = 0
    remaining = 0
    if redis and request.session_id:
        try:
            api_call_count, remaining = await session_service.record_tool_call(
                redis, request.session_id, request.tool_name,
                request.arguments, elapsed_ms,
            )
        except HTTPException:
            raise
        except Exception:
            pass  # Session tracking failure shouldn't block tool execution

    return ToolCallResponse(
        session_id=request.session_id,
        tool_name=request.tool_name,
        result=result,
        api_call_count=api_call_count,
        remaining_calls=remaining,
    )


async def _dispatch_tool(
    tool: str,
    args: dict,
    db: AsyncSession,
    role: str,
    session_id: str = "",
) -> dict | list | str:
    """Route tool calls to service functions."""
    match tool:
        case "search_patients":
            patients = await epic_service.search_patients(
                db,
                name=args.get("name"),
                gender=args.get("gender"),
                mrn=args.get("mrn"),
                limit=args.get("limit", 10),
            )
            return patients

        case "open_chart":
            summary = await epic_service.get_patient_summary(
                db, args["patient_id"], role
            )
            return summary.model_dump()

        case "view_encounters":
            encs = await epic_service.get_encounters(
                db, args["patient_id"], role
            )
            return [e.model_dump() for e in encs]

        case "view_encounter_detail":
            detail = await epic_service.get_encounter_detail(
                db, args["encounter_id"], role
            )
            if detail is None:
                return {"error": "Encounter not found"}
            return detail.model_dump()

        case "view_section":
            section = await epic_service.get_encounter_section(
                db, args["section_id"], role
            )
            if section is None:
                return {"error": "Section not found or access denied"}
            return section.model_dump()

        case "view_results":
            rt = args["result_type"]
            pid = args["patient_id"]
            if rt == "labs":
                results = await epic_service.get_lab_results(db, pid)
            elif rt == "imaging":
                results = await epic_service.get_imaging_results(db, pid)
            elif rt == "pathology":
                results = await epic_service.get_pathology_results(db, pid)
            else:
                return {"error": f"Unknown result_type: {rt}"}
            return [r.model_dump() for r in results]

        case "search_chart":
            from epic_sim.app.services.search_service import search_service

            results = await search_service.hybrid_search(
                db, args["patient_id"], args["query"], role, limit=20,
            )
            return [s.model_dump() for s in results]

        case "view_problem_list":
            problems = await epic_service.get_problem_list(db, args["patient_id"])
            return [p.model_dump() for p in problems]

        case "view_medications":
            meds = await epic_service.get_medications(db, args["patient_id"])
            return [m.model_dump() for m in meds]

        case "submit_diagnosis":
            # Write to agent_submissions + return confirmation
            submission = AgentSubmission(
                submission_id=str(uuid.uuid4()),
                session_id=session_id,
                patient_id=args["patient_id"],
                submission_type="diagnosis",
                payload={
                    "diagnosis_name": args["diagnosis_name"],
                    "icd10_code": args.get("icd10_code"),
                },
            )
            db.add(submission)
            await db.commit()
            return {"status": "recorded", "patient_id": args["patient_id"]}

        case "submit_summary":
            submission = AgentSubmission(
                submission_id=str(uuid.uuid4()),
                session_id=session_id,
                patient_id=args["patient_id"],
                submission_type="summary",
                payload={"summary": args["summary"]},
            )
            db.add(submission)
            await db.commit()
            return {"status": "recorded", "patient_id": args["patient_id"]}

        case "submit_pre_read":
            submission = AgentSubmission(
                submission_id=str(uuid.uuid4()),
                session_id=session_id,
                submission_type="pre_read",
                payload={
                    "order_id": args["order_id"],
                    "summary": args["summary"],
                    "impression": args.get("impression"),
                    "findings": args.get("findings"),
                },
            )
            db.add(submission)
            await db.commit()
            return {"status": "recorded", "order_id": args["order_id"]}

        case "submit_rankings":
            submission = AgentSubmission(
                submission_id=str(uuid.uuid4()),
                session_id=session_id,
                patient_id=args.get("patient_id"),
                submission_type="rankings",
                payload={"rankings": args["rankings"]},
            )
            db.add(submission)
            await db.commit()
            return {"status": "recorded", "patient_id": args.get("patient_id")}

        case _:
            raise HTTPException(400, f"Unknown tool: {tool}")
