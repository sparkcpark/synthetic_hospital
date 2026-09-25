"""Pydantic request/response models for Epic simulation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chart Review
# ---------------------------------------------------------------------------

class ProblemEntry(BaseModel):
    diagnosis_id: int
    display_name: str
    icd10_code: str | None = None
    snomed_id: str | None = None


class EncounterSummary(BaseModel):
    encounter_id: int
    date: str | None = None
    type: str | None = None
    chief_complaint: str | None = None
    department: str | None = None
    attending: str | None = None


class SectionEntry(BaseModel):
    section_id: int
    encounter_id: int
    section_type: str
    section_text: str
    section_order: int


class PatientSummary(BaseModel):
    patient_id: int
    name: str | None = None
    age: int | None = None
    sex: str | None = None
    race_ethnicity: str | None = None
    insurance: str | None = None
    pcp: str | None = None
    active_problems: list[ProblemEntry] = Field(default_factory=list)
    recent_encounters: list[EncounterSummary] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


class EncounterDetail(BaseModel):
    encounter_id: int
    patient_id: int
    date: str | None = None
    type: str | None = None
    chief_complaint: str | None = None
    department: str | None = None
    attending: str | None = None
    sections: list[SectionEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class LabResult(BaseModel):
    encounter_id: int
    encounter_date: str | None = None
    section_text: str


class ImagingResult(BaseModel):
    encounter_id: int
    encounter_date: str | None = None
    section_text: str
    has_order: bool = False


class PathologyResult(BaseModel):
    encounter_id: int
    encounter_date: str | None = None
    section_text: str


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

class ImagingOrderRequest(BaseModel):
    patient_id: int
    encounter_id: int
    modality: str
    body_region: str
    clinical_indication: str
    priority: str = "routine"


class LabOrderRequest(BaseModel):
    patient_id: int
    encounter_id: int
    test_name: str
    loinc_code: str | None = None
    clinical_indication: str
    priority: str = "routine"


class OrderResponse(BaseModel):
    order_id: int
    status: str = "placed"


class AddProblemRequest(BaseModel):
    display_name: str
    icd10_code: str | None = None


# ---------------------------------------------------------------------------
# Radiology
# ---------------------------------------------------------------------------

class WorklistItem(BaseModel):
    order_id: int
    patient_id: int
    patient_name: str | None = None
    encounter_id: int
    modality: str
    body_region: str
    clinical_indication: str
    priority: str
    order_datetime: str | None = None


class PreReadContext(BaseModel):
    order: WorklistItem
    encounter: EncounterDetail
    relevant_history: list[SectionEntry] = Field(default_factory=list)


class PreReadSubmission(BaseModel):
    summary: str
    impression: str | None = None
    findings: str | None = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    gt_id: int | None = None
    patient_id: int | None = None
    role: str = "attending"
    department: str = "Internal Medicine"
    max_api_calls: int | None = None


class SessionResponse(BaseModel):
    session_id: str
    patient_id: int | None = None
    role: str
    department: str
    api_call_count: int = 0
    max_api_calls: int = 50
    status: str = "active"
    created_at: str = ""


# ---------------------------------------------------------------------------
# Agent Tool Protocol
# ---------------------------------------------------------------------------

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


class ToolCallRequest(BaseModel):
    session_id: str = ""
    tool_name: str
    arguments: dict = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    session_id: str = ""
    tool_name: str
    result: dict | list | str | None = None
    api_call_count: int = 0
    remaining_calls: int = 0
