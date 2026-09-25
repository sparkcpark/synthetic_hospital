"""FHIR R4 Observation resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import CodeableConcept, Coding, DomainResource, Quantity, Reference

if TYPE_CHECKING:
    from epic_sim.app.models.ontology import ClinicalFinding
    from epic_sim.app.models.relationships import QuestionFinding

# Map finding_type → FHIR Observation category
_CATEGORY_MAP = {
    "lab_value": ("laboratory", "Laboratory"),
    "vital_sign": ("vital-signs", "Vital Signs"),
    "imaging_finding": ("imaging", "Imaging"),
    "procedure_result": ("procedure", "Procedure"),
    "symptom": ("exam", "Exam"),
    "sign": ("exam", "Exam"),
    "history_item": ("social-history", "Social History"),
    "medication": ("therapy", "Therapy"),
    "demographic": ("social-history", "Social History"),
}


class Observation(DomainResource):
    resource_type: str = Field("Observation", alias="resourceType")
    status: str = "final"
    category: list[CodeableConcept] = Field(default_factory=list)
    code: CodeableConcept | None = None
    subject: Reference | None = None
    encounter: Reference | None = None
    value_quantity: Quantity | None = Field(None, alias="valueQuantity")
    value_string: str | None = Field(None, alias="valueString")
    value_codeable_concept: CodeableConcept | None = Field(None, alias="valueCodeableConcept")
    interpretation: list[CodeableConcept] = Field(default_factory=list)
    note: list[dict] = Field(default_factory=list)

    @classmethod
    def from_db(
        cls,
        cf: ClinicalFinding,
        qf: QuestionFinding | None = None,
        patient_id: int | None = None,
        encounter_id: int | None = None,
    ) -> Observation:
        # Code — LOINC primary for lab_value, SNOMED secondary
        codings = []
        if hasattr(cf, "loinc_code") and cf.loinc_code:
            codings.append(Coding(
                system="http://loinc.org",
                code=cf.loinc_code,
                display=cf.loinc_desc or cf.display_name,
            ))
        if cf.snomed_id:
            codings.append(Coding(
                system="http://snomed.info/sct",
                code=cf.snomed_id,
                display=cf.snomed_desc or cf.display_name,
            ))
        code = CodeableConcept(coding=codings, text=cf.display_name)

        # Category based on finding_type
        finding_type = cf.finding_type.value if cf.finding_type else "symptom"
        cat_code, cat_display = _CATEGORY_MAP.get(finding_type, ("exam", "Exam"))
        category = [
            CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/observation-category",
                    code=cat_code,
                    display=cat_display,
                )]
            )
        ]

        # Value from QuestionFinding if available
        value_string = None
        value_quantity = None
        if qf:
            if qf.value_numeric is not None:
                value_quantity = Quantity(value=qf.value_numeric)
            elif qf.value_text:
                value_string = qf.value_text

        subject = Reference(reference=f"Patient/{patient_id}") if patient_id else None
        encounter_ref = Reference(reference=f"Encounter/{encounter_id}") if encounter_id else None

        return cls(
            id=str(cf.finding_id),
            category=category,
            code=code,
            subject=subject,
            encounter=encounter_ref,
            value_quantity=value_quantity,
            value_string=value_string,
        )
