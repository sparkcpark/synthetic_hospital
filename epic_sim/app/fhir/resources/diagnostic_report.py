"""FHIR R4 DiagnosticReport resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import CodeableConcept, Coding, DomainResource, Reference

if TYPE_CHECKING:
    from epic_sim.app.models.longitudinal import EncounterEhrSection


class DiagnosticReport(DomainResource):
    resource_type: str = Field("DiagnosticReport", alias="resourceType")
    status: str = "final"
    category: list[CodeableConcept] = Field(default_factory=list)
    code: CodeableConcept | None = None
    subject: Reference | None = None
    encounter: Reference | None = None
    effective_date_time: str | None = Field(None, alias="effectiveDateTime")
    conclusion: str | None = None
    conclusion_code: list[CodeableConcept] = Field(default_factory=list, alias="conclusionCode")

    @classmethod
    def from_ehr_section(
        cls,
        section: EncounterEhrSection,
        patient_id: int,
        encounter_date: str | None = None,
        section_type: str = "imaging",
    ) -> DiagnosticReport:
        # Category
        cat_map = {
            "imaging": ("RAD", "Radiology"),
            "pathology": ("SP", "Surgical Pathology"),
            "labs": ("LAB", "Laboratory"),
        }
        cat_code, cat_display = cat_map.get(section_type, ("OTH", "Other"))
        category = [
            CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/v2-0074",
                    code=cat_code,
                    display=cat_display,
                )]
            )
        ]

        code = CodeableConcept(text=f"{section_type.title()} Report")

        return cls(
            id=str(section.id),
            category=category,
            code=code,
            subject=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{section.encounter_id}"),
            effective_date_time=encounter_date,
            conclusion=section.section_text,
        )
