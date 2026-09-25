"""FHIR R4 MedicationRequest resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import CodeableConcept, Coding, DomainResource, Reference

if TYPE_CHECKING:
    from epic_sim.app.models.longitudinal import EncounterEhrSection


class MedicationRequest(DomainResource):
    resource_type: str = Field("MedicationRequest", alias="resourceType")
    status: str = "active"
    intent: str = "order"
    medication_codeable_concept: CodeableConcept | None = Field(None, alias="medicationCodeableConcept")
    subject: Reference | None = None
    encounter: Reference | None = None
    authored_on: str | None = Field(None, alias="authoredOn")
    note: list[dict] = Field(default_factory=list)

    @classmethod
    def from_ehr_section(
        cls,
        section: EncounterEhrSection,
        patient_id: int,
        encounter_date: str | None = None,
    ) -> MedicationRequest:
        """Create a MedicationRequest from a medications EHR section.

        Since our data stores medications as free-text sections rather than
        structured med lists, we create a single MedicationRequest per section
        with the full text in the note field.
        """
        return cls(
            id=str(section.id),
            status="active",
            intent="order",
            medication_codeable_concept=CodeableConcept(text="Medication List"),
            subject=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{section.encounter_id}"),
            authored_on=encounter_date,
            note=[{"text": section.section_text}],
        )
