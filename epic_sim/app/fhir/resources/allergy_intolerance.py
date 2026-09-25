"""FHIR R4 AllergyIntolerance resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import CodeableConcept, Coding, DomainResource, Reference

if TYPE_CHECKING:
    from epic_sim.app.models.longitudinal import EncounterEhrSection


class AllergyIntolerance(DomainResource):
    resource_type: str = Field("AllergyIntolerance", alias="resourceType")
    clinical_status: CodeableConcept | None = Field(None, alias="clinicalStatus")
    verification_status: CodeableConcept | None = Field(None, alias="verificationStatus")
    type: str | None = None
    category: list[str] = Field(default_factory=list)
    patient: Reference | None = None
    encounter: Reference | None = None
    recorded_date: str | None = Field(None, alias="recordedDate")
    note: list[dict] = Field(default_factory=list)

    @classmethod
    def from_ehr_section(
        cls,
        section: EncounterEhrSection,
        patient_id: int,
        encounter_date: str | None = None,
    ) -> AllergyIntolerance:
        """Create an AllergyIntolerance from an allergies EHR section.

        Since our data stores allergies as free-text, we wrap the section
        text as a note on the resource.
        """
        clinical_status = CodeableConcept(
            coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                code="active",
                display="Active",
            )]
        )

        verification_status = CodeableConcept(
            coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
                code="confirmed",
                display="Confirmed",
            )]
        )

        return cls(
            id=str(section.id),
            clinical_status=clinical_status,
            verification_status=verification_status,
            patient=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{section.encounter_id}"),
            recorded_date=encounter_date,
            note=[{"text": section.section_text}],
        )
