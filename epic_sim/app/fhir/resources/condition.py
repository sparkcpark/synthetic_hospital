"""FHIR R4 Condition resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import CodeableConcept, Coding, DomainResource, Reference

if TYPE_CHECKING:
    from epic_sim.app.models.ontology import Diagnosis


class Condition(DomainResource):
    resource_type: str = Field("Condition", alias="resourceType")
    clinical_status: CodeableConcept | None = Field(None, alias="clinicalStatus")
    verification_status: CodeableConcept | None = Field(None, alias="verificationStatus")
    category: list[CodeableConcept] = Field(default_factory=list)
    severity: CodeableConcept | None = None
    code: CodeableConcept | None = None
    subject: Reference | None = None
    encounter: Reference | None = None
    onset_string: str | None = Field(None, alias="onsetString")
    recorded_date: str | None = Field(None, alias="recordedDate")

    @classmethod
    def from_db(
        cls,
        dx: Diagnosis,
        patient_id: int | None = None,
        encounter_id: int | None = None,
        category_code: str = "encounter-diagnosis",
    ) -> Condition:
        # Build CodeableConcept for the diagnosis
        codings = []
        if dx.icd10_code:
            codings.append(Coding(
                system="http://hl7.org/fhir/sid/icd-10-cm",
                code=dx.icd10_code,
                display=dx.icd10_desc or dx.display_name,
            ))
        if dx.snomed_id:
            codings.append(Coding(
                system="http://snomed.info/sct",
                code=dx.snomed_id,
                display=dx.snomed_desc or dx.display_name,
            ))

        code = CodeableConcept(coding=codings, text=dx.display_name)

        # Clinical status
        clinical_status = CodeableConcept(
            coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/condition-clinical",
                code="active",
                display="Active",
            )]
        )

        # Verification status
        verification_status = CodeableConcept(
            coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/condition-verificationstatus",
                code="confirmed",
                display="Confirmed",
            )]
        )

        # Category
        category = [
            CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/condition-category",
                    code=category_code,
                    display=category_code.replace("-", " ").title(),
                )]
            )
        ]

        # Subject and encounter refs
        subject = Reference(reference=f"Patient/{patient_id}") if patient_id else None
        encounter_ref = Reference(reference=f"Encounter/{encounter_id}") if encounter_id else None

        return cls(
            id=str(dx.diagnosis_id),
            clinical_status=clinical_status,
            verification_status=verification_status,
            category=category,
            code=code,
            subject=subject,
            encounter=encounter_ref,
        )
