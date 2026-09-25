"""FHIR R4 Patient resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import (
    Address,
    CodeableConcept,
    Coding,
    ContactPoint,
    DomainResource,
    HumanName,
    Identifier,
    Reference,
)

if TYPE_CHECKING:
    from epic_sim.app.models.longitudinal import LongitudinalPatient


class Patient(DomainResource):
    resource_type: str = Field("Patient", alias="resourceType")
    identifier: list[Identifier] = Field(default_factory=list)
    active: bool = True
    name: list[HumanName] = Field(default_factory=list)
    telecom: list[ContactPoint] = Field(default_factory=list)
    gender: str | None = None
    birth_date: str | None = Field(None, alias="birthDate")
    address: list[Address] = Field(default_factory=list)
    marital_status: CodeableConcept | None = Field(None, alias="maritalStatus")
    general_practitioner: list[Reference] = Field(default_factory=list, alias="generalPractitioner")

    @classmethod
    def from_db(cls, patient: LongitudinalPatient, base_url: str = "") -> Patient:
        profile = patient.profile or {}

        # Name from profile
        name_parts = []
        full_name = profile.get("name", f"Patient {patient.patient_id}")
        if full_name:
            parts = full_name.split()
            name_parts.append(
                HumanName(
                    use="official",
                    text=full_name,
                    family=parts[-1] if parts else None,
                    given=parts[:-1] if len(parts) > 1 else [],
                )
            )

        # Gender mapping
        sex = patient.sex or profile.get("sex", "")
        gender_map = {"M": "male", "F": "female"}
        gender = gender_map.get(sex, "unknown")

        # Birth date: compute from age
        birth_date = None
        if patient.age:
            # Approximate birth year (patient data generated ~2024)
            birth_year = 2024 - patient.age
            birth_date = f"{birth_year}-01-01"

        # PCP
        gp = []
        if patient.pcp_name:
            gp.append(Reference(display=patient.pcp_name))

        # MRN identifier
        identifiers = [
            Identifier(
                use="usual",
                type=CodeableConcept(
                    coding=[Coding(system="http://terminology.hl7.org/CodeSystem/v2-0203", code="MR")],
                    text="Medical Record Number",
                ),
                system=f"{base_url}/mrn" if base_url else "urn:epic:mrn",
                value=str(patient.patient_id),
            )
        ]

        return cls(
            id=str(patient.patient_id),
            identifier=identifiers,
            name=name_parts,
            gender=gender,
            birth_date=birth_date,
            general_practitioner=gp,
        )
