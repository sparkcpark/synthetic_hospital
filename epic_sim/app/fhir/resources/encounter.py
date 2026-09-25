"""FHIR R4 Encounter resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import (
    CodeableConcept,
    Coding,
    DomainResource,
    Period,
    Reference,
)

if TYPE_CHECKING:
    from epic_sim.app.models.longitudinal import LongitudinalEncounter

# Map internal encounter_type → FHIR Encounter.class
_CLASS_MAP = {
    "outpatient": Coding(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="AMB", display="ambulatory"),
    "ed": Coding(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="EMER", display="emergency"),
    "inpatient": Coding(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="IMP", display="inpatient encounter"),
    "icu": Coding(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="ACUTE", display="inpatient acute"),
    "follow_up": Coding(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="AMB", display="ambulatory"),
    "procedure": Coding(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="SS", display="short stay"),
    "telehealth": Coding(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="VR", display="virtual"),
}


class EncounterParticipant(DomainResource):
    resource_type: str = Field("EncounterParticipant", alias="resourceType", exclude=True)
    type: list[CodeableConcept] = Field(default_factory=list)
    individual: Reference | None = None


class Encounter(DomainResource):
    resource_type: str = Field("Encounter", alias="resourceType")
    identifier: list = Field(default_factory=list)
    status: str = "finished"
    class_field: Coding | None = Field(None, alias="class")
    type: list[CodeableConcept] = Field(default_factory=list)
    subject: Reference | None = None
    participant: list[dict] = Field(default_factory=list)
    period: Period | None = None
    reason_code: list[CodeableConcept] = Field(default_factory=list, alias="reasonCode")
    service_provider: Reference | None = Field(None, alias="serviceProvider")

    @classmethod
    def from_db(cls, enc: LongitudinalEncounter, base_url: str = "") -> Encounter:
        enc_type = enc.encounter_type.value if enc.encounter_type else "outpatient"
        class_coding = _CLASS_MAP.get(enc_type, _CLASS_MAP["outpatient"])

        # Period
        period = None
        if enc.encounter_date:
            period = Period(start=enc.encounter_date)

        # Reason
        reason_codes = []
        if enc.chief_complaint:
            reason_codes.append(CodeableConcept(text=enc.chief_complaint))

        # Participant (attending)
        participants = []
        if enc.attending_name:
            participants.append({
                "type": [CodeableConcept(
                    coding=[Coding(
                        system="http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
                        code="ATND",
                        display="attender",
                    )]
                ).model_dump(by_alias=True, exclude_none=True)],
                "individual": Reference(display=enc.attending_name).model_dump(by_alias=True, exclude_none=True),
            })

        # Type
        type_concepts = []
        if enc.department:
            type_concepts.append(CodeableConcept(text=enc.department))

        return cls(
            id=str(enc.encounter_id),
            status="finished",
            class_field=class_coding,
            type=type_concepts,
            subject=Reference(reference=f"Patient/{enc.patient_id}"),
            participant=participants,
            period=period,
            reason_code=reason_codes,
        )
