"""FHIR R4 DocumentReference resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import Attachment, CodeableConcept, Coding, DomainResource, Reference

if TYPE_CHECKING:
    from epic_sim.app.models.longitudinal import EncounterEhrSection

# Map section_type → LOINC document type
_LOINC_MAP = {
    "hpi": ("10164-2", "History of present illness"),
    "pmh": ("11348-0", "Past medical history"),
    "psh": ("10167-5", "Past surgical history"),
    "medications": ("10160-0", "Medication list"),
    "allergies": ("48765-2", "Allergies and adverse reactions"),
    "family_history": ("10157-6", "Family history"),
    "social_history": ("29762-2", "Social history"),
    "ros": ("10187-3", "Review of systems"),
    "vitals": ("8716-3", "Vital signs"),
    "physical_exam": ("29545-1", "Physical examination"),
    "labs": ("26436-6", "Laboratory studies"),
    "imaging": ("18748-4", "Diagnostic imaging study"),
    "pathology": ("27898-6", "Pathology studies"),
    "assessment": ("51848-0", "Assessment"),
    "plan": ("18776-5", "Plan of care"),
    "demographics": ("52460-3", "Patient information"),
    "chief_complaint": ("10154-3", "Chief complaint"),
    "other_studies": ("30954-2", "Relevant diagnostic tests"),
}


class DocumentContent(DomainResource):
    resource_type: str = Field("DocumentContent", alias="resourceType", exclude=True)
    attachment: Attachment | None = None


class DocumentContext(DomainResource):
    resource_type: str = Field("DocumentContext", alias="resourceType", exclude=True)
    encounter: list[Reference] = Field(default_factory=list)
    period: dict | None = None


class DocumentReference(DomainResource):
    resource_type: str = Field("DocumentReference", alias="resourceType")
    status: str = "current"
    type: CodeableConcept | None = None
    category: list[CodeableConcept] = Field(default_factory=list)
    subject: Reference | None = None
    date: str | None = None
    content: list[dict] = Field(default_factory=list)
    context: dict | None = None

    @classmethod
    def from_ehr_section(
        cls,
        section: EncounterEhrSection,
        patient_id: int,
        encounter_date: str | None = None,
    ) -> DocumentReference:
        section_type = section.section_type

        # Type from LOINC
        loinc_code, loinc_display = _LOINC_MAP.get(section_type, ("47420-5", "Functional status assessment note"))
        doc_type = CodeableConcept(
            coding=[Coding(
                system="http://loinc.org",
                code=loinc_code,
                display=loinc_display,
            )],
            text=section_type.replace("_", " ").title(),
        )

        # Category = clinical-note
        category = [
            CodeableConcept(
                coding=[Coding(
                    system="http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category",
                    code="clinical-note",
                    display="Clinical Note",
                )]
            )
        ]

        # Content as inline text attachment
        content = [{
            "attachment": Attachment(
                content_type="text/plain",
                title=section_type.replace("_", " ").title(),
            ).model_dump(by_alias=True, exclude_none=True),
        }]

        # Store section_text inline (Epic pattern for small documents)
        content[0]["attachment"]["data"] = section.section_text

        context = {
            "encounter": [Reference(reference=f"Encounter/{section.encounter_id}").model_dump(by_alias=True, exclude_none=True)],
        }

        return cls(
            id=str(section.id),
            type=doc_type,
            category=category,
            subject=Reference(reference=f"Patient/{patient_id}"),
            date=encounter_date,
            content=content,
            context=context,
        )
