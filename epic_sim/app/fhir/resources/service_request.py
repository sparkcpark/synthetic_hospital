"""FHIR R4 ServiceRequest resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from epic_sim.app.fhir.types import CodeableConcept, Coding, DomainResource, Reference

if TYPE_CHECKING:
    from epic_sim.app.models.benchmark import ImagingOrder

# Modality → FHIR ServiceRequest code
_MODALITY_MAP = {
    "xray": ("XRAY", "X-Ray"),
    "ct": ("CT", "CT Scan"),
    "ct_angio": ("CTA", "CT Angiography"),
    "mri": ("MRI", "MRI"),
    "ultrasound": ("US", "Ultrasound"),
    "nuclear": ("NM", "Nuclear Medicine"),
    "fluoroscopy": ("FL", "Fluoroscopy"),
    "pet": ("PET", "PET Scan"),
    "mammography": ("MG", "Mammography"),
}


class ServiceRequest(DomainResource):
    resource_type: str = Field("ServiceRequest", alias="resourceType")
    status: str = "active"
    intent: str = "order"
    category: list[CodeableConcept] = Field(default_factory=list)
    priority: str | None = None
    code: CodeableConcept | None = None
    subject: Reference | None = None
    encounter: Reference | None = None
    authored_on: str | None = Field(None, alias="authoredOn")
    requester: Reference | None = None
    reason_code: list[CodeableConcept] = Field(default_factory=list, alias="reasonCode")
    body_site: list[CodeableConcept] = Field(default_factory=list, alias="bodySite")

    @classmethod
    def from_db(cls, order: ImagingOrder, patient_id: int) -> ServiceRequest:
        # Code from modality
        mod_code, mod_display = _MODALITY_MAP.get(order.modality, (order.modality, order.modality))
        code = CodeableConcept(
            coding=[Coding(
                system="http://dicom.nema.org/resources/ontology/DCM",
                code=mod_code,
                display=mod_display,
            )],
            text=f"{mod_display} - {order.body_region}",
        )

        # Category = imaging
        category = [
            CodeableConcept(
                coding=[Coding(
                    system="http://snomed.info/sct",
                    code="363679005",
                    display="Imaging",
                )]
            )
        ]

        # Reason
        reason_codes = []
        if order.clinical_indication:
            reason_codes.append(CodeableConcept(text=order.clinical_indication))

        # Body site
        body_sites = []
        if order.body_region:
            body_sites.append(CodeableConcept(text=order.body_region))

        # Requester
        requester = Reference(display=order.ordering_provider) if order.ordering_provider else None

        # Priority
        priority = order.order_priority.value if order.order_priority else "routine"

        return cls(
            id=str(order.order_id),
            status="active",
            intent="order",
            category=category,
            priority=priority,
            code=code,
            subject=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{order.encounter_id}"),
            authored_on=order.order_datetime,
            requester=requester,
            reason_code=reason_codes,
            body_site=body_sites,
        )
