"""FHIR R4 resource models."""

from epic_sim.app.fhir.resources.allergy_intolerance import AllergyIntolerance
from epic_sim.app.fhir.resources.condition import Condition
from epic_sim.app.fhir.resources.diagnostic_report import DiagnosticReport
from epic_sim.app.fhir.resources.document_reference import DocumentReference
from epic_sim.app.fhir.resources.encounter import Encounter
from epic_sim.app.fhir.resources.medication_request import MedicationRequest
from epic_sim.app.fhir.resources.observation import Observation
from epic_sim.app.fhir.resources.patient import Patient
from epic_sim.app.fhir.resources.service_request import ServiceRequest

__all__ = [
    "AllergyIntolerance",
    "Condition",
    "DiagnosticReport",
    "DocumentReference",
    "Encounter",
    "MedicationRequest",
    "Observation",
    "Patient",
    "ServiceRequest",
]
