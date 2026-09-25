"""FHIR scope definitions for role-based access control.

Scope format follows SMART on FHIR v2:
  patient/<ResourceType>.<read|write|*>
"""

# All available FHIR scopes
ALL_SCOPES = {
    "patient/Patient.read",
    "patient/Encounter.read",
    "patient/Condition.read",
    "patient/Observation.read",
    "patient/DiagnosticReport.read",
    "patient/ServiceRequest.read",
    "patient/DocumentReference.read",
    "patient/MedicationRequest.read",
    "patient/AllergyIntolerance.read",
    "patient/Patient.write",
    "patient/Encounter.write",
    "patient/Condition.write",
    "patient/ServiceRequest.write",
}

# Role → default scopes
ROLE_SCOPES: dict[str, set[str]] = {
    "attending": {
        "patient/Patient.read",
        "patient/Encounter.read",
        "patient/Condition.read",
        "patient/Observation.read",
        "patient/DiagnosticReport.read",
        "patient/ServiceRequest.read",
        "patient/DocumentReference.read",
        "patient/MedicationRequest.read",
        "patient/AllergyIntolerance.read",
        "patient/Condition.write",
        "patient/ServiceRequest.write",
    },
    "resident": {
        "patient/Patient.read",
        "patient/Encounter.read",
        "patient/Condition.read",
        "patient/Observation.read",
        "patient/DiagnosticReport.read",
        "patient/ServiceRequest.read",
        "patient/DocumentReference.read",
        "patient/MedicationRequest.read",
        "patient/AllergyIntolerance.read",
    },
    "nurse": {
        "patient/Patient.read",
        "patient/Observation.read",
        "patient/MedicationRequest.read",
        "patient/AllergyIntolerance.read",
    },
    "radiologist": {
        "patient/Patient.read",
        "patient/Encounter.read",
        "patient/Condition.read",
        "patient/DiagnosticReport.read",
        "patient/ServiceRequest.read",
        "patient/DocumentReference.read",
        "patient/Observation.read",
    },
    "lab_tech": {
        "patient/Patient.read",
        "patient/Observation.read",
    },
    "pharmacist": {
        "patient/Patient.read",
        "patient/MedicationRequest.read",
        "patient/AllergyIntolerance.read",
        "patient/Observation.read",
    },
}


def scopes_for_role(role: str) -> set[str]:
    """Return the FHIR scopes granted to a role."""
    return ROLE_SCOPES.get(role, set())


def has_scope(user_scopes: list[str], required: str) -> bool:
    """Check if the user's scopes include the required scope."""
    return required in user_scopes
