"""Role-based access control — section filtering and patient visibility.

Implements partial observability: different roles see different EHR sections
for the same patient, mirroring real Epic access controls.
"""

from __future__ import annotations

# Role → allowed encounter_ehr_sections.section_type values
# None = all sections (full access)
ROLE_SECTION_ACCESS: dict[str, set[str] | None] = {
    "attending": None,  # Full access
    "resident": None,   # Full access
    "nurse": {
        "demographics",
        "chief_complaint",
        "vitals",
        "medications",
        "allergies",
        "physical_exam",
        "ros",
    },
    "radiologist": {
        "demographics",
        "chief_complaint",
        "hpi",
        "pmh",
        "medications",
        "imaging",
        "labs",
        "assessment",
    },
    "lab_tech": {
        "demographics",
        "labs",
    },
    "pharmacist": {
        "demographics",
        "medications",
        "allergies",
        "labs",
        "pmh",
    },
}

# FHIR resource type → required scope for read access
RESOURCE_READ_SCOPES: dict[str, str] = {
    "Patient": "patient/Patient.read",
    "Encounter": "patient/Encounter.read",
    "Condition": "patient/Condition.read",
    "Observation": "patient/Observation.read",
    "DiagnosticReport": "patient/DiagnosticReport.read",
    "ServiceRequest": "patient/ServiceRequest.read",
    "DocumentReference": "patient/DocumentReference.read",
    "MedicationRequest": "patient/MedicationRequest.read",
    "AllergyIntolerance": "patient/AllergyIntolerance.read",
}


def get_allowed_sections(role: str) -> set[str] | None:
    """Return the set of section types a role can access, or None for full access."""
    return ROLE_SECTION_ACCESS.get(role)


def filter_sections(sections: list[dict], role: str) -> tuple[list[dict], list[str]]:
    """Filter EHR sections based on role access.

    Returns (filtered_sections, removed_section_types).
    """
    allowed = get_allowed_sections(role)
    if allowed is None:
        return sections, []

    filtered = []
    removed = []
    for section in sections:
        st = section.get("section_type", "")
        if st in allowed:
            filtered.append(section)
        else:
            removed.append(st)
    return filtered, removed


def can_access_resource(role: str, resource_type: str, user_scopes: list[str]) -> bool:
    """Check if a role with given scopes can access a FHIR resource type."""
    required_scope = RESOURCE_READ_SCOPES.get(resource_type)
    if required_scope is None:
        return True  # No scope requirement (e.g., metadata)
    return required_scope in user_scopes
