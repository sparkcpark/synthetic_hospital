"""Tests for RBAC and partial observability.

Verifies that different roles see different EHR sections for the same patient,
matching real Epic access controls.
"""

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio(loop_scope="session")

KNOWN_PATIENT_ID = 1672


async def test_attending_sees_all_sections(client: AsyncClient, attending_token: str):
    """Attending physician should see ALL section types."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(
        f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}&_count=100",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    section_types = {
        e["resource"]["type"]["text"]
        for e in data["entry"]
    }

    # Attending should see these types
    expected = {"Chief Complaint", "Hpi", "Pmh", "Medications", "Allergies", "Vitals", "Physical Exam"}
    assert expected.issubset(section_types), f"Missing: {expected - section_types}"


async def test_radiologist_filtered_sections(client: AsyncClient, radiologist_token: str):
    """Radiologist should see only imaging-relevant sections."""
    headers = {"Authorization": f"Bearer {radiologist_token}"}
    resp = await client.get(
        f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}&_count=100",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    section_types = {
        e["resource"]["type"]["text"]
        for e in data["entry"]
    }

    # Radiologist SHOULD see
    allowed = {"Demographics", "Chief Complaint", "Hpi", "Pmh", "Medications", "Imaging", "Labs", "Assessment"}
    for st in section_types:
        assert st in allowed, f"Radiologist saw unauthorized section: {st}"

    # Radiologist should NOT see
    forbidden = {"Physical Exam", "Family History", "Social History", "Psh", "Ros"}
    assert not section_types.intersection(forbidden), f"Radiologist saw forbidden: {section_types & forbidden}"


async def test_radiologist_fewer_sections_than_attending(
    client: AsyncClient,
    attending_token: str,
    radiologist_token: str,
):
    """Radiologist should see strictly fewer sections than attending."""
    att_headers = {"Authorization": f"Bearer {attending_token}"}
    rad_headers = {"Authorization": f"Bearer {radiologist_token}"}

    att_resp = await client.get(
        f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}&_count=100",
        headers=att_headers,
    )
    rad_resp = await client.get(
        f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}&_count=100",
        headers=rad_headers,
    )

    att_total = att_resp.json()["total"]
    rad_total = rad_resp.json()["total"]

    assert rad_total < att_total, (
        f"Radiologist ({rad_total}) should see fewer sections than attending ({att_total})"
    )


async def test_nurse_no_document_reference(client: AsyncClient, nurse_token: str):
    """Nurse should be denied DocumentReference access (no scope)."""
    headers = {"Authorization": f"Bearer {nurse_token}"}
    resp = await client.get(
        f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}",
        headers=headers,
    )
    assert resp.status_code == 403


async def test_nurse_can_see_observations(client: AsyncClient, nurse_token: str):
    """Nurse CAN see Observations (has patient/Observation.read scope)."""
    headers = {"Authorization": f"Bearer {nurse_token}"}
    resp = await client.get(
        f"/fhir/Observation?patient={KNOWN_PATIENT_ID}&_count=3",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0


async def test_nurse_no_encounter_access(client: AsyncClient, nurse_token: str):
    """Nurse should be denied Encounter access."""
    headers = {"Authorization": f"Bearer {nurse_token}"}
    resp = await client.get(
        f"/fhir/Encounter?patient={KNOWN_PATIENT_ID}",
        headers=headers,
    )
    assert resp.status_code == 403


async def test_nurse_no_condition_access(client: AsyncClient, nurse_token: str):
    """Nurse should be denied Condition access."""
    headers = {"Authorization": f"Bearer {nurse_token}"}
    resp = await client.get(
        f"/fhir/Condition?patient={KNOWN_PATIENT_ID}",
        headers=headers,
    )
    assert resp.status_code == 403


async def test_lab_tech_no_document_reference(client: AsyncClient, lab_tech_token: str):
    """Lab tech should be denied DocumentReference access."""
    headers = {"Authorization": f"Bearer {lab_tech_token}"}
    resp = await client.get(
        f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}",
        headers=headers,
    )
    assert resp.status_code == 403


async def test_lab_tech_can_see_observations(client: AsyncClient, lab_tech_token: str):
    """Lab tech CAN see Observations."""
    headers = {"Authorization": f"Bearer {lab_tech_token}"}
    resp = await client.get(
        f"/fhir/Observation?patient={KNOWN_PATIENT_ID}&_count=3",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0


async def test_diagnostic_report_filtered_by_role(
    client: AsyncClient,
    attending_token: str,
    radiologist_token: str,
):
    """DiagnosticReport should respect role-based section filtering."""
    att_headers = {"Authorization": f"Bearer {attending_token}"}
    rad_headers = {"Authorization": f"Bearer {radiologist_token}"}

    att_resp = await client.get(
        f"/fhir/DiagnosticReport?patient={KNOWN_PATIENT_ID}",
        headers=att_headers,
    )
    rad_resp = await client.get(
        f"/fhir/DiagnosticReport?patient={KNOWN_PATIENT_ID}",
        headers=rad_headers,
    )

    assert att_resp.status_code == 200
    assert rad_resp.status_code == 200

    # Both should have some reports (imaging, labs exist for this patient)
    att_total = att_resp.json()["total"]
    rad_total = rad_resp.json()["total"]
    assert att_total > 0
    assert rad_total > 0
