"""Tests for FHIR R4 endpoints."""

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio(loop_scope="session")

# Patient IDs start at 1672 based on our data
KNOWN_PATIENT_ID = 1672


async def test_patient_search(client: AsyncClient, attending_token: str):
    """Patient search returns a Bundle with correct total."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get("/fhir/Patient?_count=5", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Bundle"
    assert data["type"] == "searchset"
    assert data["total"] == 1268  # Known count from migration
    assert len(data["entry"]) == 5


async def test_patient_read(client: AsyncClient, attending_token: str):
    """Read a specific patient."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Patient/{KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Patient"
    assert data["id"] == str(KNOWN_PATIENT_ID)
    assert len(data["identifier"]) > 0
    assert data["identifier"][0]["value"] == str(KNOWN_PATIENT_ID)


async def test_patient_not_found(client: AsyncClient, attending_token: str):
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get("/fhir/Patient/999999", headers=headers)
    assert resp.status_code == 404


async def test_encounter_search(client: AsyncClient, attending_token: str):
    """Encounter search for a patient returns encounters."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Encounter?patient={KNOWN_PATIENT_ID}&_count=50", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    # Check encounter structure
    entry = data["entry"][0]["resource"]
    assert entry["resourceType"] == "Encounter"
    assert entry["status"] == "finished"
    assert "class" in entry
    assert "subject" in entry
    assert entry["subject"]["reference"] == f"Patient/{KNOWN_PATIENT_ID}"


async def test_encounter_date_filter(client: AsyncClient, attending_token: str):
    """Encounter search with date filter."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(
        f"/fhir/Encounter?patient={KNOWN_PATIENT_ID}&date=ge2020-06-01",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    # All returned encounters should be on or after 2020-06-01
    for entry in data["entry"]:
        period = entry["resource"].get("period", {})
        assert period.get("start", "9999") >= "2020-06-01"


async def test_condition_search(client: AsyncClient, attending_token: str):
    """Condition search returns diagnoses for a patient."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Condition?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    # Check condition structure
    cond = data["entry"][0]["resource"]
    assert cond["resourceType"] == "Condition"
    assert "code" in cond
    assert "clinicalStatus" in cond


async def test_condition_requires_patient(client: AsyncClient, attending_token: str):
    """Condition search requires patient parameter."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get("/fhir/Condition", headers=headers)
    assert resp.status_code == 400


async def test_observation_search(client: AsyncClient, attending_token: str):
    """Observation search returns clinical findings."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/Observation?patient={KNOWN_PATIENT_ID}&_count=5", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    obs = data["entry"][0]["resource"]
    assert obs["resourceType"] == "Observation"
    assert "category" in obs
    assert "code" in obs


async def test_document_reference_search(client: AsyncClient, attending_token: str):
    """DocumentReference search returns EHR sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/DocumentReference?patient={KNOWN_PATIENT_ID}&_count=5", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    doc = data["entry"][0]["resource"]
    assert doc["resourceType"] == "DocumentReference"
    assert doc["status"] == "current"


async def test_service_request_search(client: AsyncClient, attending_token: str):
    """ServiceRequest search returns imaging orders."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/ServiceRequest?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    sr = data["entry"][0]["resource"]
    assert sr["resourceType"] == "ServiceRequest"
    assert sr["intent"] == "order"


async def test_medication_request_search(client: AsyncClient, attending_token: str):
    """MedicationRequest search returns medication sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/MedicationRequest?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0


async def test_allergy_intolerance_search(client: AsyncClient, attending_token: str):
    """AllergyIntolerance search returns allergy sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/fhir/AllergyIntolerance?patient={KNOWN_PATIENT_ID}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0


async def test_pagination(client: AsyncClient, attending_token: str):
    """Pagination works correctly via _count and _offset."""
    headers = {"Authorization": f"Bearer {attending_token}"}

    # Page 1
    resp1 = await client.get("/fhir/Patient?_count=3&_offset=0", headers=headers)
    assert resp1.status_code == 200
    page1 = resp1.json()
    assert len(page1["entry"]) == 3

    # Page 2 should have different patients
    resp2 = await client.get("/fhir/Patient?_count=3&_offset=3", headers=headers)
    assert resp2.status_code == 200
    page2 = resp2.json()
    assert len(page2["entry"]) == 3

    # IDs should be different
    ids1 = {e["resource"]["id"] for e in page1["entry"]}
    ids2 = {e["resource"]["id"] for e in page2["entry"]}
    assert ids1.isdisjoint(ids2), "Pagination returned duplicate patients"
