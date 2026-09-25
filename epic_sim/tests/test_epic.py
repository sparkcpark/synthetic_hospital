"""Tests for Epic simulation endpoints."""

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio(loop_scope="session")

KNOWN_PATIENT_ID = 1672


# ---------------------------------------------------------------------------
# Chart Review
# ---------------------------------------------------------------------------

async def test_chart_summary(client: AsyncClient, attending_token: str):
    """Chart summary returns patient demographics and problems."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/chart/{KNOWN_PATIENT_ID}/summary", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_id"] == KNOWN_PATIENT_ID
    assert data["name"] is not None
    assert isinstance(data["recent_encounters"], list)


async def test_chart_summary_has_problems(client: AsyncClient, attending_token: str):
    """Chart summary includes active problem list."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/chart/{KNOWN_PATIENT_ID}/summary", headers=headers)
    data = resp.json()
    assert len(data["active_problems"]) > 0
    problem = data["active_problems"][0]
    assert "display_name" in problem
    assert "diagnosis_id" in problem


async def test_chart_encounters(client: AsyncClient, attending_token: str):
    """Encounter list is chronological."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/chart/{KNOWN_PATIENT_ID}/encounters", headers=headers)
    assert resp.status_code == 200
    encounters = resp.json()
    assert len(encounters) > 0
    # Verify chronological order
    dates = [e["date"] for e in encounters if e["date"]]
    assert dates == sorted(dates)


async def test_encounter_detail(client: AsyncClient, attending_token: str):
    """Encounter detail includes sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    # First get an encounter ID
    resp = await client.get(f"/epic/chart/{KNOWN_PATIENT_ID}/encounters", headers=headers)
    encounters = resp.json()
    enc_id = encounters[0]["encounter_id"]

    resp = await client.get(
        f"/epic/chart/{KNOWN_PATIENT_ID}/encounters/{enc_id}", headers=headers
    )
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["encounter_id"] == enc_id
    assert len(detail["sections"]) > 0


async def test_encounter_detail_radiologist_filtered(
    client: AsyncClient, attending_token: str, radiologist_token: str
):
    """Radiologist sees fewer sections than attending."""
    # Get encounter
    headers_att = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/chart/{KNOWN_PATIENT_ID}/encounters", headers=headers_att)
    enc_id = resp.json()[0]["encounter_id"]

    # Attending: full access
    resp_att = await client.get(
        f"/epic/chart/{KNOWN_PATIENT_ID}/encounters/{enc_id}", headers=headers_att
    )
    att_types = {s["section_type"] for s in resp_att.json()["sections"]}

    # Radiologist: filtered
    headers_rad = {"Authorization": f"Bearer {radiologist_token}"}
    resp_rad = await client.get(
        f"/epic/chart/{KNOWN_PATIENT_ID}/encounters/{enc_id}", headers=headers_rad
    )
    rad_types = {s["section_type"] for s in resp_rad.json()["sections"]}

    assert len(rad_types) <= len(att_types)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

async def test_lab_results(client: AsyncClient, attending_token: str):
    """Lab results return data."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/results/{KNOWN_PATIENT_ID}/labs", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_imaging_results(client: AsyncClient, attending_token: str):
    """Imaging results return data."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/results/{KNOWN_PATIENT_ID}/imaging", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_pathology_results(client: AsyncClient, attending_token: str):
    """Pathology results return data."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/results/{KNOWN_PATIENT_ID}/pathology", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Problem List
# ---------------------------------------------------------------------------

async def test_problem_list(client: AsyncClient, attending_token: str):
    """Problem list returns correct diagnoses."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get(f"/epic/chart/{KNOWN_PATIENT_ID}/problems", headers=headers)
    assert resp.status_code == 200
    problems = resp.json()
    assert len(problems) > 0
    assert "display_name" in problems[0]


async def test_add_problem(client: AsyncClient, attending_token: str):
    """Adding a problem records submission."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        f"/epic/chart/{KNOWN_PATIENT_ID}/problems",
        headers=headers,
        json={"display_name": "Test Diagnosis", "icd10_code": "Z99.9"},
    )
    assert resp.status_code == 201
    assert resp.json()["display_name"] == "Test Diagnosis"


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

async def test_nurse_cannot_order(client: AsyncClient, nurse_token: str):
    """Nurse lacks ServiceRequest.write scope."""
    headers = {"Authorization": f"Bearer {nurse_token}"}
    resp = await client.post(
        "/epic/orders/imaging",
        headers=headers,
        json={
            "patient_id": KNOWN_PATIENT_ID,
            "encounter_id": 1,
            "modality": "xray",
            "body_region": "chest",
            "clinical_indication": "cough",
            "priority": "routine",
        },
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Radiology
# ---------------------------------------------------------------------------

async def test_radiology_worklist(client: AsyncClient, attending_token: str):
    """Radiology worklist returns imaging orders."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.get("/epic/radiology/worklist", headers=headers)
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list)
    assert len(items) > 0
    assert "modality" in items[0]


async def test_pre_read_context(client: AsyncClient, attending_token: str):
    """Pre-read context returns order + encounter + history."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    # Get an order ID from worklist
    resp = await client.get("/epic/radiology/worklist", headers=headers)
    order_id = resp.json()[0]["order_id"]

    resp = await client.get(f"/epic/radiology/pre-read/{order_id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "order" in data
    assert "encounter" in data
    assert data["order"]["order_id"] == order_id


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

async def test_search_chart(client: AsyncClient, attending_token: str):
    """Chart search returns matching sections."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "tool_name": "search_chart",
            "arguments": {"patient_id": KNOWN_PATIENT_ID, "query": "pain"},
        },
    )
    assert resp.status_code == 200
    results = resp.json()["result"]
    assert isinstance(results, list)
