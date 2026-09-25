"""Tests for authentication and authorization."""

import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "redis" in data


async def test_register_and_login(client: AsyncClient):
    username = f"test_reg_{uuid.uuid4().hex[:8]}"
    # Register
    resp = await client.post("/auth/register", json={
        "username": username,
        "password": "securepass",
        "role": "attending",
        "display_name": "Dr. Registration Test",
        "department": "Cardiology",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == username
    assert data["role"] == "attending"
    assert data["department"] == "Cardiology"

    # Login
    resp = await client.post("/auth/token", json={
        "username": username,
        "password": "securepass",
    })
    assert resp.status_code == 200
    token_data = resp.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["role"] == "attending"
    assert "patient/Patient.read" in token_data["scope"]


async def test_invalid_login(client: AsyncClient):
    resp = await client.post("/auth/token", json={
        "username": "nonexistent_user_xyz",
        "password": "wrong",
    })
    assert resp.status_code == 401


async def test_invalid_role_registration(client: AsyncClient):
    resp = await client.post("/auth/register", json={
        "username": "bad_role_user",
        "password": "test",
        "role": "admin",  # Not a valid role
        "display_name": "Bad User",
    })
    assert resp.status_code == 400


async def test_me_endpoint(client: AsyncClient, attending_token: str):
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {attending_token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "attending"
    assert "patient/Patient.read" in data["scopes"]
    assert "patient/Condition.write" in data["scopes"]


async def test_no_auth_fhir_blocked(client: AsyncClient):
    """FHIR endpoints require auth (except metadata)."""
    resp = await client.get("/fhir/Patient")
    assert resp.status_code == 401


async def test_metadata_no_auth(client: AsyncClient):
    """FHIR metadata is public."""
    resp = await client.get("/fhir/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "CapabilityStatement"
    assert data["fhirVersion"] == "4.0.1"


async def test_nurse_scopes(client: AsyncClient, nurse_token: str):
    """Nurse should have limited scopes."""
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {nurse_token}"})
    data = resp.json()
    assert data["role"] == "nurse"
    assert "patient/Patient.read" in data["scopes"]
    assert "patient/Observation.read" in data["scopes"]
    assert "patient/MedicationRequest.read" in data["scopes"]
    # Nurse should NOT have Encounter, Condition, DiagnosticReport, etc.
    assert "patient/Encounter.read" not in data["scopes"]
    assert "patient/Condition.read" not in data["scopes"]
