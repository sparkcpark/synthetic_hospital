"""Test fixtures for Epic EHR simulation tests.

Uses the real PostgreSQL database (which has migrated data) with httpx AsyncClient.
"""

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from epic_sim.app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="session")
async def client():
    """Async HTTP client against the FastAPI app (session-scoped)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _get_token(client: AsyncClient, role: str, department: str) -> str:
    """Register a user (if needed) and return a JWT token."""
    username = f"pytest_{role}_{uuid.uuid4().hex}"
    password = "test123"

    # Register
    resp = await client.post("/auth/register", json={
        "username": username,
        "password": password,
        "role": role,
        "display_name": f"Test {role.title()}",
        "department": department,
    })
    assert resp.status_code in (201, 409), f"Register failed: {resp.text}"

    # Login
    resp = await client.post("/auth/token", json={
        "username": username,
        "password": password,
    })
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


@pytest_asyncio.fixture(scope="session")
async def attending_token(client: AsyncClient):
    return await _get_token(client, "attending", "Internal Medicine")


@pytest_asyncio.fixture(scope="session")
async def nurse_token(client: AsyncClient):
    return await _get_token(client, "nurse", "ED")


@pytest_asyncio.fixture(scope="session")
async def radiologist_token(client: AsyncClient):
    return await _get_token(client, "radiologist", "Radiology")


@pytest_asyncio.fixture(scope="session")
async def lab_tech_token(client: AsyncClient):
    return await _get_token(client, "lab_tech", "Lab")


@pytest_asyncio.fixture(scope="session")
async def pharmacist_token(client: AsyncClient):
    return await _get_token(client, "pharmacist", "Pharmacy")
