"""Tests for terminology service endpoints."""

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _system_loaded(client: AsyncClient, system: str, code: str) -> bool:
    resp = await client.get("/fhir/CodeSystem/$lookup", params={"system": system, "code": code})
    return resp.status_code != 404


@pytest.fixture(autouse=True)
async def _require_icd10(client: AsyncClient):
    """Skip this module when ICD-10-CM is not loaded.

    A fresh container fetches the public-domain CMS ICD-10-CM release on first boot
    (EPIC_SIM_DOWNLOAD_ICD10=auto); without network access the table stays empty.
    """
    if not await _system_loaded(client, "http://hl7.org/fhir/sid/icd-10-cm", "I21.01"):
        pytest.skip("ICD-10-CM terminology not loaded (see README: Terminology)")


@pytest.fixture
async def _require_snomed(client: AsyncClient):
    """SNOMED CT needs a UMLS licence and is loaded automatically once its files are mounted."""
    if not await _system_loaded(client, "http://snomed.info/sct", "22298006"):
        pytest.skip("SNOMED CT terminology not loaded (mount the release under data/ontology; see README)")


async def test_icd10_lookup(client: AsyncClient):
    """CodeSystem $lookup for ICD-10-CM code."""
    resp = await client.get(
        "/fhir/CodeSystem/$lookup",
        params={"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I21.01"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "Parameters"
    display = next(p["valueString"] for p in data["parameter"] if p["name"] == "display")
    assert "myocardial infarction" in display.lower()


@pytest.mark.usefixtures("_require_snomed")
async def test_snomed_lookup(client: AsyncClient):
    """CodeSystem $lookup for SNOMED CT concept."""
    resp = await client.get(
        "/fhir/CodeSystem/$lookup",
        params={"system": "http://snomed.info/sct", "code": "22298006"},
    )
    assert resp.status_code == 200
    data = resp.json()
    display = next(p["valueString"] for p in data["parameter"] if p["name"] == "display")
    assert "myocardial infarction" in display.lower()


async def test_lookup_not_found(client: AsyncClient):
    """CodeSystem $lookup returns 404 for unknown code."""
    resp = await client.get(
        "/fhir/CodeSystem/$lookup",
        params={"system": "icd10cm", "code": "ZZZZZ"},
    )
    assert resp.status_code == 404


async def test_valueset_expand_filter(client: AsyncClient):
    """ValueSet $expand with text filter returns matching concepts."""
    resp = await client.get(
        "/fhir/ValueSet/$expand",
        params={"system": "icd10cm", "filter": "diabetes", "count": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["resourceType"] == "ValueSet"
    assert data["expansion"]["total"] > 0
    for item in data["expansion"]["contains"]:
        assert "diabetes" in item["display"].lower()


@pytest.mark.usefixtures("_require_snomed")
async def test_valueset_expand_snomed(client: AsyncClient):
    """ValueSet $expand for SNOMED CT."""
    resp = await client.get(
        "/fhir/ValueSet/$expand",
        params={"system": "snomedct", "filter": "pneumonia", "count": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["expansion"]["total"] > 0


@pytest.mark.usefixtures("_require_snomed")
async def test_validate_code_valid(client: AsyncClient):
    """ValueSet $validate-code returns true for known code."""
    resp = await client.get(
        "/fhir/ValueSet/$validate-code",
        params={"system": "http://snomed.info/sct", "code": "22298006"},
    )
    assert resp.status_code == 200
    data = resp.json()
    result = next(p["valueBoolean"] for p in data["parameter"] if p["name"] == "result")
    assert result is True


async def test_validate_code_invalid(client: AsyncClient):
    """ValueSet $validate-code returns false for unknown code."""
    resp = await client.get(
        "/fhir/ValueSet/$validate-code",
        params={"system": "icd10cm", "code": "ZZZZZ"},
    )
    assert resp.status_code == 200
    data = resp.json()
    result = next(p["valueBoolean"] for p in data["parameter"] if p["name"] == "result")
    assert result is False
