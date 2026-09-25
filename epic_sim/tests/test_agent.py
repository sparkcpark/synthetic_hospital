"""Tests for Agent tool-use protocol and session management."""

import pytest
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio(loop_scope="session")

KNOWN_PATIENT_ID = 1672


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

async def test_get_tools_returns_13(client: AsyncClient):
    """GET /agent/tools returns 13 tool definitions (12 original + submit_rankings)."""
    resp = await client.get("/agent/tools")
    assert resp.status_code == 200
    tools = resp.json()
    assert len(tools) == 13


async def test_tool_schema_valid(client: AsyncClient):
    """Each tool has name, description, and parameters."""
    resp = await client.get("/agent/tools")
    for tool in resp.json():
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------

async def test_open_chart_via_tool(client: AsyncClient, attending_token: str):
    """open_chart tool returns patient summary."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "tool_name": "open_chart",
            "arguments": {"patient_id": KNOWN_PATIENT_ID},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool_name"] == "open_chart"
    assert data["result"]["patient_id"] == KNOWN_PATIENT_ID


async def test_view_encounters_via_tool(client: AsyncClient, attending_token: str):
    """view_encounters tool returns encounter list."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "tool_name": "view_encounters",
            "arguments": {"patient_id": KNOWN_PATIENT_ID},
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert isinstance(result, list)
    assert len(result) > 0


async def test_view_results_labs_via_tool(client: AsyncClient, attending_token: str):
    """view_results tool returns lab results."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "tool_name": "view_results",
            "arguments": {"patient_id": KNOWN_PATIENT_ID, "result_type": "labs"},
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert isinstance(result, list)


async def test_submit_diagnosis_writes_to_db(client: AsyncClient, attending_token: str):
    """submit_diagnosis records submission in agent_submissions."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "tool_name": "submit_diagnosis",
            "arguments": {
                "patient_id": KNOWN_PATIENT_ID,
                "diagnosis_name": "Acute myocardial infarction",
                "icd10_code": "I21.01",
            },
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["status"] == "recorded"


async def test_unknown_tool_400(client: AsyncClient, attending_token: str):
    """Unknown tool name returns 400."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "tool_name": "nonexistent_tool",
            "arguments": {},
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Sessions (require Redis)
# ---------------------------------------------------------------------------

async def test_create_session(client: AsyncClient, attending_token: str):
    """Create a session with patient assignment."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/epic/sessions",
        headers=headers,
        json={"patient_id": KNOWN_PATIENT_ID, "role": "attending"},
    )
    # 201 if Redis available, 503 if not
    if resp.status_code == 503:
        pytest.skip("Redis not available")
    assert resp.status_code == 201
    data = resp.json()
    assert data["patient_id"] == KNOWN_PATIENT_ID
    assert data["status"] == "active"


async def test_get_session_state(client: AsyncClient, attending_token: str):
    """Get session returns current state."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    # Create
    resp = await client.post(
        "/epic/sessions",
        headers=headers,
        json={"patient_id": KNOWN_PATIENT_ID},
    )
    if resp.status_code == 503:
        pytest.skip("Redis not available")
    session_id = resp.json()["session_id"]

    # Get
    resp = await client.get(f"/epic/sessions/{session_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["session_id"] == session_id


async def test_end_session(client: AsyncClient, attending_token: str):
    """End session marks it as completed."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/epic/sessions",
        headers=headers,
        json={"patient_id": KNOWN_PATIENT_ID},
    )
    if resp.status_code == 503:
        pytest.skip("Redis not available")
    session_id = resp.json()["session_id"]

    # Delete
    resp = await client.delete(f"/epic/sessions/{session_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


async def test_rate_limiting_429(client: AsyncClient, attending_token: str):
    """Exceeding rate limit returns 429."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/epic/sessions",
        headers=headers,
        json={"patient_id": KNOWN_PATIENT_ID, "max_api_calls": 2},
    )
    if resp.status_code == 503:
        pytest.skip("Redis not available")
    session_id = resp.json()["session_id"]

    # Make 2 calls (at limit)
    for _ in range(2):
        resp = await client.post(
            "/agent/tools",
            headers=headers,
            json={
                "session_id": session_id,
                "tool_name": "view_problem_list",
                "arguments": {"patient_id": KNOWN_PATIENT_ID},
            },
        )
        assert resp.status_code == 200

    # 3rd call should be 429
    resp = await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "session_id": session_id,
            "tool_name": "view_problem_list",
            "arguments": {"patient_id": KNOWN_PATIENT_ID},
        },
    )
    assert resp.status_code == 429


async def test_session_tracks_tool_calls(client: AsyncClient, attending_token: str):
    """Session records tool call trace."""
    headers = {"Authorization": f"Bearer {attending_token}"}
    resp = await client.post(
        "/epic/sessions",
        headers=headers,
        json={"patient_id": KNOWN_PATIENT_ID},
    )
    if resp.status_code == 503:
        pytest.skip("Redis not available")
    session_id = resp.json()["session_id"]

    # Make a tool call
    await client.post(
        "/agent/tools",
        headers=headers,
        json={
            "session_id": session_id,
            "tool_name": "open_chart",
            "arguments": {"patient_id": KNOWN_PATIENT_ID},
        },
    )

    # Check session state shows call count
    resp = await client.get(f"/epic/sessions/{session_id}", headers=headers)
    assert resp.json()["api_call_count"] == 1
