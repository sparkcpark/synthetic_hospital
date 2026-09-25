"""Tests for the reset-and-step environment (/env).

Need Redis (episode state) and the loaded benchmark; they skip otherwise.
"""

from __future__ import annotations

import json

import psycopg
import pytest
from httpx import AsyncClient

from epic_sim.app.config import settings

pytestmark = pytest.mark.asyncio(loop_scope="session")

HEADERS = {"X-Scorer-Token": settings.scorer_token}


def _pg():
    return psycopg.connect(settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://"))


def _first_gt(task: str, split: str = "public"):
    with _pg() as conn:
        row = conn.execute(
            """
            SELECT gt_id, patient_id, encounter_id, ground_truth FROM benchmark_ground_truth
            WHERE task::text = %s AND split::text = %s AND is_diagnostic
              AND COALESCE(ground_truth->>'variant', 'unconditioned') = 'unconditioned'
            ORDER BY gt_id LIMIT 1
            """,
            (task, split),
        ).fetchone()
    if row is None:
        pytest.skip(f"no {task} ground truth loaded")
    gt = row[3] if isinstance(row[3], dict) else json.loads(row[3])
    return row[0], row[1], row[2], gt


@pytest.fixture(scope="module", autouse=True)
async def _require_env(client: AsyncClient):
    # The ASGI test transport does not run the app's lifespan, so connect Redis here
    # the same way the lifespan does; skip when it is unreachable.
    from epic_sim.app.main import app
    if getattr(app.state, "redis", None) is None:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            await r.ping()
            app.state.redis = r
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"environment needs Redis (episode state): {exc}")
    try:
        with _pg() as conn:
            n = conn.execute("SELECT count(*) FROM benchmark_ground_truth").fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable: {exc}")
    if not n:
        pytest.skip("benchmark ground truth not loaded")


async def test_reset_requires_token(client: AsyncClient):
    resp = await client.post("/env/reset", json={"task": "patient_diagnosis"})
    assert resp.status_code == 401


async def test_reset_returns_prompt_tools_and_no_labels(client: AsyncClient):
    gt_id, pid, _eid, _gt = _first_gt("patient_diagnosis")
    resp = await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id, "budget": 5})
    assert resp.status_code == 200, resp.text
    obs = resp.json()
    assert obs["task"] == "patient_diagnosis" and obs["patient_id"] == pid
    assert obs["budget"] == 5 and obs["remaining"] == 5
    assert obs["submit_tool"] == "submit_diagnosis"
    assert "active_diagnoses" in obs["instructions"]        # the paper's system prompt
    assert f"patient_id = {pid}" in obs["intro"]
    names = [t["function"]["name"] for t in obs["tools"]]
    assert "open_chart" in names and "submit_diagnosis" in names
    assert not any(n in names for n in ("submit_summary", "submit_rankings", "submit_pre_read"))
    assert "ground_truth" not in obs
    # the submit tool carries the scoring schema, not the server's one-diagnosis form
    submit = next(t for t in obs["tools"] if t["function"]["name"] == "submit_diagnosis")
    assert "chronic_conditions" in submit["function"]["parameters"]["properties"]


async def test_reset_by_sampling_is_seeded(client: AsyncClient):
    a = await client.post("/env/reset", headers=HEADERS, json={"task": "patient_diagnosis", "split": "public", "seed": 7})
    b = await client.post("/env/reset", headers=HEADERS, json={"task": "patient_diagnosis", "split": "public", "seed": 7})
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["gt_id"] == b.json()["gt_id"] and a.json()["episode_id"] != b.json()["episode_id"]
    assert a.json()["split"] == "public"


async def test_step_hides_outcome_sections_and_counts_budget(client: AsyncClient):
    gt_id, pid, _eid, _gt = _first_gt("patient_diagnosis")
    ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id, "budget": 3})).json()
    eid = ep["episode_id"]
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": eid, "name": "view_encounters", "arguments": {"patient_id": pid}})
    assert r.status_code == 200, r.text
    step = r.json()
    assert step["done"] is False and step["reward"] == 0.0 and step["remaining"] == 2
    encounters = step["observation"]
    assert isinstance(encounters, list) and encounters
    enc_id = encounters[0]["encounter_id"]
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": eid, "name": "view_encounter_detail", "arguments": {"encounter_id": enc_id}})
    detail = r.json()["observation"]
    types = {s["section_type"] for s in detail.get("sections", [])}
    assert types and not (types & {"assessment", "plan"})
    assert len(r.json()["observation_text"]) <= 8000 + len("\n[OUTPUT TRUNCATED]")
    # unknown tool -> error observation, still costs a step
    r = await client.post("/env/step", headers=HEADERS, json={"episode_id": eid, "name": "nonexistent_tool"})
    assert r.status_code == 200 and "error" in r.json()["observation"] and r.json()["remaining"] == 0
    # budget spent: retrieval tools refused, submit still accepted
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": eid, "name": "open_chart", "arguments": {"patient_id": pid}})
    assert "budget" in r.json()["observation"]["error"].lower() and r.json()["done"] is False
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": eid, "name": "submit_diagnosis", "arguments": {}})
    assert r.status_code == 200 and r.json()["done"] is True and r.json()["reward"] == 0.0
    assert r.json()["info"]["forced"] is True and r.json()["info"]["malformed"] is True
    # finished episodes reject further steps
    r = await client.post("/env/step", headers=HEADERS, json={"episode_id": eid, "name": "open_chart"})
    assert r.status_code == 409


async def test_problem_list_is_chart_history_not_labels(client: AsyncClient):
    gt_id, pid, _eid, gt = _first_gt("patient_diagnosis")
    ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
    for tool in ("view_problem_list", "open_chart"):
        r = await client.post("/env/step", headers=HEADERS,
                              json={"episode_id": ep["episode_id"], "name": tool, "arguments": {"patient_id": pid}})
        obs = r.json()["observation"]
        problems = obs if tool == "view_problem_list" else obs["active_problems"]
        assert all(p.get("source") == "chart_history" and "icd10_code" not in p and "diagnosis_id" not in p
                   for p in problems)
        label_names = {d["display_name"].lower() for d in gt["active_diagnoses"]}
        assert not ({p["display_name"].lower() for p in problems} & label_names)


async def test_perfect_submission_gets_full_reward(client: AsyncClient):
    gt_id, _pid, _eid, gt = _first_gt("patient_diagnosis")
    ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
    pred = {
        "active_diagnoses": [{"icd10": d["icd10"], "name": d.get("display_name", ""), "acuity": d.get("acuity", "acute")}
                             for d in gt["active_diagnoses"]],
        "chronic_conditions": [{"icd10": d["icd10"], "name": d.get("display_name", ""), "acuity": "chronic"}
                               for d in gt["chronic_conditions"]],
    }
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": ep["episode_id"], "name": "submit_diagnosis", "arguments": pred})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["done"] and body["reward"] == pytest.approx(1.0)
    assert body["info"]["reward_metric"] == "weighted_problem_list_f1_neutral"
    state = (await client.get(f"/env/state/{ep['episode_id']}", headers=HEADERS)).json()
    assert state["submitted"] and state["reward"] == pytest.approx(1.0) and state["steps"] == 1


async def test_wrong_submit_tool_is_an_observation(client: AsyncClient):
    gt_id, _pid, _eid, _gt = _first_gt("context_summarization")
    ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
    assert ep["submit_tool"] == "submit_summary" and ep["task_inputs"]["clinical_question"]
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": ep["episode_id"], "name": "submit_diagnosis", "arguments": {}})
    assert r.status_code == 200 and r.json()["done"] is False and "submit_summary" in r.json()["observation"]["error"]
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": ep["episode_id"], "name": "submit_summary",
                                "arguments": {"summary": "No findings."}})
    assert r.json()["done"] is True and 0.0 <= r.json()["reward"] <= 1.0


async def test_imaging_episode_hides_future_encounters(client: AsyncClient):
    with _pg() as conn:
        row = conn.execute(
            """
            SELECT g.gt_id, g.patient_id, g.encounter_id FROM benchmark_ground_truth g
            JOIN longitudinal_encounters e ON e.encounter_id = g.encounter_id
            WHERE g.task::text = 'imaging_indication' AND g.split::text = 'public'
              AND e.encounter_order < (SELECT max(encounter_order) FROM longitudinal_encounters WHERE patient_id = g.patient_id)
            ORDER BY g.gt_id LIMIT 1
            """
        ).fetchone()
        if row is None:
            pytest.skip("no public imaging instance with later encounters")
        gt_id, pid, eid = row
        future = conn.execute(
            """
            SELECT encounter_id FROM longitudinal_encounters WHERE patient_id = %s
              AND encounter_order > (SELECT encounter_order FROM longitudinal_encounters WHERE encounter_id = %s)
            """,
            (pid, eid),
        ).fetchall()
    future_ids = {r[0] for r in future}
    ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
    assert ep["submit_tool"] == "submit_pre_read" and ep["task_inputs"]["modality"]
    assert f"encounter_id = {eid}" in ep["intro"]
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": ep["episode_id"], "name": "view_encounters", "arguments": {"patient_id": pid}})
    seen = {e["encounter_id"] for e in r.json()["observation"]}
    assert eid in seen and not (seen & future_ids)
    r = await client.post("/env/step", headers=HEADERS,
                          json={"episode_id": ep["episode_id"], "name": "view_encounter_detail",
                                "arguments": {"encounter_id": next(iter(future_ids))}})
    assert "error" in r.json()["observation"]


async def test_agent_token_can_act_but_not_see_reward(client: AsyncClient):
    gt_id, pid, _eid, _gt = _first_gt("patient_diagnosis")
    ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
    agent = {"X-Episode-Token": ep["agent_token"]}
    r = await client.post("/env/step", headers=agent,
                          json={"episode_id": ep["episode_id"], "name": "open_chart", "arguments": {"patient_id": pid}})
    assert r.status_code == 200 and r.json()["reward"] == 0.0
    # wrong token, and a token for another episode, are rejected
    other = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
    for bad in ({"X-Episode-Token": "nope"}, {"X-Episode-Token": other["agent_token"]}):
        r = await client.post("/env/step", headers=bad, json={"episode_id": ep["episode_id"], "name": "open_chart"})
        assert r.status_code == 401
    # oracle needs the scorer token; the agent path submits it but learns no reward
    assert (await client.get(f"/env/oracle/{ep['episode_id']}", headers=agent)).status_code == 401
    oracle = (await client.get(f"/env/oracle/{ep['episode_id']}", headers=HEADERS)).json()
    assert oracle["submit_tool"] == "submit_diagnosis" and oracle["arguments"]["active_diagnoses"] or oracle["arguments"]["chronic_conditions"]
    r = await client.post("/env/step", headers=agent,
                          json={"episode_id": ep["episode_id"], "name": oracle["submit_tool"], "arguments": oracle["arguments"]})
    body = r.json()
    assert body["done"] is True and body["reward"] is None and "metrics" not in body["info"]
    assert (await client.get(f"/env/state/{ep['episode_id']}", headers=agent)).json()["reward"] is None
    state = (await client.get(f"/env/state/{ep['episode_id']}", headers=HEADERS)).json()
    assert state["reward"] == pytest.approx(1.0)
    # privileged close cannot be done with the agent token
    assert (await client.post("/env/close", headers=agent, json={"episode_id": other["episode_id"]})).status_code == 401


async def test_oracle_scores_full_on_every_task(client: AsyncClient):
    for task in ("context_summarization", "evidence_retrieval", "imaging_indication"):
        gt_id, _pid, _eid, _gt = _first_gt(task)
        ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
        oracle = (await client.get(f"/env/oracle/{ep['episode_id']}", headers=HEADERS)).json()
        r = await client.post("/env/step", headers=HEADERS,
                              json={"episode_id": ep["episode_id"], "name": oracle["submit_tool"], "arguments": oracle["arguments"]})
        assert r.status_code == 200, r.text
        assert r.json()["done"] and r.json()["reward"] == pytest.approx(1.0), (task, r.json()["info"])


async def test_autostart_episode_is_served_by_current(client: AsyncClient):
    from epic_sim.app.main import app
    from epic_sim.app.routers.agent import TOOL_DEFINITIONS
    from epic_sim.app.services import env_service
    assert (await client.get("/env/current")).status_code == 404
    gt_id, pid, _eid, _gt = _first_gt("patient_diagnosis")
    view = await env_service.ensure_autostart(app.state.redis, [t.model_dump() for t in TOOL_DEFINITIONS], gt_id, 7)
    again = await env_service.ensure_autostart(app.state.redis, [t.model_dump() for t in TOOL_DEFINITIONS], gt_id, 7)
    assert view["episode_id"] == again["episode_id"]            # created once
    cur = (await client.get("/env/current")).json()             # no auth needed
    assert cur["episode_id"] == view["episode_id"] and cur["budget"] == 7 and cur["patient_id"] == pid
    assert cur["agent_token"] and "ground_truth" not in cur
    # clean up the autostart marker so other tests are unaffected
    await app.state.redis.delete(env_service._key(env_service.AUTOSTART_KEY))


async def test_close_abandons_episode(client: AsyncClient):
    gt_id, _pid, _eid, _gt = _first_gt("evidence_retrieval")
    ep = (await client.post("/env/reset", headers=HEADERS, json={"gt_id": gt_id})).json()
    assert ep["submit_tool"] == "submit_rankings" and ep["task_inputs"]["diagnosis_names"]
    r = await client.post("/env/close", headers=HEADERS, json={"episode_id": ep["episode_id"]})
    assert r.status_code == 200 and r.json()["done"] is True and r.json()["reward"] == 0.0
    r = await client.get("/env/state/does-not-exist", headers=HEADERS)
    assert r.status_code == 404
