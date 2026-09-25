"""Tests for the scoring (reward) endpoint.

These run against the loaded benchmark database (they skip when no ground truth
is present) and build "perfect" submissions from the labels to check that the
reward reaches 1.0, plus the authentication, listing, mismatch, batch and
empty-submission paths.
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


def _gt_row(task: str, split: str = "public", extra_sql: str = "", params: tuple = ()):
    with _pg() as conn:
        row = conn.execute(
            f"""
            SELECT gt_id, patient_id, ground_truth FROM benchmark_ground_truth
            WHERE task::text = %s AND split::text = %s AND is_diagnostic {extra_sql}
            ORDER BY gt_id LIMIT 1
            """,
            (task, split, *params),
        ).fetchone()
    if row is None:
        pytest.skip(f"no {task} ground truth loaded")
    gt = row[2] if isinstance(row[2], dict) else json.loads(row[2])
    return row[0], row[1], gt


@pytest.fixture(scope="module", autouse=True)
def _require_ground_truth():
    try:
        with _pg() as conn:
            n = conn.execute("SELECT count(*) FROM benchmark_ground_truth").fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable: {exc}")
    if not n:
        pytest.skip("benchmark ground truth not loaded")


async def test_requires_token(client: AsyncClient):
    resp = await client.get("/score/tasks")
    assert resp.status_code == 401
    resp = await client.get("/score/tasks", headers={"X-Scorer-Token": "wrong"})
    assert resp.status_code == 401


async def test_tasks_listing(client: AsyncClient):
    resp = await client.get("/score/tasks", headers=HEADERS)
    assert resp.status_code == 200
    tasks = {t["task"]: t["reward_metric"] for t in resp.json()}
    assert tasks["patient_diagnosis"] == "weighted_problem_list_f1_neutral"
    assert tasks["evidence_retrieval"] == "precision_5"


async def test_instances_have_no_labels(client: AsyncClient):
    resp = await client.get(
        "/score/instances", headers=HEADERS,
        params={"task": "patient_diagnosis", "split": "public", "limit": 5},
    )
    assert resp.status_code == 200
    items = resp.json()
    assert 0 < len(items) <= 5
    for it in items:
        assert it["task"] == "patient_diagnosis" and it["split"] == "public"
        assert "ground_truth" not in it and "active_diagnoses" not in it


async def test_patient_diagnosis_perfect_submission(client: AsyncClient):
    gt_id, _pid, gt = _gt_row("patient_diagnosis")
    pred = {
        "active_diagnoses": [{"icd10": d["icd10"], "acuity": d.get("acuity", "acute")}
                             for d in gt["active_diagnoses"] if not d.get("excluded_nondiagnostic")],
        "chronic_conditions": [{"icd10": d["icd10"], "acuity": "chronic"}
                               for d in gt["chronic_conditions"] if not d.get("excluded_nondiagnostic")],
    }
    resp = await client.post("/score", headers=HEADERS,
                             json={"gt_id": gt_id, "task": "patient_diagnosis", "prediction": pred})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reward"] == pytest.approx(1.0)
    assert body["reward_metric"] == "weighted_problem_list_f1_neutral"
    assert body["metrics"]["weighted_problem_list_f1"] == pytest.approx(1.0)


async def test_patient_diagnosis_chart_neutral_not_penalized(client: AsyncClient):
    """A chart-documented comorbidity absent from the reference costs nothing."""
    from eval.chart_neutral import neutral_categories_for

    with _pg() as conn:
        rows = conn.execute(
            """
            SELECT gt_id, patient_id, ground_truth FROM benchmark_ground_truth
            WHERE task::text = 'patient_diagnosis' AND split::text = 'public' AND is_diagnostic
            ORDER BY gt_id LIMIT 50
            """
        ).fetchall()
        picked = None
        for gt_id, pid, gt in rows:
            gt = gt if isinstance(gt, dict) else json.loads(gt)
            gt_cats = {d["icd10"][:3] for d in gt["active_diagnoses"] + gt["chronic_conditions"]}
            extra = sorted(neutral_categories_for(conn, pid) - gt_cats)
            if extra:
                picked = (gt_id, gt, extra[0])
                break
    if picked is None:
        pytest.skip("no public patient with a chart-neutral category outside its reference")
    gt_id, gt, neutral_cat = picked
    pred = {
        "active_diagnoses": [{"icd10": d["icd10"], "acuity": d.get("acuity", "acute")}
                             for d in gt["active_diagnoses"]],
        "chronic_conditions": [{"icd10": d["icd10"], "acuity": "chronic"} for d in gt["chronic_conditions"]]
                              + [{"icd10": f"{neutral_cat}.9", "acuity": "chronic"}],
    }
    resp = await client.post("/score", headers=HEADERS, json={"gt_id": gt_id, "prediction": pred})
    assert resp.status_code == 200, resp.text
    m = resp.json()["metrics"]
    assert m["problem_list_precision"] < 1.0            # strict rule penalizes it
    assert m["problem_list_precision_neutral"] == pytest.approx(1.0)
    assert m["n_neutral_predictions"] == 1
    assert resp.json()["reward"] == pytest.approx(1.0)


async def test_empty_submission_scores_zero(client: AsyncClient):
    gt_id, _pid, _gt = _gt_row("patient_diagnosis")
    resp = await client.post("/score", headers=HEADERS, json={"gt_id": gt_id, "prediction": {}})
    assert resp.status_code == 200
    assert resp.json()["reward"] == 0.0


async def test_task_mismatch_rejected(client: AsyncClient):
    gt_id, _pid, _gt = _gt_row("patient_diagnosis")
    resp = await client.post("/score", headers=HEADERS,
                             json={"gt_id": gt_id, "task": "evidence_retrieval", "prediction": {}})
    assert resp.status_code == 422
    resp = await client.post("/score", headers=HEADERS, json={"gt_id": -1, "prediction": {}})
    assert resp.status_code == 404


async def test_retrieval_perfect_ranking(client: AsyncClient):
    with _pg() as conn:
        row = conn.execute(
            """
            SELECT g.gt_id FROM benchmark_ground_truth g
            WHERE g.task::text = 'evidence_retrieval' AND g.split::text = 'public'
              AND (SELECT count(*) FROM relevance_judgments r WHERE r.gt_id = g.gt_id AND r.relevance_grade >= 2) >= 10
            ORDER BY g.gt_id LIMIT 1
            """
        ).fetchone()
        if row is None:
            pytest.skip("no retrieval instance with 10 relevant passages")
        gt_id = row[0]
        top = conn.execute(
            "SELECT passage_id FROM relevance_judgments WHERE gt_id = %s ORDER BY relevance_grade DESC, passage_id LIMIT 10",
            (gt_id,),
        ).fetchall()
    pred = {"rankings": [{"passage_id": p[0], "grade": 3} for p in top]}
    resp = await client.post("/score", headers=HEADERS, json={"gt_id": gt_id, "prediction": pred})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reward_metric"] == "precision_5"
    assert body["reward"] == pytest.approx(1.0)
    assert body["metrics"]["ndcg_10"] == pytest.approx(1.0)


async def test_summarization_must_include_recall(client: AsyncClient):
    gt_id, _pid, gt = _gt_row("context_summarization", extra_sql="AND ground_truth->>'variant' IS NULL")
    names = [f.get("display_name") or f.get("name") for f in gt["must_include_findings"]]
    resp = await client.post("/score", headers=HEADERS,
                             json={"gt_id": gt_id, "prediction": {"summary": ". ".join(n for n in names if n) + "."}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reward_metric"] == "clinical_f1"
    assert body["reward"] == pytest.approx(1.0)


async def test_imaging_identical_question(client: AsyncClient):
    gt_id, _pid, gt = _gt_row("imaging_indication")
    pred = {"clinical_question": gt["inferred_clinical_question"], "pre_read_summary": "", "must_include_findings": []}
    resp = await client.post("/score", headers=HEADERS, json={"gt_id": gt_id, "prediction": pred})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reward_metric"] == "clinical_question_concept_f1"
    assert body["reward"] == pytest.approx(1.0)
    assert body["metrics"]["clinical_question_f1"] == pytest.approx(1.0)


async def test_batch(client: AsyncClient):
    gt_id, _pid, _gt = _gt_row("patient_diagnosis")
    resp = await client.post("/score/batch", headers=HEADERS,
                             json={"items": [{"gt_id": gt_id, "prediction": {}}, {"gt_id": -5, "prediction": {}}]})
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert items[0]["ok"] is True and items[0]["result"]["reward"] == 0.0
    assert items[1]["ok"] is False and "no benchmark instance" in items[1]["error"]
