"""Score a single submission against one benchmark instance.

This is the reward function of the RL environment and the backend of the
simulator's `/score` endpoint. It reuses the exact scorer used for the paper
(`eval.scoring.compute_all_metrics`) on a one-item batch, attaching the same
per-instance context the batch scorer uses:

  patient_diagnosis      chart-neutral categories (eval/chart_neutral.py)
  evidence_retrieval     graded relevance judgments over chart sections
  context_summarization  nothing extra (hallucination rate, which needs the chart
                         text, is a secondary metric and is not computed here)
  imaging_indication     the concept extractor built from the graph (cached per process)

Reward = the task's primary metric, always in [0, 1]:

  patient_diagnosis      weighted_problem_list_f1_neutral   (severity-weighted, chart-neutral)
  context_summarization  clinical_f1 (whole-patient / current-visit),
                         conditioned_f1 (specialty, involved) or
                         abstention_accuracy (specialty, absent)
  evidence_retrieval     precision_5   (ndcg_10 is also returned)
  imaging_indication     clinical_question_concept_f1 (ontology-grounded concept F1,
                         eval/imaging_concepts.py; token-level clinical_question_f1 also returned)

Malformed or empty submissions score 0, never an error: the scorer treats an
unparseable prediction exactly as the evaluation harness does.
"""

from __future__ import annotations

import json
import math
from typing import Any

from eval.chart_neutral import neutral_categories_for
from eval.config import EVAL_TASKS
from eval.scoring import compute_all_metrics

PRIMARY_METRIC: dict[str, str] = {
    "patient_diagnosis": "weighted_problem_list_f1_neutral",
    "context_summarization": "clinical_f1",   # specialty variant handled in _primary_for
    "evidence_retrieval": "precision_5",
    "imaging_indication": "clinical_question_concept_f1",
}


class InstanceNotFound(LookupError):
    pass


class TaskMismatch(ValueError):
    pass


def _jsonable(value: Any) -> Any:
    """Make metric values JSON-safe (NaN -> None, numpy scalars -> float)."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_instance(conn, gt_id: int) -> dict:
    """Load one ground-truth row. Raises InstanceNotFound."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT gt_id, task::text, granularity::text, split::text, patient_id,
                   encounter_id, question_id, difficulty::text, is_diagnostic, ground_truth
            FROM benchmark_ground_truth
            WHERE gt_id = %s
            """,
            (gt_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise InstanceNotFound(f"no benchmark instance with gt_id={gt_id}")
    gt = row[9] if isinstance(row[9], dict) else json.loads(row[9])
    return {
        "gt_id": row[0],
        "task": row[1],
        "granularity": row[2],
        "split": row[3],
        "patient_id": row[4],
        "encounter_id": row[5],
        "question_id": row[6],
        "difficulty": row[7],
        "is_diagnostic": bool(row[8]),
        "variant": gt.get("variant", "unconditioned") if row[1] == "context_summarization" else None,
        "ground_truth": gt,
    }


def _attach_context(conn, inst: dict) -> dict:
    """Return a copy of the GT dict with the scorer's per-instance context attached."""
    gt = dict(inst["ground_truth"])
    task = inst["task"]
    if task == "patient_diagnosis":
        gt["_neutral_categories"] = sorted(neutral_categories_for(conn, inst["patient_id"]))
    elif task == "evidence_retrieval":
        with conn.cursor() as cur:
            cur.execute(
                "SELECT passage_id, relevance_grade FROM relevance_judgments WHERE gt_id = %s",
                (inst["gt_id"],),
            )
            gt["_judgments"] = {pid: grade for pid, grade in cur.fetchall()}
    return gt


def _primary_for(task: str, gt: dict, metrics: dict) -> str:
    if task == "context_summarization" and gt.get("variant") == "specialty_conditioned":
        return "conditioned_f1" if "conditioned_f1" in metrics else "abstention_accuracy"
    return PRIMARY_METRIC[task]


def score_submission(conn, gt_id: int, prediction: dict, task: str | None = None,
                     reward_metric: str | None = None) -> dict:
    """Score `prediction` against instance `gt_id`.

    Returns a dict with the instance identifiers, the reward, the name of the
    metric the reward was taken from, and every metric the scorer computed.
    Never returns the ground truth itself.
    """
    inst = load_instance(conn, gt_id)
    if task is not None and task != inst["task"]:
        raise TaskMismatch(f"gt_id={gt_id} belongs to task '{inst['task']}', not '{task}'")
    if inst["task"] not in EVAL_TASKS:
        raise TaskMismatch(f"task '{inst['task']}' is not scorable")

    gt = _attach_context(conn, inst)
    if not isinstance(prediction, dict):
        prediction = {}
    kwargs: dict[str, Any] = {}
    if inst["task"] == "imaging_indication":
        from eval.imaging_concepts import ConceptExtractor
        kwargs["concept_extractor"] = ConceptExtractor.from_db(conn)
    metrics = compute_all_metrics(inst["task"], [prediction], [gt], **kwargs)
    metrics = _jsonable(metrics)

    metric_name = reward_metric or _primary_for(inst["task"], gt, metrics)
    if metric_name not in metrics:
        raise ValueError(f"metric '{metric_name}' not available; choose from {sorted(metrics)}")
    reward = metrics[metric_name]
    if not isinstance(reward, (int, float)) or reward is None:
        reward = 0.0
    reward = float(min(1.0, max(0.0, reward)))

    return {
        "gt_id": inst["gt_id"],
        "task": inst["task"],
        "split": inst["split"],
        "patient_id": inst["patient_id"],
        "encounter_id": inst["encounter_id"],
        "variant": inst["variant"],
        "reward": reward,
        "reward_metric": metric_name,
        "metrics": metrics,
    }


def list_instances(conn, task: str | None = None, split: str | None = None,
                   variant: str | None = None, limit: int = 500, offset: int = 0) -> list[dict]:
    """Enumerate benchmark instances without their labels (for env reset / sampling)."""
    clauses = ["is_diagnostic"]
    params: list[Any] = []
    if task:
        clauses.append("task::text = %s"); params.append(task)
    if split:
        clauses.append("split::text = %s"); params.append(split)
    if variant:
        clauses.append("COALESCE(ground_truth->>'variant', 'unconditioned') = %s"); params.append(variant)
    params += [limit, offset]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT gt_id, task::text, granularity::text, split::text, patient_id, encounter_id,
                   difficulty::text, COALESCE(ground_truth->>'variant', 'unconditioned'),
                   ground_truth->>'clinical_question', ground_truth->>'specialty'
            FROM benchmark_ground_truth
            WHERE {' AND '.join(clauses)}
            ORDER BY gt_id
            LIMIT %s OFFSET %s
            """,
            params,
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        item = {
            "gt_id": r[0], "task": r[1], "granularity": r[2], "split": r[3],
            "patient_id": r[4], "encounter_id": r[5], "difficulty": r[6],
        }
        if r[1] == "context_summarization":
            item["variant"] = r[7]
            item["clinical_question"] = r[8]
            if r[9]:
                item["specialty"] = r[9]
        out.append(item)
    return out
