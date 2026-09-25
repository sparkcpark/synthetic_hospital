"""Episode logic for the reset-and-step RL environment (see routers/env.py).

An episode is one benchmark instance (a `benchmark_ground_truth` row) played as a
tool-use task: the policy receives the paper's agent system prompt and patient
intro, calls the same 13 EHR tools the paper's agents used through the same
dispatcher, and ends the episode by calling the task's submit tool, which is
scored with the paper's scorer (eval.score_one). Episode state lives in Redis
under its own key so that any process holding the episode id can continue it.

Server-side guarantees that the paper's harness applied client-side:
- assessment and plan sections are removed from every observation (they state
  the answer), matching the single-turn baselines' input;
- for imaging-indication episodes, encounters after the imaging order are not
  observable (temporal leakage guard; the paper only instructed agents not to
  look);
- after the action budget is spent only the submit tool is accepted, mirroring
  the harness's forced final turn;
- the problem list shown by open_chart / view_problem_list is the chart's own
  documented history (the profile's chronic conditions), not the graph-derived
  encounter diagnoses that the simulator's tool returns and that constitute the
  patient-diagnosis reference;
- the reward is computed once, at submission, and the labels are never exposed.
"""

from __future__ import annotations

import json
import random
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg
from starlette.concurrency import run_in_threadpool

from epic_sim.app.config import settings
from eval.agents.prompts import SUBMIT_TOOL_SCHEMAS, build_patient_intro, build_system_prompt
from eval.score_one import InstanceNotFound, list_instances, load_instance, score_submission

OUTCOME_SECTIONS = {"assessment", "plan"}
SUBMIT_TOOLS = {"submit_diagnosis", "submit_summary", "submit_pre_read", "submit_rankings"}
MAX_OBS_CHARS = 8000

TASK_KEYS = {
    "patient_diagnosis": ("active_diagnoses", "chronic_conditions"),
    "context_summarization": ("summary",),
    "evidence_retrieval": ("rankings",),
    "imaging_indication": ("clinical_question", "differential", "findings"),
}


class EpisodeError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _pg() -> psycopg.Connection:
    return psycopg.connect(settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://"))


def _key(episode_id: str) -> str:
    return f"{settings.redis_session_prefix}env:{episode_id}"


def submit_tool_for(task: str) -> str:
    return SUBMIT_TOOL_SCHEMAS[task]["function"]["name"]


# ---------------------------------------------------------------------------
# Instance context (what the paper's runner loaded per item)
# ---------------------------------------------------------------------------

def _load_context(gt_id: int) -> dict[str, Any]:
    """Instance identifiers plus the task-specific prompt inputs. Never the labels."""
    with _pg() as conn:
        inst = load_instance(conn, gt_id)
        task = inst["task"]
        gt = inst["ground_truth"]
        ctx: dict[str, Any] = {
            "gt_id": inst["gt_id"], "task": task, "split": inst["split"],
            "patient_id": inst["patient_id"], "encounter_id": inst["encounter_id"],
            "variant": inst["variant"], "task_kwargs": {}, "allowed_encounter_ids": None,
            "chart_problems": [],
        }
        with conn.cursor() as cur:
            cur.execute("SELECT profile FROM longitudinal_patients WHERE patient_id = %s", (inst["patient_id"],))
            row = cur.fetchone()
            profile = (row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")) if row else {}
            ctx["chart_problems"] = [
                {"display_name": str(c if not isinstance(c, dict) else c.get("name") or c.get("condition") or ""),
                 "source": "chart_history"}
                for c in (profile.get("chronic_conditions") or []) if c
            ]
            if task == "context_summarization":
                ctx["task_kwargs"]["clinical_question"] = gt.get(
                    "clinical_question", "Summarize this patient's clinical course.")
            elif task == "evidence_retrieval":
                dx_ids = [d["diagnosis_id"] for d in gt.get("query_diagnoses", [])]
                names: dict[int, str] = {}
                if dx_ids:
                    cur.execute("SELECT diagnosis_id, display_name FROM diagnoses WHERE diagnosis_id = ANY(%s)", (dx_ids,))
                    names = {r[0]: r[1] for r in cur.fetchall()}
                ctx["task_kwargs"]["diagnosis_names"] = ", ".join(names.get(d, f"diagnosis_{d}") for d in dx_ids)
            elif task == "imaging_indication":
                cur.execute(
                    "SELECT modality, body_region, clinical_indication FROM imaging_orders WHERE encounter_id = %s LIMIT 1",
                    (inst["encounter_id"],),
                )
                row = cur.fetchone() or (None, None, None)
                ctx["task_kwargs"].update({
                    "modality": row[0] or "imaging study",
                    "body_region": row[1] or "unspecified",
                    "clinical_indication": row[2] or "clinical concern",
                })
                cur.execute(
                    """
                    SELECT e.encounter_id FROM longitudinal_encounters e
                    WHERE e.patient_id = %s AND e.encounter_order <= (
                        SELECT encounter_order FROM longitudinal_encounters WHERE encounter_id = %s)
                    """,
                    (inst["patient_id"], inst["encounter_id"]),
                )
                ctx["allowed_encounter_ids"] = sorted(r[0] for r in cur.fetchall())
    return ctx


def _tool_schemas(task: str, tool_definitions: list[dict]) -> list[dict]:
    """OpenAI-style function schemas: the EHR tools plus the task's submit tool
    in the scoring schema (the paper's harness did the same substitution)."""
    override = SUBMIT_TOOL_SCHEMAS[task]
    out = []
    for t in tool_definitions:
        name = t["name"]
        if name == override["function"]["name"]:
            out.append(override)
        elif name in SUBMIT_TOOLS:
            continue
        else:
            out.append({"type": "function", "function": {
                "name": name, "description": t["description"], "parameters": t["parameters"]}})
    return out


# ---------------------------------------------------------------------------
# Observation post-processing
# ---------------------------------------------------------------------------

def strip_outcome_sections(obj):
    """Recursively drop assessment/plan section entries from a tool result."""
    if isinstance(obj, list):
        return [strip_outcome_sections(i) for i in obj
                if not (isinstance(i, dict) and str(i.get("section_type", "")).lower() in OUTCOME_SECTIONS)]
    if isinstance(obj, dict):
        return {k: strip_outcome_sections(v) for k, v in obj.items()
                if not (k.lower() in OUTCOME_SECTIONS and isinstance(v, str))}
    return obj


def filter_future_encounters(obj, allowed: set[int]):
    """Remove anything tied to an encounter outside `allowed` (imaging episodes)."""
    if isinstance(obj, list):
        return [filter_future_encounters(i, allowed) for i in obj
                if not (isinstance(i, dict) and "encounter_id" in i and i["encounter_id"] not in allowed)]
    if isinstance(obj, dict):
        if "encounter_id" in obj and isinstance(obj["encounter_id"], int) and obj["encounter_id"] not in allowed:
            return {"error": "This encounter is after the imaging order and is not accessible in this task."}
        return {k: filter_future_encounters(v, allowed) for k, v in obj.items()}
    return obj


def replace_problem_list(tool: str, obs, chart_problems: list[dict]):
    """The simulator's problem list is the graph-derived reference for patient diagnosis
    (correct diagnoses of the source questions, with ICD-10 codes). In the environment it
    is replaced by the chart's documented history so the label cannot be read off a tool."""
    if tool == "view_problem_list":
        return list(chart_problems)
    if tool == "open_chart" and isinstance(obs, dict) and "active_problems" in obs:
        return {**obs, "active_problems": list(chart_problems)}
    return obs


def render(obs) -> str:
    text = json.dumps(obs, indent=2, default=str) if isinstance(obs, (dict, list)) else str(obs)
    if len(text) > MAX_OBS_CHARS:
        text = text[:MAX_OBS_CHARS] + "\n[OUTPUT TRUNCATED]"
    return text


def normalize_submission(task: str, obj) -> dict | None:
    """Coerce submit-tool arguments into the scorer's schema (paper harness rules)."""
    keys = TASK_KEYS.get(task, ())
    if isinstance(obj, dict):
        if "payload" in obj and isinstance(obj["payload"], (dict, list)):
            return normalize_submission(task, obj["payload"])
        if any(k in obj for k in keys):
            return obj
        if task == "patient_diagnosis" and "icd10" in obj:
            return {"active_diagnoses": [obj], "chronic_conditions": []}
        if task == "context_summarization":
            for k in ("text", "narrative", "assessment"):
                if isinstance(obj.get(k), str):
                    return {"summary": obj[k]}
        return None
    if isinstance(obj, list) and obj:
        if task == "patient_diagnosis":
            dx = [d for d in obj if isinstance(d, dict) and "icd10" in d]
            if dx:
                return {"active_diagnoses": dx, "chronic_conditions": []}
        if task == "evidence_retrieval":
            return {"rankings": obj}
    return None


# ---------------------------------------------------------------------------
# Episode store
# ---------------------------------------------------------------------------

async def _save(redis, ep: dict) -> None:
    ep["updated_at"] = datetime.now(timezone.utc).isoformat()
    await redis.setex(_key(ep["episode_id"]), settings.session_ttl_seconds, json.dumps(ep, default=str))


async def load_episode(redis, episode_id: str) -> dict:
    raw = await redis.get(_key(episode_id))
    if raw is None:
        raise EpisodeError(404, f"episode {episode_id} not found or expired")
    return json.loads(raw)


async def reset(redis, tool_definitions: list[dict], *, gt_id: int | None, task: str | None,
                split: str | None, seed: int | None, budget: int | None) -> dict:
    """Start an episode on a given instance, or on a random one of a task/split."""
    if gt_id is None:
        with _pg() as conn:
            items = await run_in_threadpool(list_instances, conn, task, split or "train", None, 5000, 0)
        if not items:
            raise EpisodeError(404, f"no instances for task={task} split={split or 'train'}")
        gt_id = random.Random(seed).choice(items)["gt_id"]
    try:
        ctx = await run_in_threadpool(_load_context, gt_id)
    except InstanceNotFound as exc:
        raise EpisodeError(404, str(exc)) from exc
    if task and task != ctx["task"]:
        raise EpisodeError(422, f"gt_id={gt_id} belongs to task '{ctx['task']}', not '{task}'")

    budget = budget or settings.env_default_budget
    kwargs = dict(ctx["task_kwargs"])
    instructions = build_system_prompt(task=ctx["task"], arm="structured", budget=budget,
                                       encounter_id=ctx["encounter_id"], gt_id=gt_id, **kwargs)
    intro = build_patient_intro(patient_id=ctx["patient_id"], task=ctx["task"],
                                encounter_id=ctx["encounter_id"], **kwargs)
    tools = _tool_schemas(ctx["task"], tool_definitions)
    ep = {
        "episode_id": str(uuid.uuid4()),
        "agent_token": secrets.token_urlsafe(24),
        "gt_id": gt_id, "task": ctx["task"], "split": ctx["split"], "variant": ctx["variant"],
        "patient_id": ctx["patient_id"], "encounter_id": ctx["encounter_id"],
        "allowed_encounter_ids": ctx["allowed_encounter_ids"],
        "chart_problems": ctx["chart_problems"],
        "task_inputs": kwargs,
        "budget": budget, "steps": 0, "done": False, "submitted": False, "reward": None,
        "trace": [], "created_at": datetime.now(timezone.utc).isoformat(),
        "brief": {"instructions": instructions, "intro": intro, "tools": tools},
    }
    await _save(redis, ep)
    return reset_view(ep)


def reset_view(ep: dict) -> dict:
    """The agent-facing brief for an episode (no labels, no reward)."""
    return {
        "episode_id": ep["episode_id"], "agent_token": ep["agent_token"],
        "gt_id": ep["gt_id"], "task": ep["task"], "split": ep["split"],
        "variant": ep["variant"], "patient_id": ep["patient_id"], "encounter_id": ep["encounter_id"],
        "budget": ep["budget"], "remaining": max(ep["budget"] - ep["steps"], 0),
        "submit_tool": submit_tool_for(ep["task"]),
        "instructions": ep["brief"]["instructions"], "intro": ep["brief"]["intro"],
        "task_inputs": ep["task_inputs"], "tools": ep["brief"]["tools"],
    }


AUTOSTART_KEY = "autostart"


async def ensure_autostart(redis, tool_definitions: list[dict], gt_id: int, budget: int | None) -> dict:
    """Create (once) the episode a container was started for (EPIC_SIM_AUTOSTART_GT_ID).

    Used by Harbor tasks: the agent container reads GET /env/current and never
    needs the scorer token; the verifier reads the reward with it afterwards.
    """
    existing = await redis.get(_key(AUTOSTART_KEY))
    if existing:
        try:
            return reset_view(await load_episode(redis, existing))
        except EpisodeError:
            pass
    view = await reset(redis, tool_definitions, gt_id=gt_id, task=None, split=None, seed=None, budget=budget)
    await redis.setex(_key(AUTOSTART_KEY), settings.session_ttl_seconds, view["episode_id"])
    return view


async def current_episode(redis) -> dict:
    eid = await redis.get(_key(AUTOSTART_KEY))
    if not eid:
        raise EpisodeError(404, "no autostart episode (EPIC_SIM_AUTOSTART_GT_ID is not set)")
    return reset_view(await load_episode(redis, eid))


def oracle_submission(gt_id: int) -> dict:
    """A submission built from the labels (for Harbor's oracle agent and for tests)."""
    with _pg() as conn:
        inst = load_instance(conn, gt_id)
        gt = inst["ground_truth"]
        task = inst["task"]
        if task == "patient_diagnosis":
            return {
                "active_diagnoses": [{"icd10": d["icd10"], "name": d.get("display_name", ""), "acuity": d.get("acuity") or "acute"}
                                     for d in gt.get("active_diagnoses", []) if not d.get("excluded_nondiagnostic")],
                "chronic_conditions": [{"icd10": d["icd10"], "name": d.get("display_name", ""), "acuity": "chronic"}
                                       for d in gt.get("chronic_conditions", []) if not d.get("excluded_nondiagnostic")],
            }
        if task == "context_summarization":
            if gt.get("variant") == "specialty_conditioned":
                tiers = gt.get("tiers", {})
                names = [f.get("display_name") for f in tiers.get("primary", []) + tiers.get("relevant", [])]
                if not names:
                    return {"summary": "No active problems relevant to this specialty are documented in the chart."}
                return {"summary": ". ".join(n for n in names if n) + "."}
            names = [f.get("display_name") or f.get("name") for f in gt.get("must_include_findings", [])]
            return {"summary": ". ".join(n for n in names if n) + "."}
        if task == "evidence_retrieval":
            with conn.cursor() as cur:
                cur.execute("SELECT passage_id FROM relevance_judgments WHERE gt_id = %s "
                            "ORDER BY relevance_grade DESC, passage_id LIMIT 20", (gt_id,))
                return {"rankings": [{"passage_id": r[0], "grade": 3} for r in cur.fetchall()]}
        if task == "imaging_indication":
            return {
                "clinical_question": gt.get("inferred_clinical_question", ""),
                "pre_read_summary": gt.get("pre_read_summary", ""),
                "must_include_findings": list(gt.get("must_include_findings", [])),
                "differential": [{"diagnosis": d.get("diagnosis", ""), "icd10": d.get("icd10", "")}
                                 for d in gt.get("differential_context", [])],
            }
        raise EpisodeError(422, f"no oracle for task {task}")


async def step(redis, db, dispatch, episode_id: str, name: str, arguments: dict,
               privileged: bool = True) -> dict:
    """Execute one action. `dispatch` is the agent router's tool dispatcher.

    `privileged` callers (scorer token) see the reward and metrics at submission;
    agent-token callers only learn that the episode is over.
    """
    from fastapi import HTTPException  # local import: keep the service free of router types

    ep = await load_episode(redis, episode_id)
    if ep["done"]:
        raise EpisodeError(409, "episode is finished; call reset")
    task = ep["task"]
    submit_tool = submit_tool_for(task)
    remaining = ep["budget"] - ep["steps"]
    base = {"episode_id": episode_id, "step": ep["steps"] + 1, "tool_name": name,
            "reward": 0.0, "done": False, "info": {}}

    # -- terminal action ---------------------------------------------------
    if name == submit_tool:
        prediction = normalize_submission(task, arguments)
        with _pg() as conn:
            result = await run_in_threadpool(score_submission, conn, ep["gt_id"], prediction or {}, task, None)
        ep.update({"done": True, "submitted": True, "reward": result["reward"], "steps": ep["steps"] + 1})
        ep["trace"].append({"step": ep["steps"], "tool_name": name, "submitted": True,
                            "forced": remaining <= 0, "malformed": prediction is None})
        await _save(redis, ep)
        obs = {"status": "submitted", "malformed": prediction is None}
        info = {"forced": remaining <= 0, "malformed": prediction is None, "steps": ep["steps"]}
        if privileged:
            info.update({"reward_metric": result["reward_metric"], "metrics": result["metrics"]})
        return {**base, "step": ep["steps"], "observation": obs, "observation_text": render(obs),
                "remaining": max(remaining - 1, 0), "reward": result["reward"] if privileged else None,
                "done": True, "info": info}

    # -- non-terminal actions ----------------------------------------------
    if name in SUBMIT_TOOLS:
        obs = {"error": f"This task is scored through {submit_tool}; use that tool to finish the episode."}
    elif remaining <= 0:
        obs = {"error": f"Action budget of {ep['budget']} spent. Only {submit_tool} is accepted now; "
                        f"submit your best answer."}
    else:
        try:
            raw = await dispatch(name, arguments or {}, db, "attending", episode_id)
        except HTTPException as exc:
            raw = {"error": str(exc.detail)}
        except KeyError as exc:
            raw = {"error": f"missing argument {exc}"}
        except Exception as exc:  # noqa: BLE001 — a bad tool call is an observation, not a crash
            raw = {"error": f"{type(exc).__name__}: {exc}"}
        obs = strip_outcome_sections(raw)
        obs = replace_problem_list(name, obs, ep.get("chart_problems") or [])
        if ep.get("allowed_encounter_ids") is not None:
            obs = filter_future_encounters(obs, set(ep["allowed_encounter_ids"]))
        ep["steps"] += 1
        remaining -= 1
    ep["trace"].append({"step": ep["steps"], "tool_name": name, "arguments": arguments,
                        "error": obs.get("error") if isinstance(obs, dict) else None})
    await _save(redis, ep)
    return {**base, "step": ep["steps"], "observation": obs, "observation_text": render(obs),
            "remaining": remaining, "info": {"steps": ep["steps"]}}


async def close(redis, episode_id: str) -> dict:
    ep = await load_episode(redis, episode_id)
    if not ep["done"]:
        ep["done"] = True
        ep["reward"] = 0.0
        await _save(redis, ep)
    return state_view(ep)


def state_view(ep: dict, privileged: bool = True) -> dict:
    view = {k: ep[k] for k in ("episode_id", "gt_id", "task", "split", "patient_id", "encounter_id",
                               "budget", "steps", "done", "submitted", "reward", "trace", "created_at")}
    if not privileged:
        view["reward"] = None
    return view
