"""Phase F orchestrator: patient selection, arm dispatch, trace storage.

Manages the complete agent evaluation pipeline — selecting patients,
running both arms (structured + bash) for each model, scoring results,
and computing agent-specific metrics.
"""

import json
import logging
import subprocess
import time
from datetime import datetime, timezone

from eval.adapters import create_adapter
from eval.agents import AgentResult
from eval.agents.agent_loop import AgentLoop
from eval.agents.bash_agent import BashAgentHarness
from eval.agents.prompts import build_patient_intro, build_system_prompt
from eval.agents.structured_agent import StructuredAgentHarness
from eval.config import EVAL_TASKS, MODEL_REGISTRY, get_pg_connection
from eval.scoring import compute_all_metrics

log = logging.getLogger(__name__)

# Phase F models (§F1)
AGENT_MODELS = [
    "gpt-5.3", "gemini-3.1",           # Tier 2 primary pair
    "glm-5-agent", "opus-4.6",         # Tier 3 replication
    "kimi-2.5-thinking-improved",       # Tier 3 (thinking, 2 workers max, 50 patients)
    # Capability-tier sweep (mid / low) for arms B and C. Gemma 3 is listed but is
    # not usable agentically: of the five OpenRouter providers serving it only
    # DeepInfra supports function calling, and that endpoint returns intermittent
    # 429s that stall the agent loop. Llama 4 Scout is the low tier in practice.
    "mistral-3", "llama-4", "gemma-3",
]

# Kimi thinking constraint: gateway 504s at higher concurrency (confirmed in E1.2)
KIMI_MAX_WORKERS = 2
KIMI_DEFAULT_PATIENTS = 50

# Phase F tasks — patient-level and encounter-level only (no question-level diagnosis_accuracy)
AGENT_TASKS = [
    "patient_diagnosis",
    "context_summarization",
    "evidence_retrieval",
    "imaging_indication",
]

# Default budget
DEFAULT_BUDGET = 40

# FastAPI base URL
DEFAULT_API_BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Patient Selection (§F1)
# ---------------------------------------------------------------------------

def select_patients(
    conn,
    split: str = "val",
    n: int = 100,
    min_encounters: int = 2,
    max_encounters: int = 8,
    exclude_ceiling: bool = True,
    ceiling_threshold: int = 7,
) -> list[dict]:
    """Select patients for Phase F evaluation.

    Criteria:
    - From the specified split
    - Have 2–8 encounters
    - Have GT for all required tasks (diagnosis, patient_dx, summarization, retrieval)
    - Optionally exclude ceiling items (≥7/11 models correct in E1 zero_shot)

    Returns list of dicts with patient_id, num_encounters, difficulty, gt_ids per task.
    """
    with conn.cursor() as cur:
        # Get patients with encounter counts
        cur.execute("""
            SELECT lp.patient_id, lp.num_encounters
            FROM longitudinal_patients lp
            WHERE lp.num_encounters BETWEEN %s AND %s
            ORDER BY lp.patient_id
        """, (min_encounters, max_encounters))
        candidates = {row[0]: {"patient_id": row[0], "num_encounters": row[1]} for row in cur.fetchall()}

        if not candidates:
            log.warning("No patients found with %d-%d encounters", min_encounters, max_encounters)
            return []

        # Get GT coverage per patient per task
        # Mirror the single-turn LLM-only loaders: skip non-diagnostic GT, and take
        # only the unconditioned summarization row per patient. Each patient also has
        # ~6 specialty-conditioned summarization rows; including them would run ~1,183
        # summarization sessions instead of 100 and would not be comparable to the
        # LLM-only baseline, whose loader filters on the same variant.
        cur.execute("""
            SELECT bgt.patient_id, bgt.task, bgt.gt_id, bgt.difficulty
            FROM benchmark_ground_truth bgt
            WHERE bgt.split = %s
              AND bgt.patient_id IS NOT NULL
              AND bgt.patient_id = ANY(%s)
              AND bgt.is_diagnostic
              AND NOT (bgt.task = 'context_summarization'
                       AND COALESCE(bgt.ground_truth->>'variant', 'unconditioned')
                           <> 'unconditioned')
        """, (split, list(candidates.keys())))

        patient_tasks: dict[int, dict[str, list]] = {}
        for pid, task, gt_id, difficulty in cur.fetchall():
            if pid not in patient_tasks:
                patient_tasks[pid] = {}
            if task not in patient_tasks[pid]:
                patient_tasks[pid][task] = []
            patient_tasks[pid][task].append({"gt_id": gt_id, "difficulty": difficulty})

        # Filter: must have at least patient_diagnosis and context_summarization GT
        required_tasks = {"patient_diagnosis", "context_summarization"}
        eligible = []
        for pid, tasks in patient_tasks.items():
            if pid not in candidates:
                continue
            if required_tasks.issubset(tasks.keys()):
                info = candidates[pid].copy()
                info["tasks"] = tasks
                eligible.append(info)

        log.info("Eligible patients: %d/%d (have required GT)", len(eligible), len(candidates))

        # Exclude ceiling items if requested
        if exclude_ceiling and eligible:
            patient_ids = [p["patient_id"] for p in eligible]
            cur.execute("""
                SELECT bgt.patient_id, COUNT(DISTINCT er.model_name) as correct_models
                FROM benchmark_ground_truth bgt
                JOIN evaluation_predictions ep ON bgt.gt_id = ep.gt_id
                JOIN evaluation_runs er ON ep.run_id = er.run_id
                WHERE bgt.patient_id = ANY(%s)
                  AND er.prompt_strategy = 'zero_shot'
                  AND er.completed_at IS NOT NULL
                  AND ep.score >= 0.5
                GROUP BY bgt.patient_id
                HAVING COUNT(DISTINCT er.model_name) >= %s
            """, (patient_ids, ceiling_threshold))
            ceiling_pids = {row[0] for row in cur.fetchall()}
            before = len(eligible)
            eligible = [p for p in eligible if p["patient_id"] not in ceiling_pids]
            log.info("Excluded %d ceiling patients (≥%d models correct)",
                     before - len(eligible), ceiling_threshold)

    # Sort by difficulty (prefer hard) then by encounter count spread
    eligible.sort(key=lambda p: (-p["num_encounters"], p["patient_id"]))

    selected = eligible[:n]
    log.info("Selected %d patients for Phase F", len(selected))
    return selected


# ---------------------------------------------------------------------------
# Agent Session Execution
# ---------------------------------------------------------------------------

def _preloaded_chart(patient_id: int, task: str = "", gt_id: int | None = None) -> str:
    """Assemble the context the single-turn LLM-only baseline receives.

    Task-aware, because "the whole context" differs by task. Evidence retrieval
    is scored on ranking a fixed passage corpus by `passage_id`; handing it the
    chart narrative instead makes the agent invent IDs like
    "enc1_fhr_decelerations" that match no passage, scoring 0 regardless of
    reasoning quality. Every other task takes the longitudinal chart, assembled
    by the same helper the baseline uses (which drops assessment/plan).
    """
    from eval.config import get_pg_connection

    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            if task == "evidence_retrieval" and gt_id is not None:
                import json as _json
                from eval.tasks.retrieval import _build_corpus

                cur.execute(
                    "SELECT patient_id, ground_truth FROM benchmark_ground_truth WHERE gt_id = %s",
                    (gt_id,),
                )
                row = cur.fetchone()
                pid = row[0] if row else patient_id
                gt = row[1] if isinstance(row[1], dict) else _json.loads(row[1])
                corpus = _build_corpus(cur, pid, gt)
                body = "\n\n".join(f"[{p.passage_id}] {p.text}" for p in corpus)
                return (
                    "The full candidate passage corpus is provided below. Grade these "
                    "passages directly — no further retrieval is required. Use the exact "
                    "passage_id shown in brackets.\n\n"
                    "=== BEGIN PASSAGES ===\n" + body + "\n=== END PASSAGES ==="
                )

            from eval.tasks.patient_diagnosis import _assemble_patient_ehr
            text = _assemble_patient_ehr(cur, patient_id)
    finally:
        conn.close()
    return (
        "The complete chart for this patient is provided below. You may still use "
        "the tools to verify or search, but no further retrieval is required.\n\n"
        "=== BEGIN PATIENT CHART ===\n" + text + "\n=== END PATIENT CHART ==="
    )


def run_agent_session(
    model_name: str,
    arm: str,
    task: str,
    patient_id: int,
    gt_id: int,
    encounter_id: int | None = None,
    api_base: str = DEFAULT_API_BASE,
    budget: int = DEFAULT_BUDGET,
    container_id: str | None = None,
    auth_token: str | None = None,
    **task_kwargs,
) -> AgentResult:
    """Run a single agent session for one (model, arm, task, patient).

    Args:
        model_name: Model from AGENT_MODELS.
        arm: 'structured' or 'bash'.
        task: One of EVAL_TASKS.
        patient_id: Target patient.
        gt_id: Ground truth ID for scoring.
        encounter_id: For question/encounter-level tasks.
        api_base: FastAPI base URL.
        budget: Action budget.
        container_id: Docker container ID (bash arm only).
        auth_token: Pre-authenticated JWT.
        **task_kwargs: Additional task-specific args (diagnosis_names, modality, etc.)
    """
    config = MODEL_REGISTRY[model_name]
    adapter = create_adapter(config)

    # Build prompts
    system_prompt = build_system_prompt(
        task=task,
        arm=arm,
        budget=budget,
        api_base=api_base,
        token=auth_token or "",
        encounter_id=encounter_id,
        gt_id=gt_id,
        **task_kwargs,
    )

    user_intro = build_patient_intro(
        patient_id=patient_id,
        task=task,
        encounter_id=encounter_id,
        **task_kwargs,
    )

    # Cell B ("context" arm): hand the agent the same chart the single-turn
    # LLM-only baseline receives, so that arm and baseline differ only by the
    # agentic loop. Chart tools stay available; whether the agent still uses
    # them is itself a measurement.
    if arm == "context":
        user_intro = user_intro + "\n\n" + _preloaded_chart(patient_id, task, gt_id)

    # Create harness
    if arm == "bash":
        if not container_id:
            raise ValueError("container_id required for bash arm")
        harness = BashAgentHarness(
            container_id=container_id,
            api_base=api_base,
            auth_token=auth_token or "",
            max_commands=budget,
        )
    else:
        session_id = _create_api_session(api_base, auth_token, patient_id, gt_id)
        harness = StructuredAgentHarness(
            api_base=api_base,
            token=auth_token or "",
            session_id=session_id,
            gt_id=gt_id,
            max_calls=budget,
        )

    # Run agent loop
    loop = AgentLoop(
        model_adapter=adapter,
        harness=harness,
        task=task,
        patient_id=patient_id,
        gt_id=gt_id,
        system_prompt=system_prompt,
        user_intro=user_intro,
        max_turns=budget,
        encounter_id=encounter_id,
    )

    result = loop.run()
    log.info(
        "Agent session: model=%s arm=%s task=%s patient=%d gt=%d "
        "turns=%d submitted=%s",
        model_name, arm, task, patient_id, gt_id,
        result.turns_used, result.submitted,
    )
    return result


def _create_api_session(api_base: str, token: str, patient_id: int, gt_id: int) -> str:
    """Create a session via the Epic API for structured arm."""
    import httpx
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{api_base}/epic/sessions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json={
                    "patient_id": patient_id,
                    "gt_id": gt_id,
                    "role": "attending",
                    "department": "Internal Medicine",
                },
            )
            resp.raise_for_status()
            return resp.json()["session_id"]
    except Exception as e:
        log.warning("Failed to create API session: %s (using fallback)", e)
        import uuid
        return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# DB Storage
# ---------------------------------------------------------------------------

def store_agent_prediction(
    conn,
    run_id: int,
    gt_id: int,
    result: AgentResult,
    arm: str,
    prompt_strategy: str = "agent",
) -> int:
    """Store an agent prediction in evaluation_predictions + agent_traces."""
    config_json = {
        "arm": arm,
        "budget": result.turns_used + (1 if result.budget_exhausted else 0),
        "turns_used": result.turns_used,
        "submitted": result.submitted,
        "forced_submission": result.forced_submission,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
    }

    with conn.cursor() as cur:
        # Store prediction
        cur.execute("""
            INSERT INTO evaluation_predictions
                (run_id, gt_id, prediction, latency_ms, token_count,
                 input_tokens, output_tokens, prompt_strategy, raw_output)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, gt_id) DO NOTHING
            RETURNING prediction_id
        """, (
            run_id,
            gt_id,
            json.dumps(result.prediction) if result.prediction else json.dumps({}),
            result.total_latency_ms,
            result.total_input_tokens + result.total_output_tokens,
            result.total_input_tokens,
            result.total_output_tokens,
            prompt_strategy,
            json.dumps(config_json),
        ))
        row = cur.fetchone()
        if not row:
            log.warning("Prediction already exists for run_id=%d gt_id=%d", run_id, gt_id)
            return -1
        prediction_id = row[0]

        # Store trace entries
        for entry in result.trace:
            cur.execute("""
                INSERT INTO agent_traces
                    (prediction_id, turn_number, action_type, action_name,
                     action_args, output_text, output_length, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                prediction_id,
                entry.get("turn", 0),
                entry.get("action_type", "unknown"),
                entry.get("action_name"),
                json.dumps(entry.get("action_args", {})),
                None,  # Don't store full output text (can be large)
                entry.get("output_length", 0),
                entry.get("latency_ms", 0),
            ))

    conn.commit()
    return prediction_id


def create_agent_run(
    conn,
    model_name: str,
    arm: str,
    task: str,
    split: str = "val",
) -> int:
    """Create an evaluation_runs entry for an agent run."""
    method_type = f"agent_{arm}"
    run_name = f"{model_name}_{task}_{split}_agent_{arm}"

    config = MODEL_REGISTRY[model_name]
    run_config = {
        "model_id": config.model_id,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "arm": arm,
        "budget": DEFAULT_BUDGET,
    }

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO evaluation_runs
                (run_name, method_name, method_type, task, split,
                 config, prompt_strategy, model_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING run_id
        """, (
            run_name,
            model_name,
            method_type,
            task,
            split,
            json.dumps(run_config),
            "agent",
            model_name,
        ))
        run_id = cur.fetchone()[0]
    conn.commit()
    log.info("Created agent run_id=%d: %s/%s/%s/%s", run_id, model_name, arm, task, split)
    return run_id


def complete_agent_run(conn, run_id: int, metrics: dict):
    """Mark an agent run as completed with metrics."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE evaluation_runs
            SET metrics = %s, completed_at = %s
            WHERE run_id = %s
        """, (
            json.dumps(metrics),
            datetime.now(timezone.utc).isoformat(),
            run_id,
        ))
    conn.commit()


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def _load_task_kwargs(conn, task: str, patients: list[dict]) -> dict[int, dict]:
    """Pre-load task-specific kwargs needed by prompt templates.

    Returns {gt_id: {kwarg_name: value, ...}} for each GT item.
    """
    if task == "patient_diagnosis":
        return {}  # No extra kwargs needed

    # Collect all gt_ids for this task
    all_gt_ids = []
    for p in patients:
        for gt_entry in p.get("tasks", {}).get(task, []):
            all_gt_ids.append(gt_entry["gt_id"])

    if not all_gt_ids:
        return {}

    result = {}

    with conn.cursor() as cur:
        if task == "context_summarization":
            # Need clinical_question from GT JSON
            cur.execute("""
                SELECT gt_id, ground_truth
                FROM benchmark_ground_truth
                WHERE gt_id = ANY(%s)
            """, (all_gt_ids,))
            for gt_id, gt_json in cur.fetchall():
                gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
                result[gt_id] = {
                    "clinical_question": gt.get("clinical_question", "Summarize this patient's clinical course."),
                }

        elif task == "evidence_retrieval":
            # Need diagnosis_names from query_diagnoses → diagnoses table
            cur.execute("""
                SELECT gt_id, ground_truth
                FROM benchmark_ground_truth
                WHERE gt_id = ANY(%s)
            """, (all_gt_ids,))
            gt_dx_ids = {}
            for gt_id, gt_json in cur.fetchall():
                gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
                dx_ids = [d["diagnosis_id"] for d in gt.get("query_diagnoses", [])]
                gt_dx_ids[gt_id] = dx_ids

            # Batch-load diagnosis names
            all_dx_ids = list({did for ids in gt_dx_ids.values() for did in ids})
            dx_names = {}
            if all_dx_ids:
                cur.execute(
                    "SELECT diagnosis_id, display_name FROM diagnoses WHERE diagnosis_id = ANY(%s)",
                    (all_dx_ids,),
                )
                dx_names = {row[0]: row[1] for row in cur.fetchall()}

            for gt_id, dx_ids in gt_dx_ids.items():
                names = [dx_names.get(did, f"diagnosis_{did}") for did in dx_ids]
                result[gt_id] = {
                    "diagnosis_names": ", ".join(names),
                }

        elif task == "imaging_indication":
            # Need modality, body_region, clinical_indication from imaging_orders + encounter_id from GT
            cur.execute("""
                SELECT bgt.gt_id, bgt.encounter_id,
                       io.modality, io.body_region, io.clinical_indication
                FROM benchmark_ground_truth bgt
                LEFT JOIN imaging_orders io ON bgt.encounter_id = io.encounter_id
                WHERE bgt.gt_id = ANY(%s)
            """, (all_gt_ids,))
            for gt_id, enc_id, modality, body_region, indication in cur.fetchall():
                result[gt_id] = {
                    "encounter_id": enc_id,
                    "modality": modality or "imaging study",
                    "body_region": body_region or "unspecified",
                    "clinical_indication": indication or "clinical concern",
                }

    return result


def run_agent_evaluation(
    model_name: str,
    arm: str,
    task: str,
    split: str = "val",
    patients: list[dict] | None = None,
    pilot: int | None = None,
    max_patients: int | None = None,
    api_base: str = DEFAULT_API_BASE,
    container_id: str | None = None,
    auth_token: str | None = None,
    resume_run_id: int | None = None,
) -> int:
    """Run a complete agent evaluation for one (model, arm, task).

    Args:
        resume_run_id: If set, resume an existing incomplete run instead of
            creating a new one. Skips gt_ids that already have predictions.
        max_patients: Truncate patient list to N (useful for Kimi 50-patient runs).

    Returns the run_id.
    """
    conn = get_pg_connection()

    try:
        # Apply model-specific patient cap (Kimi thinking defaults to 50)
        effective_n = pilot or max_patients or 100
        if "kimi" in model_name and max_patients is None and pilot is None:
            effective_n = KIMI_DEFAULT_PATIENTS
            log.info("Kimi model detected — defaulting to %d patients", effective_n)

        # Select patients if not provided
        if patients is None:
            patients = select_patients(conn, split=split, n=effective_n)

        if pilot and len(patients) > pilot:
            patients = patients[:pilot]
        elif max_patients and len(patients) > max_patients:
            patients = patients[:max_patients]

        # Create or resume run
        if resume_run_id is not None:
            run_id = resume_run_id
            # Verify the run exists and matches
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT model_name, task, config->>'arm' FROM evaluation_runs WHERE run_id = %s",
                    (run_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Run {run_id} not found")
                if row[0] != model_name or row[1] != task or row[2] != arm:
                    raise ValueError(
                        f"Run {run_id} is {row[0]}/{row[2]}/{row[1]}, "
                        f"not {model_name}/{arm}/{task}"
                    )
            log.info("Resuming run_id=%d", run_id)
        else:
            run_id = create_agent_run(conn, model_name, arm, task, split)

        # Get already-completed gt_ids for resume
        with conn.cursor() as cur:
            cur.execute(
                "SELECT gt_id FROM evaluation_predictions WHERE run_id = %s",
                (run_id,),
            )
            completed_gt_ids = {row[0] for row in cur.fetchall()}

        if completed_gt_ids:
            log.info("Resuming: %d predictions already completed, skipping",
                     len(completed_gt_ids))

        # Pre-load task-specific kwargs from GT / DB
        task_kwargs_map = _load_task_kwargs(conn, task, patients)

        # Run sessions
        success = 0
        errors = 0

        for patient_info in patients:
            pid = patient_info["patient_id"]
            tasks_gt = patient_info.get("tasks", {})

            if task not in tasks_gt:
                continue

            for gt_entry in tasks_gt[task]:
                gt_id = gt_entry["gt_id"]
                if gt_id in completed_gt_ids:
                    continue

                task_kwargs = task_kwargs_map.get(gt_id, {})

                try:
                    result = run_agent_session(
                        model_name=model_name,
                        arm=arm,
                        task=task,
                        patient_id=pid,
                        gt_id=gt_id,
                        api_base=api_base,
                        container_id=container_id,
                        auth_token=auth_token,
                        **task_kwargs,
                    )
                    store_agent_prediction(conn, run_id, gt_id, result, arm)
                    success += 1
                except Exception as e:
                    errors += 1
                    log.error("Failed patient=%d gt=%d: %s", pid, gt_id, e)
                    # Rollback to recover from aborted transaction state
                    try:
                        conn.rollback()
                    except Exception:
                        pass

        log.info("Agent eval complete: %d success, %d errors", success, errors)

        # Score the run
        from eval.tasks import TASK_LOADERS
        loader = TASK_LOADERS[task]
        inputs = loader(conn, split=split)
        input_map = {inp.gt_id: inp for inp in inputs}

        with conn.cursor() as cur:
            cur.execute("""
                SELECT ep.gt_id, ep.prediction, bgt.ground_truth
                FROM evaluation_predictions ep
                JOIN benchmark_ground_truth bgt ON ep.gt_id = bgt.gt_id
                WHERE ep.run_id = %s
            """, (run_id,))
            rows = cur.fetchall()

        predictions = []
        ground_truths = []
        for gt_id, pred_json, gt_json in rows:
            pred = pred_json if isinstance(pred_json, dict) else json.loads(pred_json)
            gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
            predictions.append(pred)
            ground_truths.append(gt)

            if task == "evidence_retrieval":
                inp_obj = input_map.get(gt_id)
                if inp_obj:
                    gt["_judgments"] = inp_obj.judgments

        if predictions:
            metrics = compute_all_metrics(task, predictions, ground_truths)
            metrics["n_predictions"] = len(predictions)
            metrics["n_submitted"] = success
            metrics["n_errors"] = errors
        else:
            metrics = {"n_predictions": 0, "n_submitted": 0, "n_errors": errors}

        complete_agent_run(conn, run_id, metrics)
        log.info("Run %d completed. Metrics: %s", run_id, json.dumps(metrics, indent=2))
        return run_id

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bash container management
# ---------------------------------------------------------------------------

def start_bash_container(compose_file: str = "docker-compose.yml") -> str:
    """Start the bash_sandbox container and return its container ID.

    `bash_sandbox` lives in docker-compose.override.yml. Passing `-f` explicitly
    disables Compose's automatic override merge, so the service would not be
    found; include the override file when it exists.
    """
    import os as _os
    files = ["-f", compose_file]
    override = "docker-compose.override.yml"
    if _os.path.exists(override):
        files += ["-f", override]

    result = subprocess.run(
        ["docker", "compose", *files, "up", "-d", "bash_sandbox"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start bash_sandbox: {result.stderr}")

    # Get container ID
    result = subprocess.run(
        ["docker", "compose", *files, "ps", "-q", "bash_sandbox"],
        capture_output=True, text=True,
    )
    container_id = result.stdout.strip()
    if not container_id:
        raise RuntimeError("bash_sandbox container not found after start")

    log.info("bash_sandbox started: %s", container_id)
    return container_id


def stop_bash_container(compose_file: str = "docker-compose.yml"):
    """Stop the bash_sandbox container."""
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "stop", "bash_sandbox"],
        capture_output=True, text=True,
    )
    log.info("bash_sandbox stopped")


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def get_auth_token(api_base: str = DEFAULT_API_BASE) -> str:
    """Authenticate and return a JWT token (attending role)."""
    import httpx
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            f"{api_base}/auth/token",
            json={"username": "agent_attending", "password": "agent_pass_2026"},
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
