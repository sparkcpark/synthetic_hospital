"""Stage 10: Build Ground Truth for all evaluation tasks.

Sub-stages:
  10a: Diagnosis Accuracy — question-level (deterministic)
  10b: Diagnosis Accuracy — patient-level (deterministic)
  10c: Context Summarization — patient-level (LLM)
  10d: Evidence Retrieval — patient-level (deterministic)
  10e: Imaging Clinical Indication — encounter-level (LLM)

Usage:
    python -m etl.stages.s10_ground_truth --step 10a [--pilot N] [--db PATH]
    python -m etl.stages.s10_ground_truth --step 10c [--pilot N] [--workers N]
    python -m etl.stages.s10_ground_truth --step 10e [--pilot N] [--workers N]
    python -m etl.stages.s10_ground_truth --all [--pilot N] [--workers N]
    python -m etl.stages.s10_ground_truth --verify-only [--db PATH]
    python -m etl.stages.s10_ground_truth --export-csv [--db PATH]
"""

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from etl.config import DB_PATH, DATA_DIR
from etl.db import get_connection
from etl.stages.s05_ontology import (
    MODEL,
    _call_with_retry,
)
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s10_ground_truth")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_10C = "s10c_patient_summary"
STAGE_10C_VISIT = "s10c_current_visit"
STAGE_10E = "s10e_imaging"
DEFAULT_WORKERS = 5

# Cap on findings/deltas stored per current-visit GT row (sanity).
CURRENT_VISIT_MAX_FINDINGS = 20

# Per-encounter prior-note length fed to the reference-summary prompt.
# None = full notes (no truncation); set an int to cap chars/encounter.
CURRENT_VISIT_PRIOR_NOTE_CHARS: int | None = None

# Number of index encounters (anchors) to sample per patient for the eval set.
# Each anchors a current-visit summary with all later encounters held out.
CURRENT_VISIT_INDEX_POINTS = 3

# Map section types to finding types for evidence grading fallback
SECTION_FINDING_TYPE_MAP: dict[str, set[str]] = {
    "demographics": {"demographic"},
    "chief_complaint": {"symptom"},
    "hpi": {"symptom", "sign", "history_item"},
    "pmh": {"history_item"},
    "psh": {"history_item"},
    "medications": {"medication"},
    "allergies": {"medication"},
    "family_history": {"history_item"},
    "social_history": {"history_item"},
    "ros": {"symptom"},
    "vitals": {"vital_sign"},
    "physical_exam": {"sign"},
    "labs": {"lab_value"},
    "imaging": {"imaging_finding"},
    "pathology": {"procedure_result"},
    "other_studies": {"procedure_result", "imaging_finding"},
}

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

_MIGRATION_SQL = """
-- Drop empty benchmark tables in FK order (all have 0 rows)
DROP TABLE IF EXISTS imaging_orders;
DROP TABLE IF EXISTS evaluation_predictions;
DROP TABLE IF EXISTS evaluation_runs;
DROP TABLE IF EXISTS relevance_judgments;
DROP TABLE IF EXISTS benchmark_ground_truth;

CREATE TABLE IF NOT EXISTS benchmark_ground_truth (
    gt_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task            TEXT NOT NULL CHECK (task IN (
        'diagnosis_accuracy', 'context_summarization',
        'evidence_retrieval', 'imaging_indication'
    )),
    granularity     TEXT NOT NULL CHECK (granularity IN ('question', 'patient', 'encounter')),
    question_id     INTEGER REFERENCES board_questions(question_id),
    patient_id      INTEGER REFERENCES longitudinal_patients(patient_id),
    encounter_id    INTEGER REFERENCES longitudinal_encounters(encounter_id),
    ground_truth    TEXT NOT NULL,
    difficulty      TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    split           TEXT CHECK (split IN ('train', 'val', 'test')),
    num_diagnoses   INTEGER,
    num_evidence    INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    CHECK (
        (granularity = 'question' AND question_id IS NOT NULL) OR
        (granularity = 'patient' AND patient_id IS NOT NULL) OR
        (granularity = 'encounter' AND encounter_id IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_bgt_task ON benchmark_ground_truth(task);
CREATE INDEX IF NOT EXISTS idx_bgt_split ON benchmark_ground_truth(split);
CREATE INDEX IF NOT EXISTS idx_bgt_question ON benchmark_ground_truth(question_id);
CREATE INDEX IF NOT EXISTS idx_bgt_patient ON benchmark_ground_truth(patient_id);
CREATE INDEX IF NOT EXISTS idx_bgt_encounter ON benchmark_ground_truth(encounter_id);

CREATE TABLE IF NOT EXISTS relevance_judgments (
    judgment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    gt_id           INTEGER NOT NULL REFERENCES benchmark_ground_truth(gt_id),
    passage_id      TEXT NOT NULL,
    passage_source  TEXT NOT NULL CHECK (passage_source IN (
        'ehr_section', 'fact_card', 'encounter_section'
    )),
    relevance_grade INTEGER NOT NULL CHECK (relevance_grade BETWEEN 0 AND 3),
    rationale       TEXT,
    source          TEXT NOT NULL CHECK (source IN ('rule_based', 'llm', 'manual'))
);
CREATE INDEX IF NOT EXISTS idx_rj_gt ON relevance_judgments(gt_id);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name        TEXT NOT NULL,
    method_name     TEXT NOT NULL,
    method_type     TEXT NOT NULL CHECK (method_type IN (
        'llm_direct', 'rag', 'fine_tuned', 'ensemble',
        'sparse_retrieval', 'dense_retrieval', 'hybrid_retrieval'
    )),
    task            TEXT NOT NULL CHECK (task IN (
        'diagnosis_accuracy', 'context_summarization',
        'evidence_retrieval', 'imaging_indication'
    )),
    split           TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    config          TEXT,
    metrics         TEXT,
    started_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at    TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES evaluation_runs(run_id),
    gt_id           INTEGER NOT NULL REFERENCES benchmark_ground_truth(gt_id),
    prediction      TEXT NOT NULL,
    score           REAL,
    latency_ms      INTEGER,
    token_count     INTEGER,
    raw_output      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_ep_run ON evaluation_predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_ep_gt ON evaluation_predictions(gt_id);

CREATE TABLE IF NOT EXISTS imaging_orders (
    order_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id        INTEGER NOT NULL REFERENCES longitudinal_encounters(encounter_id),
    gt_id               INTEGER NOT NULL REFERENCES benchmark_ground_truth(gt_id),
    modality            TEXT NOT NULL,
    body_region         TEXT NOT NULL,
    clinical_indication TEXT NOT NULL,
    ordering_provider   TEXT,
    order_priority      TEXT CHECK (order_priority IN ('routine', 'stat', 'urgent')),
    order_datetime      TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_io_encounter ON imaging_orders(encounter_id);
CREATE INDEX IF NOT EXISTS idx_io_gt ON imaging_orders(gt_id);
"""


def _apply_schema_migration_10(conn: sqlite3.Connection) -> None:
    """Drop empty benchmark tables and recreate with updated schema."""
    # Verify tables are empty before dropping
    for table in ["benchmark_ground_truth", "relevance_judgments",
                  "evaluation_runs", "evaluation_predictions"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            if count > 0:
                log.warning(f"Table {table} has {count} rows — skipping migration")
                return
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet

    log.info("Applying Stage 10 schema migration (all benchmark tables empty)...")
    conn.executescript(_MIGRATION_SQL)
    conn.commit()
    log.info("Schema migration complete.")


# ---------------------------------------------------------------------------
# Difficulty helpers
# ---------------------------------------------------------------------------

def _compute_difficulty_question(
    num_distractors: int,
    num_key_findings: int,
    num_secondary: int,
    has_pathognomonic: bool,
) -> str:
    score = 0.0
    if num_distractors <= 3:
        score += 0
    elif num_distractors <= 5:
        score += 1
    else:
        score += 2
    if has_pathognomonic:
        score -= 1
    if num_key_findings >= 5:
        score += 0
    elif num_key_findings >= 3:
        score += 1
    else:
        score += 2
    if num_secondary >= 3:
        score += 1
    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    return "hard"


def _compute_difficulty_patient(
    num_encounters: int,
    num_unique_dx: int,
    has_chronic_and_acute: bool,
    num_organ_systems: int,
) -> str:
    score = 0.0
    if num_encounters <= 2:
        score += 0
    elif num_encounters <= 5:
        score += 1
    else:
        score += 2
    if num_organ_systems >= 3:
        score += 1.5
    elif num_organ_systems >= 2:
        score += 0.5
    if has_chronic_and_acute:
        score += 1
    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    return "hard"


def _compute_difficulty_imaging(
    indication_word_count: int,
    num_differential: int,
    is_stat: bool,
) -> str:
    score = 0.0
    if indication_word_count <= 2:
        score += 2
    elif indication_word_count <= 4:
        score += 1
    if num_differential >= 4:
        score += 1.5
    elif num_differential >= 2:
        score += 0.5
    if is_stat:
        score += 0.5
    if score <= 1:
        return "easy"
    elif score <= 2.5:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Cache helpers (same pattern as s09)
# ---------------------------------------------------------------------------

def _compute_input_hash(prefix: str, key: int) -> str:
    return hashlib.sha256(f"{prefix}:{key}".encode()).hexdigest()


def _check_cache(conn: sqlite3.Connection, stage: str, input_hash: str) -> dict | None:
    row = conn.execute(
        "SELECT output_json FROM llm_call_log WHERE stage = ? AND input_hash = ? AND error IS NULL",
        (stage, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Sub-stage 10a: Diagnosis Accuracy — Question-Level (Deterministic)
# ---------------------------------------------------------------------------

def run_10a(conn: sqlite3.Connection, pilot: int | None = None) -> dict:
    """Diagnosis Accuracy ground truth: question-level (deterministic)."""
    log.info("Sub-stage 10a: Diagnosis Accuracy (question-level)...")

    # Phase 1: Load data
    rows = conn.execute("""
        SELECT qd.question_id, qd.diagnosis_id, d.icd10_code, d.display_name,
               d.snomed_id, qd.role, qd.confidence
        FROM question_diagnoses qd
        JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
        ORDER BY qd.question_id, qd.role, qd.confidence DESC
    """).fetchall()

    # Group by question_id
    dx_by_q: dict[int, list[tuple]] = defaultdict(list)
    for r in rows:
        dx_by_q[r[0]].append(r)

    # Load key finding counts + pathognomonic status per question
    qf_rows = conn.execute("""
        SELECT qf.question_id, qf.relevance, qf.finding_id
        FROM question_findings qf
    """).fetchall()
    key_count: dict[int, int] = defaultdict(int)
    all_qids: set[int] = set()
    finding_by_q: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for qid, rel, fid in qf_rows:
        all_qids.add(qid)
        if rel:
            finding_by_q[qid][rel].add(fid)
            if rel == "key":
                key_count[qid] += 1

    # Load pathognomonic findings per diagnosis
    pathog_findings: set[int] = set()
    for (fid,) in conn.execute(
        "SELECT DISTINCT finding_id FROM diagnosis_findings WHERE relationship = 'pathognomonic'"
    ).fetchall():
        pathog_findings.add(fid)

    question_ids = sorted(dx_by_q.keys())
    if pilot:
        question_ids = question_ids[:pilot]

    # Phase 2: Delete existing + Insert
    conn.execute(
        "DELETE FROM benchmark_ground_truth WHERE task = 'diagnosis_accuracy' AND granularity = 'question'"
    )

    inserted = 0
    for qid in question_ids:
        dx_rows = dx_by_q[qid]
        primary = None
        alternatives = []
        differential = []
        secondary = []

        for _, dx_id, icd10, name, snomed, role, conf in dx_rows:
            entry = {
                "diagnosis_id": dx_id,
                "icd10": icd10,
                "snomed_id": snomed,
                "display_name": name,
            }
            if role == "correct":
                if primary is None:
                    primary = entry
                else:
                    alternatives.append(entry)
            elif role == "distractor":
                entry["rank"] = len(differential) + 2
                differential.append(entry)
            elif role in ("secondary", "predisposing"):
                secondary.append(entry)

        if primary is None:
            continue

        gt = {
            "primary_diagnosis": primary,
            "acceptable_alternatives": alternatives,
            "differential": differential,
            "secondary_diagnoses": secondary,
        }

        # Difficulty
        num_key = key_count.get(qid, 0)
        key_fids = finding_by_q.get(qid, {}).get("key", set())
        has_pathog = bool(key_fids & pathog_findings)
        difficulty = _compute_difficulty_question(
            len(differential), num_key, len(secondary), has_pathog
        )

        conn.execute(
            "INSERT INTO benchmark_ground_truth "
            "(task, granularity, question_id, ground_truth, difficulty, num_diagnoses) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("diagnosis_accuracy", "question", qid,
             json.dumps(gt, ensure_ascii=False), difficulty,
             1 + len(differential)),
        )
        inserted += 1

    conn.commit()
    log.info(f"Sub-stage 10a complete: {inserted} question-level dx accuracy entries")
    return {"inserted": inserted}


# ---------------------------------------------------------------------------
# Sub-stage 10b: Diagnosis Accuracy — Patient-Level (Deterministic)
# ---------------------------------------------------------------------------

def run_10b(conn: sqlite3.Connection, pilot: int | None = None) -> dict:
    """Diagnosis Accuracy ground truth: patient-level (deterministic)."""
    log.info("Sub-stage 10b: Diagnosis Accuracy (patient-level)...")

    # Load patient encounter data
    enc_rows = conn.execute("""
        SELECT le.patient_id, le.encounter_id, le.encounter_date,
               le.encounter_order, le.source_question_ids
        FROM longitudinal_encounters le
        ORDER BY le.patient_id, le.encounter_order
    """).fetchall()

    # Group encounters by patient
    enc_by_patient: dict[int, list[tuple]] = defaultdict(list)
    for r in enc_rows:
        enc_by_patient[r[0]].append(r)

    # Load correct diagnoses per question
    correct_dx = conn.execute("""
        SELECT qd.question_id, qd.diagnosis_id, d.icd10_code,
               d.display_name, d.acuity, d.snomed_id
        FROM question_diagnoses qd
        JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
        WHERE qd.role = 'correct'
    """).fetchall()
    dx_by_qid: dict[int, list[tuple]] = defaultdict(list)
    for r in correct_dx:
        dx_by_qid[r[0]].append(r)

    # Load organ systems per question for difficulty
    os_by_qid: dict[int, str | None] = {}
    for qid, os in conn.execute(
        "SELECT question_id, organ_system FROM board_questions"
    ).fetchall():
        os_by_qid[qid] = os

    patient_ids = sorted(enc_by_patient.keys())
    if pilot:
        patient_ids = patient_ids[:pilot]

    conn.execute(
        "DELETE FROM benchmark_ground_truth WHERE task = 'diagnosis_accuracy' AND granularity = 'patient'"
    )

    inserted = 0
    for pid in patient_ids:
        encounters = enc_by_patient[pid]
        active_diagnoses = []
        chronic_conditions = []
        encounter_dx_map: dict[str, list[dict]] = {}
        seen_dx: dict[int, dict] = {}  # dx_id -> first occurrence info
        organ_systems: set[str] = set()
        has_chronic = False
        has_acute = False

        for _, enc_id, enc_date, enc_order, src_qids_json in encounters:
            try:
                src_qids = json.loads(src_qids_json) if src_qids_json else []
            except (json.JSONDecodeError, TypeError):
                src_qids = []

            enc_dx_list = []
            for qid in src_qids:
                qid = int(qid)
                os_val = os_by_qid.get(qid)
                if os_val:
                    organ_systems.add(os_val)

                for _, dx_id, icd10, name, acuity, snomed in dx_by_qid.get(qid, []):
                    enc_dx_list.append({"diagnosis_id": dx_id, "role": "correct"})
                    if dx_id not in seen_dx:
                        entry = {
                            "diagnosis_id": dx_id,
                            "icd10": icd10,
                            "snomed_id": snomed,
                            "display_name": name,
                            "first_encounter_id": enc_id,
                            "first_encounter_date": enc_date,
                            "acuity": acuity,
                        }
                        seen_dx[dx_id] = entry
                        if acuity in ("chronic", "acute_on_chronic"):
                            chronic_conditions.append(entry)
                            has_chronic = True
                        else:
                            active_diagnoses.append(entry)
                            has_acute = True

            if enc_dx_list:
                encounter_dx_map[str(enc_id)] = enc_dx_list

        gt = {
            "active_diagnoses": active_diagnoses,
            "chronic_conditions": chronic_conditions,
            "encounter_diagnosis_map": encounter_dx_map,
        }

        difficulty = _compute_difficulty_patient(
            len(encounters), len(seen_dx),
            has_chronic and has_acute, len(organ_systems),
        )

        conn.execute(
            "INSERT INTO benchmark_ground_truth "
            "(task, granularity, patient_id, ground_truth, difficulty, num_diagnoses) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("diagnosis_accuracy", "patient", pid,
             json.dumps(gt, ensure_ascii=False), difficulty,
             len(seen_dx)),
        )
        inserted += 1

    conn.commit()
    log.info(f"Sub-stage 10b complete: {inserted} patient-level dx accuracy entries")
    return {"inserted": inserted}


# ---------------------------------------------------------------------------
# Sub-stage 10c: Context Summarization — Patient-Level (LLM)
# ---------------------------------------------------------------------------

PATIENT_SUMMARY_PROMPT = """You are a board-certified physician creating a reference clinical summary for evaluating AI summarization systems on longitudinal patient records.

PATIENT PROFILE:
{patient_profile}

ENCOUNTER HISTORY ({num_encounters} encounters):
{encounters_block}

ALL DIAGNOSES ACROSS ENCOUNTERS:
{diagnoses_block}

INSTRUCTIONS:
Write a comprehensive clinical summary (5-10 sentences) that synthesizes this patient's entire longitudinal course. The summary must:
1. Open with patient demographics, chronic conditions, and relevant history
2. Describe the clinical trajectory chronologically
3. Highlight key diagnostic findings at each major encounter
4. Note how chronic conditions interact with acute presentations
5. End with the current clinical status and active problem list
6. Use standard medical documentation language

Return ONLY valid JSON (no markdown, no explanation):
{{"reference_summary": "...", "clinical_question": "What is the current active problem list and clinical trajectory for this patient?", "key_encounters": [<encounter_ids of the most clinically significant encounters>], "summary_word_count": <integer>}}"""


def _format_patient_summary_prompt(
    profile_json: str,
    encounters: list[dict],
    diagnoses: list[dict],
) -> str:
    enc_lines = []
    for enc in encounters:
        enc_lines.append(
            f"Encounter #{enc['encounter_order']} ({enc['encounter_date']}, "
            f"{enc['encounter_type']}, {enc['department']}):\n"
            f"  CC: {enc['chief_complaint']}\n"
            f"  Provider: {enc['attending_name']}\n"
            f"  Note:\n{enc['note_text']}\n"
        )
    dx_lines = []
    for dx in diagnoses:
        dx_lines.append(
            f"- {dx['display_name']} (ICD-10: {dx['icd10']}, "
            f"first seen: {dx['first_encounter_date']}, acuity: {dx['acuity']})"
        )
    return PATIENT_SUMMARY_PROMPT.format(
        patient_profile=profile_json,
        num_encounters=len(encounters),
        encounters_block="\n".join(enc_lines) if enc_lines else "(no encounters)",
        diagnoses_block="\n".join(dx_lines) if dx_lines else "(no diagnoses)",
    )


def _generate_summary_single(
    patient_id: int, prompt: str
) -> tuple[int, dict | None, str | None]:
    """Thread-safe LLM worker for patient summary. No DB access."""
    try:
        parsed, in_tok, out_tok, req_id = _call_with_retry(prompt)
        return patient_id, {
            "output_json": parsed,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "request_id": req_id,
        }, None
    except Exception as e:
        return patient_id, None, str(e)


def run_10c(
    conn: sqlite3.Connection,
    pilot: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Context Summarization ground truth: patient-level (LLM)."""
    log.info("Sub-stage 10c: Context Summarization (patient-level, LLM)...")

    # Phase 1: Load patient data
    patients = conn.execute(
        "SELECT patient_id, profile FROM longitudinal_patients ORDER BY patient_id"
    ).fetchall()

    enc_rows = conn.execute("""
        SELECT patient_id, encounter_id, encounter_date, encounter_type,
               encounter_order, chief_complaint, attending_name, department,
               note_text, source_question_ids
        FROM longitudinal_encounters
        ORDER BY patient_id, encounter_order
    """).fetchall()

    enc_by_patient: dict[int, list[dict]] = defaultdict(list)
    for r in enc_rows:
        enc_by_patient[r[0]].append({
            "encounter_id": r[1], "encounter_date": r[2],
            "encounter_type": r[3], "encounter_order": r[4],
            "chief_complaint": r[5], "attending_name": r[6],
            "department": r[7], "note_text": r[8] or "",
            "source_question_ids": r[9],
        })

    # Load correct diagnoses per question for must_include_findings
    correct_dx_rows = conn.execute("""
        SELECT qd.question_id, d.diagnosis_id, d.icd10_code,
               d.display_name, d.acuity
        FROM question_diagnoses qd
        JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
        WHERE qd.role = 'correct'
    """).fetchall()
    dx_by_qid: dict[int, list[dict]] = defaultdict(list)
    for qid, dx_id, icd10, name, acuity in correct_dx_rows:
        dx_by_qid[qid].append({
            "diagnosis_id": dx_id, "icd10": icd10,
            "display_name": name, "acuity": acuity,
        })

    # Load key findings per question
    key_findings_rows = conn.execute("""
        SELECT qf.question_id, cf.display_name, cf.finding_type
        FROM question_findings qf
        JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
        WHERE qf.relevance = 'key'
    """).fetchall()
    key_findings_by_qid: dict[int, list[dict]] = defaultdict(list)
    for qid, name, ftype in key_findings_rows:
        key_findings_by_qid[qid].append({"display_name": name, "finding_type": ftype})

    # Load organ systems per question for difficulty
    os_by_qid_10c: dict[int, str | None] = {}
    for qid, os_val in conn.execute(
        "SELECT question_id, organ_system FROM board_questions"
    ).fetchall():
        os_by_qid_10c[qid] = os_val

    patient_ids = [p[0] for p in patients]
    profiles = {p[0]: p[1] for p in patients}
    if pilot:
        patient_ids = patient_ids[:pilot]

    # Phase 2: Check cache
    to_generate = []
    cached_results: dict[int, dict] = {}
    for pid in patient_ids:
        ih = _compute_input_hash(STAGE_10C, pid)
        cached = _check_cache(conn, STAGE_10C, ih)
        if cached:
            cached_results[pid] = cached
        else:
            to_generate.append(pid)

    log.info(
        f"Context summarization: {len(cached_results)} cached, "
        f"{len(to_generate)} to generate ({workers} workers)"
    )

    # Phase 3: LLM generation
    ok_count = 0
    err_count = 0
    if to_generate:
        # Build prompts
        prompts: dict[int, str] = {}
        for pid in to_generate:
            encounters = enc_by_patient.get(pid, [])
            # Collect diagnoses across encounters
            all_dx: list[dict] = []
            seen_dx_ids: set[int] = set()
            for enc in encounters:
                try:
                    src_qids = json.loads(enc["source_question_ids"]) if enc["source_question_ids"] else []
                except (json.JSONDecodeError, TypeError):
                    src_qids = []
                for qid in src_qids:
                    for dx in dx_by_qid.get(int(qid), []):
                        if dx["diagnosis_id"] not in seen_dx_ids:
                            seen_dx_ids.add(dx["diagnosis_id"])
                            all_dx.append({**dx, "first_encounter_date": enc["encounter_date"]})

            prompts[pid] = _format_patient_summary_prompt(
                profiles.get(pid, "{}"), encounters, all_dx
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_generate_summary_single, pid, prompts[pid]): pid
                for pid in to_generate
            }
            done_count = 0
            for future in as_completed(futures):
                pid, result, error = future.result()
                ih = _compute_input_hash(STAGE_10C, pid)
                if error:
                    err_count += 1
                    conn.execute(
                        "INSERT INTO llm_call_log (stage, model, input_hash, error) "
                        "VALUES (?, ?, ?, ?)",
                        (STAGE_10C, MODEL, ih, error),
                    )
                else:
                    ok_count += 1
                    cached_results[pid] = result["output_json"]
                    conn.execute(
                        "INSERT INTO llm_call_log "
                        "(stage, model, input_hash, output_json, input_tokens, output_tokens, request_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (STAGE_10C, MODEL, ih,
                         json.dumps(result["output_json"], ensure_ascii=False),
                         result["input_tokens"], result["output_tokens"],
                         result["request_id"]),
                    )
                done_count += 1
                if done_count % 10 == 0:
                    conn.commit()
                if done_count % 50 == 0:
                    log.info(f"  Progress: {done_count}/{len(to_generate)}")

        conn.commit()

    # Phase 4: Insert ground truth
    conn.execute(
        "DELETE FROM benchmark_ground_truth "
        "WHERE task = 'context_summarization' AND granularity = 'patient'"
    )

    inserted = 0
    for pid in patient_ids:
        llm_result = cached_results.get(pid)
        if not llm_result:
            continue

        encounters = enc_by_patient.get(pid, [])
        # Collect must_include_findings from key findings
        must_include: list[dict] = []
        seen_finding_names: set[str] = set()
        for enc in encounters:
            try:
                src_qids = json.loads(enc["source_question_ids"]) if enc["source_question_ids"] else []
            except (json.JSONDecodeError, TypeError):
                src_qids = []
            for qid in src_qids:
                for f in key_findings_by_qid.get(int(qid), []):
                    if f["display_name"] not in seen_finding_names:
                        seen_finding_names.add(f["display_name"])
                        must_include.append(f)

        gt = {
            "reference_summary": llm_result.get("reference_summary", ""),
            "clinical_question": llm_result.get(
                "clinical_question",
                "What is the current active problem list and clinical trajectory?",
            ),
            "key_encounters": llm_result.get("key_encounters", []),
            "must_include_findings": must_include[:20],  # cap for sanity
        }

        # Difficulty (reuse patient difficulty from 10b data)
        all_dx_ids: set[int] = set()
        organ_systems: set[str] = set()
        has_chronic = False
        has_acute = False
        for enc in encounters:
            try:
                src_qids = json.loads(enc["source_question_ids"]) if enc["source_question_ids"] else []
            except (json.JSONDecodeError, TypeError):
                src_qids = []
            for qid in src_qids:
                qid = int(qid)
                os_val = os_by_qid_10c.get(qid)
                if os_val:
                    organ_systems.add(os_val)
                for dx in dx_by_qid.get(qid, []):
                    all_dx_ids.add(dx["diagnosis_id"])
                    if dx["acuity"] in ("chronic", "acute_on_chronic"):
                        has_chronic = True
                    else:
                        has_acute = True

        difficulty = _compute_difficulty_patient(
            len(encounters), len(all_dx_ids),
            has_chronic and has_acute, len(organ_systems),
        )

        conn.execute(
            "INSERT INTO benchmark_ground_truth "
            "(task, granularity, patient_id, ground_truth, difficulty) "
            "VALUES (?, ?, ?, ?, ?)",
            ("context_summarization", "patient", pid,
             json.dumps(gt, ensure_ascii=False), difficulty),
        )
        inserted += 1

    conn.commit()
    log.info(
        f"Sub-stage 10c complete: {inserted} patient summaries "
        f"({ok_count} generated, {len(cached_results) - ok_count} cached, {err_count} errors)"
    )
    return {"inserted": inserted, "ok": ok_count, "errors": err_count,
            "cached": len(cached_results) - ok_count}


# ---------------------------------------------------------------------------
# Sub-stage 10c-visit: Context Summarization — Current-Visit Variant (LLM)
#
# Point-in-time summary anchored on an index encounter (the most recent visit).
# Conditioning is "free" from the data (encounter position + department): no
# specialty taxonomy and no relatedness graph are needed (spec_v1.33 §17.3/§17.7).
# Deltas are scoped to new/resolved (set operations on per-encounter key-finding
# sets); "changed" is deferred. The eval loader truncates the model input to
# encounters <= the index encounter (no-future-leakage control, §17.7).
# ---------------------------------------------------------------------------

CURRENT_VISIT_SUMMARY_PROMPT = """You are a board-certified physician writing a concise point-in-time summary for the CURRENT encounter of a longitudinal patient record, used to evaluate AI current-visit summarization.

PATIENT PROFILE:
{patient_profile}

PRIOR ENCOUNTERS (chronological, context only):
{prior_block}

CURRENT (INDEX) ENCOUNTER — {index_date} ({index_dept}); chief complaint: {chief_complaint}:
{index_note}

INTERVAL CHANGES SINCE THE PRIOR ENCOUNTER:
  New findings: {new_findings}
  Resolved/absent findings: {resolved_findings}

INSTRUCTIONS:
Write a focused current-visit summary that captures ALL clinically relevant information for this visit. Do not impose an arbitrary length — be as thorough as the clinical picture requires, but stay focused on this encounter and OMIT unrelated, normal, or incidental detail. The summary should:
1. State the reason for the current visit (the chief complaint / presenting problem).
2. Describe what has changed since the prior encounter (new and resolved findings).
3. List the active problems being managed at this visit.
4. Use ONLY information available up to and including the current encounter — do NOT reference or infer anything from future encounters.
Use standard medical documentation language.

Return ONLY valid JSON (no markdown, no explanation):
{{"reference_summary": "...", "summary_word_count": <integer>}}"""


def _parse_qids(raw: str | None) -> list[int]:
    try:
        v = json.loads(raw) if raw else []
        return [int(q) for q in v] if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _encounter_key_findings(
    enc: dict, key_findings_by_qid: dict[int, list[dict]]
) -> dict[str, dict]:
    """Key findings present at an encounter, keyed by casefolded display name."""
    out: dict[str, dict] = {}
    for qid in _parse_qids(enc.get("source_question_ids")):
        for f in key_findings_by_qid.get(qid, []):
            k = f["display_name"].casefold()
            out.setdefault(k, {
                "display_name": f["display_name"],
                "finding_type": f["finding_type"],
            })
    return out


def _select_index_positions(n: int, max_per_patient: int = CURRENT_VISIT_INDEX_POINTS) -> list[int]:
    """Up to `max_per_patient` index positions (0-based) to anchor current-visit
    summaries on. Only positions with >=1 prior encounter are eligible (1..n-1).
    When more than max are eligible, sample early / mid / late (the last is the
    terminal encounter = the no-future deployment case)."""
    eligible = list(range(1, n))  # positions 1..n-1 each have >=1 prior
    if len(eligible) <= max_per_patient:
        return eligible
    picks = {eligible[0], eligible[len(eligible) // 2], eligible[-1]}
    return sorted(picks)


def _build_current_visit_core(
    encs: list[dict],
    k: int,
    key_findings_by_qid: dict[int, list[dict]],
    off_target_by_qid: dict[int, list[dict]] | None = None,
) -> dict | None:
    """Deterministic GT core for one current-visit anchor.

    `k` is the 0-based index position (the "current visit"); encounters AFTER k
    are HELD OUT (no-future-leakage). Deltas: new = first-time at k vs all priors;
    resolved = present at k-1 but absent at k. `off_target_by_qid` -> the index
    encounter's background/distractor findings (irrelevant-inclusion precision
    set). `future_findings` = key findings that appear ONLY in the held-out
    future (temporal-leakage set the summary must not assert). Returns None if k
    has no prior encounter (spec_v1.33 §17.7/§17.9).
    """
    encs = sorted(encs, key=lambda e: e["encounter_order"])
    n = len(encs)
    if k < 1 or k >= n:
        return None
    index_enc = encs[k]
    prev_enc = encs[k - 1]
    prior_encs = encs[:k]
    future_encs = encs[k + 1:]

    index_f = _encounter_key_findings(index_enc, key_findings_by_qid)
    prev_f = _encounter_key_findings(prev_enc, key_findings_by_qid)
    prior_union: dict[str, dict] = {}
    for e in prior_encs:
        prior_union.update(_encounter_key_findings(e, key_findings_by_qid))

    new = [v for kk, v in index_f.items() if kk not in prior_union]
    resolved = [v for kk, v in prev_f.items() if kk not in index_f]
    must_include = list(index_f.values())[:CURRENT_VISIT_MAX_FINDINGS]

    # Future-only findings: key findings appearing only in the held-out future
    # (not present at or before the index). The summary must NOT assert these.
    seen_le_index = set(prior_union) | set(index_f)
    future_f: dict[str, dict] = {}
    for e in future_encs:
        for kk, v in _encounter_key_findings(e, key_findings_by_qid).items():
            if kk not in seen_le_index:
                future_f.setdefault(kk, v)

    # Off-target (precision) set: background/distractor findings of the index
    # encounter that a focused summary should NOT foreground. Excludes anything
    # key at this visit and excludes demographics (conventionally included).
    off_target: dict[str, dict] = {}
    if off_target_by_qid:
        for kk, v in _encounter_key_findings(index_enc, off_target_by_qid).items():
            if kk not in index_f and v.get("finding_type") != "demographic":
                off_target[kk] = v

    cc = (index_enc.get("chief_complaint") or "").strip() or "this encounter"
    clinical_question = (
        f"Summarize this patient's current encounter on {index_enc['encounter_date']} "
        f"(chief complaint: {cc}): the reason for the visit, what has changed since "
        f"the prior encounter, and the current active problems."
    )

    return {
        "variant": "current_visit",
        "clinical_question": clinical_question,
        "must_include_findings": must_include,
        "deltas": {
            "new": new[:CURRENT_VISIT_MAX_FINDINGS],
            "resolved": resolved[:CURRENT_VISIT_MAX_FINDINGS],
        },
        "off_target_findings": list(off_target.values())[:CURRENT_VISIT_MAX_FINDINGS],
        "future_findings": list(future_f.values())[:CURRENT_VISIT_MAX_FINDINGS],
        "index_encounter_id": index_enc["encounter_id"],
        "index_encounter_order": index_enc["encounter_order"],
        "index_position": k,
        "n_encounters": n,
        "has_future": k < n - 1,
        "_index_enc": index_enc,
        "_prior_encs": prior_encs,
    }


def _format_current_visit_prompt(profile_json: str, core: dict) -> str:
    index_enc = core["_index_enc"]
    prior_lines = []
    for e in core["_prior_encs"]:
        note = (e.get("note_text") or "").strip()
        cap = CURRENT_VISIT_PRIOR_NOTE_CHARS
        if cap is not None and len(note) > cap:
            note = note[:cap] + " …"
        dept = e.get("department") or e.get("encounter_type") or "encounter"
        prior_lines.append(
            f"- #{e['encounter_order']} {e['encounter_date']} ({dept}); "
            f"CC: {e.get('chief_complaint') or 'n/a'}\n  {note}"
        )
    new_str = ", ".join(f["display_name"] for f in core["deltas"]["new"]) or "(none identified)"
    res_str = ", ".join(f["display_name"] for f in core["deltas"]["resolved"]) or "(none identified)"
    return CURRENT_VISIT_SUMMARY_PROMPT.format(
        patient_profile=profile_json or "{}",
        prior_block="\n".join(prior_lines) if prior_lines else "(none)",
        index_date=index_enc["encounter_date"],
        index_dept=index_enc.get("department") or index_enc.get("encounter_type") or "n/a",
        chief_complaint=index_enc.get("chief_complaint") or "n/a",
        index_note=(index_enc.get("note_text") or "").strip() or "(no note text)",
        new_findings=new_str,
        resolved_findings=res_str,
    )


def _compute_visit_input_hash(enc_id: int, prompt: str) -> str:
    """Content-sensitive cache key: changing the prompt (e.g. context cap)
    invalidates the cached reference so it regenerates."""
    return hashlib.sha256(
        f"{STAGE_10C_VISIT}:{enc_id}:{prompt}".encode()
    ).hexdigest()


def _generate_visit_summary_single(
    enc_id: int, prompt: str
) -> tuple[int, dict | None, str | None]:
    """Thread-safe LLM worker for current-visit reference summary. No DB access."""
    try:
        parsed, in_tok, out_tok, req_id = _call_with_retry(prompt)
        return enc_id, {
            "output_json": parsed,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "request_id": req_id,
        }, None
    except Exception as e:
        return enc_id, None, str(e)


def run_10c_visit(
    conn: sqlite3.Connection,
    pilot: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Current-visit Context Summarization ground truth (encounter-level, LLM)."""
    log.info("Sub-stage 10c-visit: Current-Visit Summarization (encounter-level, LLM)...")

    # Phase 1: Load patients, encounters, key findings
    patients = conn.execute(
        "SELECT patient_id, profile FROM longitudinal_patients ORDER BY patient_id"
    ).fetchall()
    profiles = {p[0]: p[1] for p in patients}
    patient_ids = [p[0] for p in patients]
    if pilot:
        patient_ids = patient_ids[:pilot]

    enc_rows = conn.execute("""
        SELECT patient_id, encounter_id, encounter_date, encounter_type,
               encounter_order, chief_complaint, department, note_text,
               source_question_ids
        FROM longitudinal_encounters
        ORDER BY patient_id, encounter_order
    """).fetchall()
    enc_by_patient: dict[int, list[dict]] = defaultdict(list)
    for r in enc_rows:
        enc_by_patient[r[0]].append({
            "encounter_id": r[1], "encounter_date": r[2],
            "encounter_type": r[3], "encounter_order": r[4],
            "chief_complaint": r[5], "department": r[6],
            "note_text": r[7] or "", "source_question_ids": r[8],
        })

    key_findings_by_qid: dict[int, list[dict]] = defaultdict(list)
    off_target_by_qid: dict[int, list[dict]] = defaultdict(list)
    for qid, name, ftype, rel in conn.execute("""
        SELECT qf.question_id, cf.display_name, cf.finding_type, qf.relevance
        FROM question_findings qf
        JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
        WHERE qf.relevance IN ('key', 'background', 'distractor')
    """).fetchall():
        entry = {"display_name": name, "finding_type": ftype}
        if rel == "key":
            key_findings_by_qid[qid].append(entry)
        else:  # background / distractor -> off-target (precision) set
            off_target_by_qid[qid].append(entry)

    # Phase 2: Build deterministic cores — up to CURRENT_VISIT_INDEX_POINTS index
    # anchors per patient (each with its future held out), keyed by index enc_id.
    cores: dict[int, dict] = {}      # index_encounter_id -> core
    core_pid: dict[int, int] = {}    # index_encounter_id -> patient_id
    for pid in patient_ids:
        encs = enc_by_patient.get(pid, [])
        for k in _select_index_positions(len(encs)):
            core = _build_current_visit_core(encs, k, key_findings_by_qid, off_target_by_qid)
            if core:
                eid = core["index_encounter_id"]
                cores[eid] = core
                core_pid[eid] = pid
    n_patients = len(set(core_pid.values()))
    log.info("Current-visit: %d anchors across %d patients (<=%d index pts/patient)",
             len(cores), n_patients, CURRENT_VISIT_INDEX_POINTS)

    # Phase 3: LLM reference summaries. Build all prompts first (cheap, no LLM)
    # so the cache key can be content-sensitive — changing the prompt (e.g. the
    # prior-note context cap) invalidates cached references and regenerates them.
    prompts: dict[int, str] = {}
    ih_by_enc: dict[int, str] = {}
    for enc_id, core in cores.items():
        p = _format_current_visit_prompt(profiles.get(core_pid[enc_id], "{}"), core)
        prompts[enc_id] = p
        ih_by_enc[enc_id] = _compute_visit_input_hash(enc_id, p)

    to_generate: list[int] = []   # enc_ids
    ref_by_enc: dict[int, dict] = {}
    for enc_id, ih in ih_by_enc.items():
        cached = _check_cache(conn, STAGE_10C_VISIT, ih)
        if cached:
            ref_by_enc[enc_id] = cached
        else:
            to_generate.append(enc_id)

    log.info("Current-visit references: %d cached, %d to generate (%d workers)",
             len(ref_by_enc), len(to_generate), workers)

    ok_count = err_count = 0
    if to_generate:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_generate_visit_summary_single, eid, prompts[eid]): eid
                for eid in to_generate
            }
            done = 0
            for future in as_completed(futures):
                enc_id, result, error = future.result()
                ih = ih_by_enc[enc_id]
                if error:
                    err_count += 1
                    conn.execute(
                        "INSERT INTO llm_call_log (stage, model, input_hash, error) "
                        "VALUES (?, ?, ?, ?)",
                        (STAGE_10C_VISIT, MODEL, ih, error),
                    )
                else:
                    ok_count += 1
                    ref_by_enc[enc_id] = result["output_json"]
                    conn.execute(
                        "INSERT INTO llm_call_log "
                        "(stage, model, input_hash, output_json, input_tokens, output_tokens, request_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (STAGE_10C_VISIT, MODEL, ih,
                         json.dumps(result["output_json"], ensure_ascii=False),
                         result["input_tokens"], result["output_tokens"],
                         result["request_id"]),
                    )
                done += 1
                if done % 10 == 0:
                    conn.commit()
                if done % 50 == 0:
                    log.info("  Progress: %d/%d", done, len(to_generate))
        conn.commit()

    # Phase 4: Insert ground truth (encounter granularity)
    conn.execute(
        "DELETE FROM benchmark_ground_truth "
        "WHERE task = 'context_summarization' AND granularity = 'encounter'"
    )
    inserted = 0
    for enc_id, core in cores.items():
        pid = core_pid[enc_id]
        ref = ref_by_enc.get(enc_id, {})
        gt = {
            "variant": "current_visit",
            "clinical_question": core["clinical_question"],
            "reference_summary": ref.get("reference_summary", ""),
            "must_include_findings": core["must_include_findings"],
            "deltas": core["deltas"],
            "off_target_findings": core["off_target_findings"],
            "future_findings": core["future_findings"],
            "index_encounter_id": enc_id,
            "index_encounter_order": core["index_encounter_order"],
            "index_position": core["index_position"],
            "n_encounters": core["n_encounters"],
            "has_future": core["has_future"],
        }
        # Difficulty scales with how much chart is visible at this anchor
        # (encounters up to and including the index = index_position + 1).
        difficulty = _compute_difficulty_patient(
            core["index_position"] + 1,
            len(core["must_include_findings"]),
            bool(core["deltas"]["new"]) and bool(core["deltas"]["resolved"]),
            1,
        )
        conn.execute(
            "INSERT INTO benchmark_ground_truth "
            "(task, granularity, encounter_id, patient_id, ground_truth, difficulty) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("context_summarization", "encounter", enc_id, pid,
             json.dumps(gt, ensure_ascii=False), difficulty),
        )
        inserted += 1

    conn.commit()
    log.info(
        "Sub-stage 10c-visit complete: %d current-visit GT rows "
        "(%d generated, %d cached, %d errors)",
        inserted, ok_count, len(ref_by_enc) - ok_count, err_count,
    )
    return {"inserted": inserted, "ok": ok_count, "errors": err_count,
            "cached": len(ref_by_enc) - ok_count}


# ---------------------------------------------------------------------------
# Sub-stage 10d: Evidence Retrieval — Patient-Level (Deterministic)
# ---------------------------------------------------------------------------

def run_10d(conn: sqlite3.Connection, pilot: int | None = None) -> dict:
    """Evidence Retrieval ground truth: patient-level (deterministic)."""
    log.info("Sub-stage 10d: Evidence Retrieval (patient-level)...")

    # Load encounter data
    enc_rows = conn.execute("""
        SELECT le.patient_id, le.encounter_id, le.source_question_ids
        FROM longitudinal_encounters le
        ORDER BY le.patient_id, le.encounter_order
    """).fetchall()
    enc_by_patient: dict[int, list[tuple]] = defaultdict(list)
    for pid, eid, sqids in enc_rows:
        enc_by_patient[pid].append((eid, sqids))

    # Load correct diagnoses per question
    correct_dx = conn.execute("""
        SELECT qd.question_id, qd.diagnosis_id
        FROM question_diagnoses qd WHERE qd.role = 'correct'
    """).fetchall()
    correct_dx_by_qid: dict[int, set[int]] = defaultdict(set)
    for qid, dx_id in correct_dx:
        correct_dx_by_qid[qid].add(dx_id)

    # Load question_findings with relevance
    qf_rows = conn.execute("""
        SELECT qf.question_id, qf.finding_id, qf.relevance, qf.ehr_section,
               cf.finding_type
        FROM question_findings qf
        JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
    """).fetchall()
    findings_by_qid: dict[int, list[tuple]] = defaultdict(list)
    for qid, fid, rel, ehr_sec, ftype in qf_rows:
        findings_by_qid[qid].append((fid, rel, ehr_sec, ftype))

    # Load diagnosis_findings relationships
    df_rows = conn.execute("""
        SELECT diagnosis_id, finding_id, relationship
        FROM diagnosis_findings
    """).fetchall()
    df_lookup: dict[tuple[int, int], str] = {}
    for dx_id, fid, rel in df_rows:
        df_lookup[(dx_id, fid)] = rel

    # Load fact_diagnosis_links
    fdl_rows = conn.execute("""
        SELECT diagnosis_id, fact_id, relevance
        FROM fact_diagnosis_links
    """).fetchall()
    facts_by_dx: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for dx_id, fid, rel in fdl_rows:
        facts_by_dx[dx_id].append((fid, rel))

    # Load encounter_ehr_sections
    ees_rows = conn.execute("""
        SELECT id, encounter_id, section_type
        FROM encounter_ehr_sections
    """).fetchall()
    ees_by_enc: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for ees_id, enc_id, stype in ees_rows:
        ees_by_enc[enc_id].append((ees_id, stype))

    # Load organ systems + acuity per question for difficulty
    os_by_qid: dict[int, str | None] = {}
    for qid, os_val in conn.execute(
        "SELECT question_id, organ_system FROM board_questions"
    ).fetchall():
        os_by_qid[qid] = os_val

    acuity_by_dx: dict[int, str | None] = {}
    for dx_id, acuity in conn.execute(
        "SELECT diagnosis_id, acuity FROM diagnoses"
    ).fetchall():
        acuity_by_dx[dx_id] = acuity

    patient_ids = sorted(enc_by_patient.keys())
    if pilot:
        patient_ids = patient_ids[:pilot]

    # Delete existing
    conn.execute(
        "DELETE FROM relevance_judgments WHERE gt_id IN "
        "(SELECT gt_id FROM benchmark_ground_truth WHERE task = 'evidence_retrieval')"
    )
    conn.execute(
        "DELETE FROM benchmark_ground_truth WHERE task = 'evidence_retrieval'"
    )

    gt_inserted = 0
    rj_inserted = 0

    for pid in patient_ids:
        encounters = enc_by_patient[pid]

        # Collect correct dx ids for this patient + difficulty metadata
        patient_correct_dx: set[int] = set()
        organ_systems: set[str] = set()
        has_chronic = False
        has_acute = False
        for enc_id, sqids_json in encounters:
            try:
                src_qids = json.loads(sqids_json) if sqids_json else []
            except (json.JSONDecodeError, TypeError):
                src_qids = []
            for qid in src_qids:
                qid_int = int(qid)
                patient_correct_dx.update(correct_dx_by_qid.get(qid_int, set()))
                os_val = os_by_qid.get(qid_int)
                if os_val:
                    organ_systems.add(os_val)
        for dx_id in patient_correct_dx:
            acuity = acuity_by_dx.get(dx_id)
            if acuity in ("chronic", "acute_on_chronic"):
                has_chronic = True
            else:
                has_acute = True

        if not patient_correct_dx:
            continue

        judgments: list[dict] = []

        # Grade encounter_ehr_sections
        for enc_id, sqids_json in encounters:
            try:
                src_qids = json.loads(sqids_json) if sqids_json else []
            except (json.JSONDecodeError, TypeError):
                src_qids = []

            # Collect findings for this encounter's source questions
            enc_findings: dict[int, tuple[str, str | None, str]] = {}  # fid -> (relevance, ehr_section, type)
            for qid in src_qids:
                for fid, rel, ehr_sec, ftype in findings_by_qid.get(int(qid), []):
                    if fid not in enc_findings or _rel_priority(rel) > _rel_priority(enc_findings[fid][0]):
                        enc_findings[fid] = (rel, ehr_sec, ftype)

            for ees_id, stype in ees_by_enc.get(enc_id, []):
                if stype in ("assessment", "plan"):
                    continue  # Skip — may contain answer

                grade = _grade_section(
                    stype, enc_findings, patient_correct_dx, df_lookup
                )
                if grade >= 0:
                    judgments.append({
                        "passage_id": f"ees_{ees_id}",
                        "passage_source": "encounter_section",
                        "relevance_grade": grade,
                        "rationale": f"section_type={stype}",
                        "source": "rule_based",
                    })

        # Grade fact cards linked to patient's correct diagnoses
        seen_facts: set[int] = set()
        for dx_id in patient_correct_dx:
            for fact_id, rel in facts_by_dx.get(dx_id, []):
                if fact_id in seen_facts:
                    continue
                seen_facts.add(fact_id)
                if rel in ("defines", "mechanism"):
                    grade = 3
                elif rel in ("treatment", "differentiates"):
                    grade = 2
                elif rel == "epidemiology":
                    grade = 1
                else:
                    grade = 1
                judgments.append({
                    "passage_id": f"fc_{fact_id}",
                    "passage_source": "fact_card",
                    "relevance_grade": grade,
                    "rationale": f"fdl_relevance={rel}",
                    "source": "rule_based",
                })

        # Build ground truth JSON
        gt = {
            "query_diagnoses": [
                {"diagnosis_id": dx_id} for dx_id in sorted(patient_correct_dx)
            ],
            "num_passages": len(judgments),
            "grade_distribution": {
                str(g): sum(1 for j in judgments if j["relevance_grade"] == g)
                for g in range(4)
            },
        }

        difficulty = _compute_difficulty_patient(
            len(encounters), len(patient_correct_dx),
            has_chronic and has_acute, len(organ_systems),
        )

        # Insert gt row
        conn.execute(
            "INSERT INTO benchmark_ground_truth "
            "(task, granularity, patient_id, ground_truth, difficulty, num_evidence, num_diagnoses) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("evidence_retrieval", "patient", pid,
             json.dumps(gt, ensure_ascii=False), difficulty,
             len(judgments), len(patient_correct_dx)),
        )
        gt_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        gt_inserted += 1

        # Insert relevance judgments
        for j in judgments:
            conn.execute(
                "INSERT INTO relevance_judgments "
                "(gt_id, passage_id, passage_source, relevance_grade, rationale, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (gt_id, j["passage_id"], j["passage_source"],
                 j["relevance_grade"], j["rationale"], j["source"]),
            )
            rj_inserted += 1

        if gt_inserted % 100 == 0:
            conn.commit()
            if gt_inserted % 200 == 0:
                log.info(f"  Progress: {gt_inserted} patients, {rj_inserted} judgments")

    conn.commit()
    log.info(
        f"Sub-stage 10d complete: {gt_inserted} patient evidence gt, "
        f"{rj_inserted} relevance judgments"
    )
    return {"gt_inserted": gt_inserted, "rj_inserted": rj_inserted}


def _rel_priority(relevance: str | None) -> int:
    """Higher = more important relevance."""
    return {"key": 3, "supporting": 2, "background": 1, "distractor": 0}.get(relevance or "", -1)


def _grade_section(
    section_type: str,
    enc_findings: dict[int, tuple[str, str | None, str]],
    correct_dx_ids: set[int],
    df_lookup: dict[tuple[int, int], str],
) -> int:
    """Grade a single EHR section based on its findings. Returns 0-3."""
    # Find findings that match this section type
    matching_types = SECTION_FINDING_TYPE_MAP.get(section_type, set())
    best_grade = 0

    for fid, (relevance, ehr_sec, ftype) in enc_findings.items():
        # Check if finding belongs in this section
        in_section = False
        if ehr_sec and ehr_sec == section_type:
            in_section = True
        elif ftype in matching_types:
            in_section = True

        if not in_section:
            continue

        # Check relationship to correct diagnoses
        best_dx_rel = None
        for dx_id in correct_dx_ids:
            dx_rel = df_lookup.get((dx_id, fid))
            if dx_rel:
                if best_dx_rel is None or _dx_rel_priority(dx_rel) > _dx_rel_priority(best_dx_rel):
                    best_dx_rel = dx_rel

        if relevance == "key" and best_dx_rel in ("pathognomonic", "highly_suggestive"):
            best_grade = max(best_grade, 3)
        elif relevance == "key":
            best_grade = max(best_grade, 2)
        elif relevance == "supporting" and best_dx_rel == "commonly_seen":
            best_grade = max(best_grade, 2)
        elif relevance == "supporting":
            best_grade = max(best_grade, 1)
        elif relevance == "background":
            best_grade = max(best_grade, 1)

    return best_grade


def _dx_rel_priority(relationship: str) -> int:
    return {
        "pathognomonic": 5, "highly_suggestive": 4, "commonly_seen": 3,
        "risk_factor": 2, "rules_out": 1, "protective": 0,
    }.get(relationship, 0)


# ---------------------------------------------------------------------------
# Sub-stage 10e: Imaging Clinical Indication — Encounter-Level (LLM)
# ---------------------------------------------------------------------------

IMAGING_INDICATION_PROMPT = """You are a radiologist building training data for an AI system that must interpret vague imaging order indications in the context of a patient's EHR.

PATIENT PROFILE SUMMARY:
{patient_summary}

ENCOUNTER CONTEXT:
- Date: {encounter_date}
- Type: {encounter_type}
- Department: {department}
- Chief Complaint: {chief_complaint}
- Attending: {attending_name}

EHR SECTIONS FOR THIS ENCOUNTER:
{ehr_context_block}

IMAGING SECTION (actual findings from the study):
{imaging_section_text}

CORRECT DIAGNOSIS FOR THIS ENCOUNTER: {correct_diagnosis}

PRIOR ENCOUNTERS:
{prior_encounters_block}

INSTRUCTIONS:
1. Determine the most likely imaging study ordered (modality + body region)
2. Generate a REALISTICALLY VAGUE clinical indication — the kind a busy clinician actually writes:
   - Examples: "eval abd pain", "r/o PE", "f/u lung nodule", "SOB", "trauma", "cp eval", "hx CA r/o mets"
   - Should be 2-8 words, abbreviated, clinically plausible
   - Must NOT contain the actual diagnosis — only the presenting symptom or concern
3. Generate the ground truth that an AI pre-read assistant should infer:
   - inferred_clinical_question: The actual clinical question the ordering provider likely had
   - pre_read_summary: A 3-5 sentence summary an AI would provide to the radiologist before reading, synthesizing relevant clinical history, labs, medications, and prior imaging
   - must_include_findings: Key imaging findings from the encounter that support the diagnosis
   - differential_context: 2-4 diagnoses to consider given the vague indication and chart context
   - relevant_clinical_data: Important non-imaging data the radiologist should know

Return ONLY valid JSON (no markdown, no explanation):
{{"imaging_order": {{"modality": "<xray|ct|ct_angio|mri|ultrasound|nuclear|fluoroscopy|mammography|pet_ct|other>", "body_region": "<e.g., chest, abdomen_pelvis, head, spine_cervical, extremity_lower>", "clinical_indication": "<vague 2-8 word indication>", "order_priority": "<routine|stat|urgent>"}}, "ground_truth": {{"inferred_clinical_question": "...", "pre_read_summary": "...", "must_include_findings": ["finding1", "finding2"], "differential_context": [{{"diagnosis": "...", "icd10": "...", "likelihood": "high|moderate|low"}}], "relevant_clinical_data": [{{"type": "lab|vital|history|medication", "detail": "..."}}]}}}}"""


def _format_imaging_prompt(
    patient_summary: str,
    encounter: dict,
    ehr_sections: list[dict],
    imaging_text: str,
    correct_dx: str,
    prior_encounters: list[dict],
) -> str:
    ehr_lines = []
    for sec in ehr_sections:
        if sec["section_type"] != "imaging":
            ehr_lines.append(f"[{sec['section_type'].upper()}]\n{sec['section_text']}\n")
    prior_lines = []
    for pe in prior_encounters[-5:]:  # Last 5 prior encounters
        prior_lines.append(
            f"- {pe['encounter_date']} ({pe['encounter_type']}): "
            f"CC: {pe['chief_complaint']}"
        )
    return IMAGING_INDICATION_PROMPT.format(
        patient_summary=patient_summary,
        encounter_date=encounter.get("encounter_date", ""),
        encounter_type=encounter.get("encounter_type", ""),
        department=encounter.get("department", ""),
        chief_complaint=encounter.get("chief_complaint", ""),
        attending_name=encounter.get("attending_name", ""),
        ehr_context_block="\n".join(ehr_lines) if ehr_lines else "(no non-imaging sections)",
        imaging_section_text=imaging_text,
        correct_diagnosis=correct_dx,
        prior_encounters_block="\n".join(prior_lines) if prior_lines else "(first encounter)",
    )


def _generate_imaging_single(
    encounter_id: int, patient_id: int, prompt: str
) -> tuple[int, int, dict | None, str | None]:
    """Thread-safe LLM worker for imaging indication. No DB access."""
    try:
        parsed, in_tok, out_tok, req_id = _call_with_retry(prompt)
        return encounter_id, patient_id, {
            "output_json": parsed,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "request_id": req_id,
        }, None
    except Exception as e:
        return encounter_id, patient_id, None, str(e)


def _validate_imaging_response(response: dict) -> list[str]:
    """Validate imaging LLM response. Returns list of warnings."""
    warnings = []
    order = response.get("imaging_order", {})
    gt = response.get("ground_truth", {})

    if not order.get("modality"):
        warnings.append("missing modality")
    if not order.get("body_region"):
        warnings.append("missing body_region")
    if not order.get("clinical_indication"):
        warnings.append("missing clinical_indication")
    indication = order.get("clinical_indication", "")
    if len(indication.split()) > 10:
        warnings.append(f"indication too long: {len(indication.split())} words")
    if not gt.get("inferred_clinical_question"):
        warnings.append("missing inferred_clinical_question")
    if not gt.get("pre_read_summary"):
        warnings.append("missing pre_read_summary")

    return warnings


def run_10e(
    conn: sqlite3.Connection,
    pilot: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Imaging Clinical Indication ground truth: encounter-level (LLM)."""
    log.info("Sub-stage 10e: Imaging Clinical Indication (encounter-level, LLM)...")

    # Phase 1: Find encounters with imaging sections
    imaging_encounters = conn.execute("""
        SELECT DISTINCT ees.encounter_id, le.patient_id,
               le.encounter_date, le.encounter_type, le.department,
               le.chief_complaint, le.attending_name, le.encounter_order,
               le.source_question_ids
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON ees.encounter_id = le.encounter_id
        WHERE ees.section_type = 'imaging'
        ORDER BY le.patient_id, le.encounter_order
    """).fetchall()

    if pilot:
        # Limit to encounters from first N patients
        seen_patients: set[int] = set()
        limited = []
        for r in imaging_encounters:
            seen_patients.add(r[1])
            if len(seen_patients) <= pilot:
                limited.append(r)
        imaging_encounters = limited

    log.info(f"  Found {len(imaging_encounters)} encounters with imaging sections")

    # Load imaging section text per encounter
    imaging_text_by_enc: dict[int, str] = {}
    for enc_id, text in conn.execute(
        "SELECT encounter_id, section_text FROM encounter_ehr_sections "
        "WHERE section_type = 'imaging'"
    ).fetchall():
        imaging_text_by_enc[enc_id] = text

    # Load all EHR sections per encounter (for context)
    all_ees = conn.execute(
        "SELECT encounter_id, section_type, section_text FROM encounter_ehr_sections"
    ).fetchall()
    ees_by_enc: dict[int, list[dict]] = defaultdict(list)
    for enc_id, stype, text in all_ees:
        ees_by_enc[enc_id].append({"section_type": stype, "section_text": text})

    # Load patient profiles
    profiles: dict[int, str] = {}
    for pid, prof in conn.execute(
        "SELECT patient_id, profile FROM longitudinal_patients"
    ).fetchall():
        profiles[pid] = prof or "{}"

    # Load correct diagnoses per question
    correct_dx_rows = conn.execute("""
        SELECT qd.question_id, d.display_name
        FROM question_diagnoses qd
        JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
        WHERE qd.role = 'correct'
    """).fetchall()
    dx_name_by_qid: dict[int, str] = {}
    for qid, name in correct_dx_rows:
        dx_name_by_qid[qid] = name

    # Load encounter data by patient for prior encounters
    enc_data = conn.execute("""
        SELECT patient_id, encounter_id, encounter_date, encounter_type,
               encounter_order, chief_complaint
        FROM longitudinal_encounters
        ORDER BY patient_id, encounter_order
    """).fetchall()
    enc_by_patient: dict[int, list[dict]] = defaultdict(list)
    for pid, eid, edate, etype, eorder, cc in enc_data:
        enc_by_patient[pid].append({
            "encounter_id": eid, "encounter_date": edate,
            "encounter_type": etype, "encounter_order": eorder,
            "chief_complaint": cc,
        })

    # Phase 2: Check cache
    to_generate: list[tuple] = []
    cached_results: dict[int, dict] = {}  # encounter_id -> result
    for r in imaging_encounters:
        enc_id = r[0]
        ih = _compute_input_hash(STAGE_10E, enc_id)
        cached = _check_cache(conn, STAGE_10E, ih)
        if cached:
            cached_results[enc_id] = cached
        else:
            to_generate.append(r)

    log.info(
        f"  Imaging indication: {len(cached_results)} cached, "
        f"{len(to_generate)} to generate ({workers} workers)"
    )

    # Phase 3: LLM generation
    ok_count = 0
    err_count = 0
    warn_count = 0
    if to_generate:
        prompts: dict[int, tuple[int, str]] = {}  # enc_id -> (patient_id, prompt)
        for r in to_generate:
            enc_id, pid = r[0], r[1]
            enc_dict = {
                "encounter_date": r[2], "encounter_type": r[3],
                "department": r[4], "chief_complaint": r[5],
                "attending_name": r[6],
            }
            # Get correct dx name
            try:
                src_qids = json.loads(r[8]) if r[8] else []
            except (json.JSONDecodeError, TypeError):
                src_qids = []
            dx_name = "Unknown"
            for qid in src_qids:
                if int(qid) in dx_name_by_qid:
                    dx_name = dx_name_by_qid[int(qid)]
                    break

            # Prior encounters
            prior = [
                e for e in enc_by_patient.get(pid, [])
                if e["encounter_order"] < r[7]
            ]

            prompt = _format_imaging_prompt(
                profiles.get(pid, "{}"),
                enc_dict,
                ees_by_enc.get(enc_id, []),
                imaging_text_by_enc.get(enc_id, ""),
                dx_name,
                prior,
            )
            prompts[enc_id] = (pid, prompt)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _generate_imaging_single, enc_id, pid, prompt
                ): enc_id
                for enc_id, (pid, prompt) in prompts.items()
            }
            done_count = 0
            for future in as_completed(futures):
                enc_id, pid, result, error = future.result()
                ih = _compute_input_hash(STAGE_10E, enc_id)
                if error:
                    err_count += 1
                    conn.execute(
                        "INSERT INTO llm_call_log (stage, model, input_hash, error) "
                        "VALUES (?, ?, ?, ?)",
                        (STAGE_10E, MODEL, ih, error),
                    )
                else:
                    ok_count += 1
                    cached_results[enc_id] = result["output_json"]
                    warnings = _validate_imaging_response(result["output_json"])
                    if warnings:
                        warn_count += 1
                        log.warning(f"  enc={enc_id}: {', '.join(warnings)}")
                    conn.execute(
                        "INSERT INTO llm_call_log "
                        "(stage, model, input_hash, output_json, input_tokens, output_tokens, request_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (STAGE_10E, MODEL, ih,
                         json.dumps(result["output_json"], ensure_ascii=False),
                         result["input_tokens"], result["output_tokens"],
                         result["request_id"]),
                    )
                done_count += 1
                if done_count % 10 == 0:
                    conn.commit()
                if done_count % 50 == 0:
                    log.info(f"  Progress: {done_count}/{len(to_generate)}")

        conn.commit()

    # Phase 4: Insert ground truth + imaging orders
    conn.execute(
        "DELETE FROM imaging_orders WHERE gt_id IN "
        "(SELECT gt_id FROM benchmark_ground_truth WHERE task = 'imaging_indication')"
    )
    conn.execute(
        "DELETE FROM benchmark_ground_truth WHERE task = 'imaging_indication'"
    )

    gt_inserted = 0
    orders_inserted = 0
    for r in imaging_encounters:
        enc_id, pid = r[0], r[1]
        llm_result = cached_results.get(enc_id)
        if not llm_result:
            continue

        order_data = llm_result.get("imaging_order", {})
        gt_data = llm_result.get("ground_truth", {})

        # Difficulty
        indication = order_data.get("clinical_indication", "")
        differentials = gt_data.get("differential_context", [])
        is_stat = order_data.get("order_priority") == "stat"
        difficulty = _compute_difficulty_imaging(
            len(indication.split()), len(differentials), is_stat
        )

        gt_json = json.dumps(gt_data, ensure_ascii=False)
        conn.execute(
            "INSERT INTO benchmark_ground_truth "
            "(task, granularity, encounter_id, patient_id, ground_truth, difficulty) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("imaging_indication", "encounter", enc_id, pid, gt_json, difficulty),
        )
        gt_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        gt_inserted += 1

        conn.execute(
            "INSERT INTO imaging_orders "
            "(encounter_id, gt_id, modality, body_region, clinical_indication, "
            "ordering_provider, order_priority, order_datetime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (enc_id, gt_id,
             order_data.get("modality", "other"),
             order_data.get("body_region", "unknown"),
             indication,
             r[6],  # attending_name as ordering provider
             order_data.get("order_priority", "routine"),
             r[2]),  # encounter_date as order datetime
        )
        orders_inserted += 1

    conn.commit()
    log.info(
        f"Sub-stage 10e complete: {gt_inserted} imaging gt, {orders_inserted} orders "
        f"({ok_count} generated, {len(cached_results) - ok_count} cached, "
        f"{err_count} errors, {warn_count} warnings)"
    )
    return {"gt_inserted": gt_inserted, "orders_inserted": orders_inserted,
            "ok": ok_count, "errors": err_count, "warnings": warn_count}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_10(conn: sqlite3.Connection) -> None:
    """Comprehensive verification for Stage 10 output."""
    log.info("--- Stage 10 Verification ---")

    # Row counts per task + granularity
    for task in ("diagnosis_accuracy", "context_summarization",
                 "evidence_retrieval", "imaging_indication"):
        for gran in ("question", "patient", "encounter"):
            count = conn.execute(
                "SELECT COUNT(*) FROM benchmark_ground_truth "
                "WHERE task = ? AND granularity = ?", (task, gran)
            ).fetchone()[0]
            if count > 0:
                log.info(f"  {task}/{gran}: {count}")

    # Total
    total = conn.execute("SELECT COUNT(*) FROM benchmark_ground_truth").fetchone()[0]
    log.info(f"  benchmark_ground_truth total: {total}")

    # Question coverage for dx accuracy
    q_total = conn.execute("SELECT COUNT(*) FROM board_questions").fetchone()[0]
    q_covered = conn.execute(
        "SELECT COUNT(DISTINCT question_id) FROM benchmark_ground_truth "
        "WHERE task = 'diagnosis_accuracy' AND granularity = 'question'"
    ).fetchone()[0]
    log.info(f"  dx_accuracy question coverage: {q_covered}/{q_total}")

    # Patient coverage
    p_total = conn.execute("SELECT COUNT(*) FROM longitudinal_patients").fetchone()[0]
    for task in ("diagnosis_accuracy", "context_summarization", "evidence_retrieval"):
        p_covered = conn.execute(
            "SELECT COUNT(DISTINCT patient_id) FROM benchmark_ground_truth "
            "WHERE task = ? AND granularity = 'patient'", (task,)
        ).fetchone()[0]
        log.info(f"  {task} patient coverage: {p_covered}/{p_total}")

    # Imaging coverage
    enc_with_imaging = conn.execute(
        "SELECT COUNT(DISTINCT encounter_id) FROM encounter_ehr_sections "
        "WHERE section_type = 'imaging'"
    ).fetchone()[0]
    imaging_gt = conn.execute(
        "SELECT COUNT(*) FROM benchmark_ground_truth WHERE task = 'imaging_indication'"
    ).fetchone()[0]
    log.info(f"  imaging_indication coverage: {imaging_gt}/{enc_with_imaging}")

    # Difficulty distribution
    log.info("  Difficulty distribution:")
    for row in conn.execute(
        "SELECT task, difficulty, COUNT(*) FROM benchmark_ground_truth "
        "GROUP BY task, difficulty ORDER BY task, difficulty"
    ).fetchall():
        log.info(f"    {row[0]}/{row[1]}: {row[2]}")

    # Relevance judgments
    rj_total = conn.execute("SELECT COUNT(*) FROM relevance_judgments").fetchone()[0]
    log.info(f"  relevance_judgments total: {rj_total}")
    for row in conn.execute(
        "SELECT relevance_grade, COUNT(*) FROM relevance_judgments "
        "GROUP BY relevance_grade ORDER BY relevance_grade"
    ).fetchall():
        log.info(f"    Grade {row[0]}: {row[1]}")

    # Imaging orders
    io_total = conn.execute("SELECT COUNT(*) FROM imaging_orders").fetchone()[0]
    log.info(f"  imaging_orders total: {io_total}")
    if io_total > 0:
        log.info("  Modality distribution:")
        for row in conn.execute(
            "SELECT modality, COUNT(*) FROM imaging_orders "
            "GROUP BY modality ORDER BY COUNT(*) DESC"
        ).fetchall():
            log.info(f"    {row[0]}: {row[1]}")

    # LLM call stats
    for stage in (STAGE_10C, STAGE_10E):
        ok = conn.execute(
            "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NULL",
            (stage,),
        ).fetchone()[0]
        err = conn.execute(
            "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NOT NULL",
            (stage,),
        ).fetchone()[0]
        log.info(f"  LLM calls ({stage}): {ok} OK / {err} errors")

    # JSON validity sample
    invalid = 0
    for (gt_json,) in conn.execute(
        "SELECT ground_truth FROM benchmark_ground_truth ORDER BY RANDOM() LIMIT 100"
    ).fetchall():
        try:
            json.loads(gt_json, strict=False)
        except Exception:
            invalid += 1
    log.info(f"  Invalid JSON (sampled 100): {invalid}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv_10(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Export Stage 10 tables to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for table, filename in [
        ("benchmark_ground_truth", "s10_benchmark_ground_truth.csv"),
        ("relevance_judgments", "s10_relevance_judgments.csv"),
        ("imaging_orders", "s10_imaging_orders.csv"),
    ]:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
        if not rows:
            log.info(f"  {table}: 0 rows, skipping")
            continue
        cols = [desc[0] for desc in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]  # noqa: S608
        path = output_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        log.info(f"  Exported {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 10: Build Ground Truth")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step", choices=["10a", "10b", "10c", "10c_visit", "10d", "10e"])
    group.add_argument("--all", action="store_true")
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--export-csv", action="store_true")

    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()
    conn = get_connection(args.db)

    # Always apply migration first
    _apply_schema_migration_10(conn)

    pilot = args.pilot if args.pilot > 0 else None

    if args.verify_only:
        verify_10(conn)
        return
    if args.export_csv:
        export_csv_10(conn, DATA_DIR)
        return

    dispatch = {
        "10a": lambda: run_10a(conn, pilot),
        "10b": lambda: run_10b(conn, pilot),
        "10c": lambda: run_10c(conn, pilot, args.workers),
        "10c_visit": lambda: run_10c_visit(conn, pilot, args.workers),
        "10d": lambda: run_10d(conn, pilot),
        "10e": lambda: run_10e(conn, pilot, args.workers),
    }

    if args.all:
        for step in ["10a", "10b", "10c", "10c_visit", "10d", "10e"]:
            log.info(f"\n{'='*60}")
            dispatch[step]()
    elif args.step:
        dispatch[args.step]()


if __name__ == "__main__":
    main()
