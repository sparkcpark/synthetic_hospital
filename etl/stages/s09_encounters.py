"""Stage 9: Encounter Planning & Clinical Note Generation.

Step 9a: Plan encounter timelines via LLM (1 call per patient, ~1,268 calls)
Step 9b: Generate clinical notes (hybrid: template assembly + LLM HPI polish)

Usage:
    python -m etl.stages.s09_encounters --step 9a [--pilot N] [--workers N] [--db PATH]
    python -m etl.stages.s09_encounters --step 9b [--pilot N] [--workers N] [--db PATH]
    python -m etl.stages.s09_encounters --verify-only [--db PATH]
    python -m etl.stages.s09_encounters --export-csv [--db PATH]
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from etl.config import DB_PATH, DATA_DIR
from etl.db import get_connection
from etl.stages.s05_ontology import (
    MODEL,
    _call_with_retry,
)
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s09_encounters")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_9A = "s09a_timeline"
DEFAULT_WORKERS = 5

VALID_ENCOUNTER_TYPES = {
    "outpatient", "ed", "inpatient", "icu",
    "telehealth", "procedure", "follow_up",
}

# Normalize common LLM variations
ENCOUNTER_TYPE_ALIASES: dict[str, str] = {
    "emergency": "ed",
    "emergency_department": "ed",
    "emergency department": "ed",
    "er": "ed",
    "office": "outpatient",
    "clinic": "outpatient",
    "ambulatory": "outpatient",
    "surgery": "procedure",
    "followup": "follow_up",
    "follow-up": "follow_up",
    "post_discharge": "follow_up",
    "admission": "inpatient",
    "hospital": "inpatient",
    "critical_care": "icu",
    "intensive_care": "icu",
    "telemedicine": "telehealth",
    "virtual": "telehealth",
}

# Stage 9b constants
STAGE_9B = "s09b_notes"

NOTE_SECTION_ORDER = [
    "chief_complaint", "hpi", "pmh", "psh", "medications", "allergies",
    "family_history", "social_history", "ros", "vitals", "physical_exam",
    "labs", "imaging", "pathology", "other_studies", "assessment", "plan",
]

SECTION_HEADERS: dict[str, str] = {
    "chief_complaint": "CHIEF COMPLAINT",
    "hpi": "HISTORY OF PRESENT ILLNESS",
    "pmh": "PAST MEDICAL HISTORY",
    "psh": "PAST SURGICAL HISTORY",
    "medications": "MEDICATIONS",
    "allergies": "ALLERGIES",
    "family_history": "FAMILY HISTORY",
    "social_history": "SOCIAL HISTORY",
    "ros": "REVIEW OF SYSTEMS",
    "vitals": "VITAL SIGNS",
    "physical_exam": "PHYSICAL EXAMINATION",
    "labs": "LABORATORY DATA",
    "imaging": "IMAGING",
    "pathology": "PATHOLOGY",
    "other_studies": "OTHER STUDIES",
    "assessment": "ASSESSMENT AND PLAN",
    "plan": "PLAN",
}

RE_AGE_YEARS = re.compile(r"(\d{1,3})\s*[-\s]?\s*year[\s-]*old", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TIMELINE_PROMPT = """You are a clinical documentation specialist planning a realistic longitudinal encounter timeline for a simulated patient in a medical education simulation.

PATIENT PROFILE:
{patient_profile_json}

This patient has {num_encounters} clinical encounters derived from USMLE board questions. Each encounter maps to one source question. Your job is to arrange them into a chronologically plausible sequence and assign clinical metadata.

SOURCE ENCOUNTERS:
{encounters_block}

TIMELINE RULES:
1. Chronic conditions (e.g., hypertension management, diabetes follow-up) should appear EARLIER in the timeline
2. Acute presentations (e.g., MI, appendicitis, stroke) should appear LATER, representing new events
3. Total timeline should span {timeline_hint} depending on the mix of conditions
4. encounter_type must reflect clinical acuity:
   - "outpatient" = routine clinic visit, chronic disease management
   - "ed" = emergency department visit for acute presentation
   - "inpatient" = hospital admission (follows ED or direct admission)
   - "icu" = intensive care unit (critical illness)
   - "telehealth" = remote visit (minor follow-ups, medication refills)
   - "procedure" = scheduled procedure (surgery, endoscopy, biopsy)
   - "follow_up" = post-discharge or post-procedure check
5. Each source question must appear exactly once
6. relative_date is in months from the first encounter (first encounter = 0)
7. Dates must be monotonically non-decreasing
8. Department should match the clinical scenario (e.g., "Internal Medicine", "Cardiology", "Emergency Medicine", "Obstetrics & Gynecology")
9. attending_name should be a realistic fictional physician name in "Dr. [First] [Last]" format — use diverse names, same department may reuse the same attending for continuity

For each encounter, return:
- source_question_id: the question_id this encounter is based on
- encounter_type: one of the types listed above
- relative_date: integer months from first encounter (0 for first visit)
- chief_complaint: 1-2 sentence reason for presenting
- department: clinical department name
- attending_name: "Dr. [First] [Last]"
- clinical_rationale: brief explanation of why this encounter occurs at this point in the timeline

Return ONLY valid JSON (no markdown, no explanation):
{{
  "encounters": [
    {{
      "source_question_id": 1234,
      "encounter_type": "outpatient",
      "relative_date": 0,
      "chief_complaint": "...",
      "department": "...",
      "attending_name": "Dr. ...",
      "clinical_rationale": "..."
    }}
  ]
}}"""


HPI_POLISH_PROMPT = """You are a clinical documentation specialist rewriting an HPI section to reflect a patient's longitudinal medical history in an EHR simulation.

PATIENT: {age}-year-old {sex} with {chronic_conditions}
Home medications: {home_medications}

CURRENT ENCOUNTER (#{encounter_number} of {total_encounters}):
- Date: {encounter_date}
- Type: {encounter_type}
- Department: {department}
- Chief Complaint: {chief_complaint}
- Diagnosis: {current_dx}

PRIOR ENCOUNTERS:
{prior_encounters_summary}

ORIGINAL HPI:
{original_hpi}

REWRITING RULES:
1. Preserve ALL clinical details from the original HPI — do not remove any symptoms, signs, timing, or clinical language
2. Add a 1-2 sentence opening that references the patient's known history and relevant prior encounters
3. If this is a follow-up visit, frame the presentation as "returning for..." or "presents with..." in context of prior care
4. Do NOT add new clinical findings not present in the original HPI
5. Do NOT include diagnoses or assessments — keep it in HPI style
6. Keep the total length within 150% of the original
7. Use clinical documentation language (third person, present tense)

Return ONLY valid JSON (no markdown, no explanation):
{{"polished_hpi": "the rewritten HPI text here"}}"""


# ---------------------------------------------------------------------------
# Helper functions (9a)
# ---------------------------------------------------------------------------


def _relative_months_to_date(months: int) -> str:
    """Convert relative_date (months from first encounter) to ISO date."""
    months = max(0, int(months))
    year = 2020 + (months // 12)
    month = 1 + (months % 12)
    return f"{year:04d}-{month:02d}-15"


def _compute_timeline_hint(acuity_types: list[str], num_encounters: int) -> str:
    """Compute timeline span hint from diagnosis acuities."""
    has_chronic = any(
        a in ("chronic", "acute_on_chronic", "mixed") for a in acuity_types
    )
    if not has_chronic:
        return "6 months to 2 years"
    elif num_encounters <= 4:
        return "1 to 5 years"
    else:
        return "3 to 10 years"


def _format_patient_encounters_block(
    encounters_context: list[dict],
) -> str:
    """Build the encounters block for the prompt.

    Each entry in encounters_context has:
        qid, dx_names, dx_acuities, organ_system, cc, hpi
    """
    parts = []
    for ctx in encounters_context:
        lines = [f"Encounter (Question ID: {ctx['qid']}):"]

        dx_strs = []
        for name, acuity in zip(ctx["dx_names"], ctx["dx_acuities"]):
            dx_strs.append(f"{name} ({acuity})")
        if dx_strs:
            lines.append(f"  Correct Diagnosis: {'; '.join(dx_strs)}")

        lines.append(f"  Organ System: {ctx['organ_system']}")

        if ctx["cc"]:
            lines.append(f"  Chief Complaint: {ctx['cc']}")
        if ctx["hpi"]:
            lines.append(f"  HPI Summary: {ctx['hpi']}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _compute_input_hash_9a(patient_id: int, encounter_qids: list[int]) -> str:
    """SHA-256 hash for caching a 9a LLM call."""
    qids_str = ",".join(str(q) for q in sorted(encounter_qids))
    content = f"{STAGE_9A}:{patient_id}:{qids_str}"
    return hashlib.sha256(content.encode()).hexdigest()


def _check_cache_9a(conn: sqlite3.Connection, input_hash: str) -> dict | None:
    """Check llm_call_log for a cached Stage 9a result."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log "
        "WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (STAGE_9A, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _normalize_encounter_type(raw: str) -> str:
    """Normalize encounter_type to valid DB values."""
    raw_lower = raw.strip().lower()
    if raw_lower in VALID_ENCOUNTER_TYPES:
        return raw_lower
    alias = ENCOUNTER_TYPE_ALIASES.get(raw_lower)
    if alias:
        return alias
    return "outpatient"  # safe default


# ---------------------------------------------------------------------------
# Thread-safe LLM worker
# ---------------------------------------------------------------------------


def _plan_timeline_single(
    patient_id: int,
    encounter_qids: list[int],
    prompt: str,
) -> dict:
    """Thread-safe LLM worker: plan encounter timeline. No DB access."""
    call_t0 = time.time()
    try:
        parsed, in_tok, out_tok, raw_text = _call_with_retry(prompt)
        latency_ms = int((time.time() - call_t0) * 1000)

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

        if "encounters" not in parsed:
            raise ValueError("Missing 'encounters' key in response")

        return {
            "patient_id": patient_id,
            "encounter_qids": encounter_qids,
            "timeline": parsed,
            "in_tok": in_tok,
            "out_tok": out_tok,
            "raw_text": raw_text,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        latency_ms = int((time.time() - call_t0) * 1000)
        return {
            "patient_id": patient_id,
            "encounter_qids": encounter_qids,
            "timeline": None,
            "in_tok": 0,
            "out_tok": 0,
            "raw_text": None,
            "latency_ms": latency_ms,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_timeline(
    timeline: dict,
    expected_qids: list[int],
) -> list[str]:
    """Validate LLM timeline response. Returns list of warnings."""
    warnings = []

    encounters = timeline.get("encounters", [])
    if not isinstance(encounters, list):
        warnings.append(f"'encounters' is not a list: {type(encounters).__name__}")
        return warnings

    # Count mismatch
    if len(encounters) != len(expected_qids):
        warnings.append(
            f"Encounter count mismatch: got {len(encounters)}, "
            f"expected {len(expected_qids)}"
        )

    # Check source_question_ids coverage
    returned_qids = set()
    for enc in encounters:
        sqid = enc.get("source_question_id")
        if sqid is not None:
            try:
                returned_qids.add(int(sqid))
            except (ValueError, TypeError):
                pass

    missing = set(expected_qids) - returned_qids
    extra = returned_qids - set(expected_qids)
    if missing:
        warnings.append(f"Missing question_ids: {missing}")
    if extra:
        warnings.append(f"Extra question_ids: {extra}")

    # Chronological order
    dates = []
    for enc in encounters:
        try:
            dates.append(int(enc.get("relative_date", 0)))
        except (ValueError, TypeError):
            dates.append(0)
    if dates != sorted(dates):
        warnings.append("Encounters not in chronological order (will be re-sorted)")

    # First encounter at 0
    if dates and min(dates) != 0:
        warnings.append(f"First encounter relative_date={min(dates)}, expected 0")

    # Valid encounter_types
    for enc in encounters:
        et = enc.get("encounter_type", "")
        normalized = _normalize_encounter_type(str(et))
        if normalized != str(et).strip().lower():
            warnings.append(f"Normalized encounter_type '{et}' → '{normalized}'")

    # Required fields
    required = {
        "source_question_id", "encounter_type", "relative_date",
        "chief_complaint", "department", "attending_name",
    }
    for i, enc in enumerate(encounters):
        missing_fields = required - set(enc.keys())
        if missing_fields:
            warnings.append(f"Encounter {i}: missing fields {missing_fields}")

    # Timeline span
    if dates:
        span = max(dates) - min(dates)
        if span > 120:
            warnings.append(f"Timeline span {span} months exceeds 10-year maximum")

    return warnings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_9a(
    conn: sqlite3.Connection,
    pilot: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Stage 9a: Plan encounter timelines via LLM.

    Phase 1: Load patients + encounters, build context and prompts
    Phase 2: Check cache, build work queue
    Phase 3: Concurrent LLM generation
    Phase 4: Validate + UPDATE longitudinal_encounters
    """
    t0 = time.time()

    # Phase 1: Load patients + build prompts
    log.info("Phase 1: Loading patients and building prompts...")

    patient_rows = conn.execute(
        "SELECT patient_id, profile, age, sex "
        "FROM longitudinal_patients "
        "WHERE profile <> '{}' "
        "ORDER BY patient_id"
    ).fetchall()

    if not patient_rows:
        log.info("No patients with profiles — run Stage 8b first")
        return {"step": "9a", "patients_total": 0}

    if pilot:
        patient_rows = patient_rows[:pilot]

    # Preload all diagnosis info
    dx_lookup: dict[int, list[tuple[str, str]]] = {}  # qid → [(name, acuity)]
    for qid, name, acuity in conn.execute(
        "SELECT qd.question_id, d.display_name, d.acuity "
        "FROM question_diagnoses qd "
        "JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id "
        "WHERE qd.role = 'correct'"
    ).fetchall():
        dx_lookup.setdefault(qid, []).append((name, acuity or "unspecified"))

    # Preload organ systems
    organ_lookup: dict[int, str] = {}
    for qid, os_val in conn.execute(
        "SELECT question_id, organ_system FROM board_questions"
    ).fetchall():
        organ_lookup[qid] = os_val or "Unknown"

    # Preload chief_complaint + hpi sections
    cc_lookup: dict[int, str] = {}
    hpi_lookup: dict[int, str] = {}
    for qid, stype, stext in conn.execute(
        "SELECT question_id, section_type, section_text FROM ehr_sections "
        "WHERE section_type IN ('chief_complaint', 'hpi') "
        "ORDER BY question_id, section_order"
    ).fetchall():
        if stype == "chief_complaint" and qid not in cc_lookup:
            cc_lookup[qid] = stext[:200]
        elif stype == "hpi" and qid not in hpi_lookup:
            hpi_lookup[qid] = stext[:300]

    # Build per-patient work items
    patient_work: list[tuple[int, list[int], str, str]] = []
    # Each item: (patient_id, encounter_qids, prompt, input_hash)

    for patient_id, profile_json, age, sex in patient_rows:
        profile = json.loads(profile_json)

        enc_rows = conn.execute(
            "SELECT encounter_id, source_question_ids, encounter_order "
            "FROM longitudinal_encounters "
            "WHERE patient_id = ? ORDER BY encounter_order",
            (patient_id,),
        ).fetchall()

        encounter_qids = []
        encounters_context = []
        acuity_types = []

        for enc_id, sqids_json, enc_order in enc_rows:
            qid = json.loads(sqids_json)[0]
            encounter_qids.append(qid)

            dx_info = dx_lookup.get(qid, [])
            dx_names = [d[0] for d in dx_info]
            dx_acuities = [d[1] for d in dx_info]
            acuity_types.extend(dx_acuities)

            encounters_context.append({
                "qid": qid,
                "dx_names": dx_names,
                "dx_acuities": dx_acuities,
                "organ_system": organ_lookup.get(qid, "Unknown"),
                "cc": cc_lookup.get(qid, ""),
                "hpi": hpi_lookup.get(qid, ""),
            })

        encounters_block = _format_patient_encounters_block(encounters_context)
        timeline_hint = _compute_timeline_hint(acuity_types, len(encounter_qids))

        prompt = TIMELINE_PROMPT.format(
            patient_profile_json=json.dumps(profile, indent=2),
            num_encounters=len(encounter_qids),
            encounters_block=encounters_block,
            timeline_hint=timeline_hint,
        )

        input_hash = _compute_input_hash_9a(patient_id, encounter_qids)
        patient_work.append((patient_id, encounter_qids, prompt, input_hash))

    log.info(f"Phase 1 complete: {len(patient_work)} patients prepared")

    # Phase 2: Check cache
    log.info("Phase 2: Checking cache...")
    work_queue: list[tuple[int, list[int], str, str]] = []
    cached_results: dict[int, dict] = {}

    for patient_id, encounter_qids, prompt, input_hash in patient_work:
        cached = _check_cache_9a(conn, input_hash)
        if cached is not None:
            cached_results[patient_id] = cached
            continue
        work_queue.append((patient_id, encounter_qids, prompt, input_hash))

    log.info(
        f"Cache check: {len(cached_results)} cached, {len(work_queue)} to generate "
        f"({workers} workers)"
    )

    # Phase 3: Concurrent LLM generation
    generated = 0
    errors = 0

    if work_queue:
        completed_count = 0
        pending = len(work_queue)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for patient_id, encounter_qids, prompt, input_hash in work_queue:
                future = pool.submit(
                    _plan_timeline_single, patient_id, encounter_qids, prompt
                )
                future_map[future] = (patient_id, encounter_qids, input_hash)

            for future in as_completed(future_map):
                patient_id, encounter_qids, input_hash = future_map[future]
                result = future.result()
                completed_count += 1

                if result["error"] is None:
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'encounter_timeline', ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            STAGE_9A, MODEL, result["in_tok"], result["out_tok"],
                            result["latency_ms"], input_hash,
                            json.dumps(result["timeline"]),
                            result["raw_text"], None,
                        ),
                    )
                    cached_results[patient_id] = result["timeline"]
                    generated += 1
                else:
                    log.error(f"patient_id={patient_id} failed: {result['error']}")
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'encounter_timeline', 0, 0, ?, ?, NULL, NULL, ?)""",
                        (
                            STAGE_9A, MODEL, result["latency_ms"],
                            input_hash, result["error"],
                        ),
                    )
                    errors += 1

                if completed_count % 10 == 0:
                    conn.commit()

                if completed_count % 50 == 0 or completed_count == pending:
                    elapsed = time.time() - t0
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = pending - completed_count
                    eta = remaining / rate if rate > 0 else 0
                    log.info(
                        f"Progress {completed_count}/{pending} | "
                        f"generated={generated} errors={errors} | "
                        f"{elapsed:.0f}s elapsed | ETA={eta:.0f}s"
                    )

        conn.commit()

    # Phase 4: Validate + UPDATE longitudinal_encounters
    log.info("Phase 4: Validating and updating encounters...")
    updated_patients = 0
    updated_encounters = 0
    validation_warnings = 0
    reorder_count = 0

    for patient_id, encounter_qids, prompt, input_hash in patient_work:
        timeline = cached_results.get(patient_id)
        if timeline is None:
            continue

        warnings = _validate_timeline(timeline, encounter_qids)
        for w in warnings:
            log.warning(f"patient_id={patient_id}: {w}")
            validation_warnings += len(warnings)

        planned = timeline.get("encounters", [])
        if not planned:
            log.warning(f"patient_id={patient_id}: empty encounters list")
            continue

        # Build qid → planned encounter map
        qid_plan: dict[int, dict] = {}
        for enc in planned:
            sqid = enc.get("source_question_id")
            if sqid is not None:
                try:
                    qid_plan[int(sqid)] = enc
                except (ValueError, TypeError):
                    pass

        # Sort planned by relative_date for new encounter_order
        def _sort_key(e):
            try:
                return int(e.get("relative_date", 0))
            except (ValueError, TypeError):
                return 0

        planned_sorted = sorted(planned, key=_sort_key)
        new_order_map: dict[int, int] = {}
        for new_order, enc in enumerate(planned_sorted):
            sqid = enc.get("source_question_id")
            if sqid is not None:
                try:
                    new_order_map[int(sqid)] = new_order
                except (ValueError, TypeError):
                    pass

        # Check if reordering happened
        old_qids = encounter_qids  # already in encounter_order
        new_qids = [
            int(enc.get("source_question_id", 0))
            for enc in planned_sorted
            if enc.get("source_question_id") is not None
        ]
        if old_qids != new_qids:
            reorder_count += 1

        # Get existing encounter rows
        existing = conn.execute(
            "SELECT encounter_id, source_question_ids, encounter_order "
            "FROM longitudinal_encounters "
            "WHERE patient_id = ? ORDER BY encounter_order",
            (patient_id,),
        ).fetchall()

        for enc_id, sqids_json, old_enc_order in existing:
            qid = json.loads(sqids_json)[0]
            plan = qid_plan.get(qid)
            if plan is None:
                log.warning(
                    f"patient_id={patient_id}, qid={qid}: "
                    "not in LLM response, keeping defaults"
                )
                continue

            enc_type = _normalize_encounter_type(
                str(plan.get("encounter_type", "outpatient"))
            )

            try:
                relative_months = int(plan.get("relative_date", 0))
            except (ValueError, TypeError):
                relative_months = 0
            encounter_date = _relative_months_to_date(relative_months)

            chief_complaint = str(plan.get("chief_complaint", ""))
            department = str(plan.get("department", ""))
            attending_name = str(plan.get("attending_name", ""))
            new_order = new_order_map.get(qid, old_enc_order)

            conn.execute(
                """UPDATE longitudinal_encounters
                   SET encounter_date = ?,
                       encounter_type = ?,
                       chief_complaint = ?,
                       department = ?,
                       attending_name = ?,
                       encounter_order = ?,
                       generation_method = 'llm'
                   WHERE encounter_id = ?""",
                (
                    encounter_date,
                    enc_type,
                    chief_complaint,
                    department,
                    attending_name,
                    new_order,
                    enc_id,
                ),
            )
            updated_encounters += 1

        updated_patients += 1

    conn.commit()

    duration = time.time() - t0
    summary = {
        "step": "9a",
        "patients_total": len(patient_work),
        "cached": len(cached_results) - generated,
        "generated": generated,
        "errors": errors,
        "updated_patients": updated_patients,
        "updated_encounters": updated_encounters,
        "reordered_patients": reorder_count,
        "validation_warnings": validation_warnings,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Stage 9a complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_9a(conn: sqlite3.Connection):
    """Run verification queries for Stage 9a output."""
    log.info("--- Stage 9a Verification ---")

    total_enc = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters"
    ).fetchone()[0]
    log.info(f"Total encounters: {total_enc}")

    if total_enc == 0:
        log.warning("No encounters — run Stage 8a first")
        return

    # Encounters with real dates
    real_dates = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters "
        "WHERE encounter_date <> '1970-01-01'"
    ).fetchone()[0]
    log.info(
        f"Encounters with real dates: {real_dates}/{total_enc} "
        f"({real_dates / total_enc * 100:.1f}%)"
    )

    # Chief complaint coverage
    with_cc = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters "
        "WHERE chief_complaint IS NOT NULL AND chief_complaint <> ''"
    ).fetchone()[0]
    log.info(f"Encounters with chief_complaint: {with_cc}/{total_enc}")

    # Attending name coverage
    with_att = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters "
        "WHERE attending_name IS NOT NULL AND attending_name <> ''"
    ).fetchone()[0]
    log.info(f"Encounters with attending_name: {with_att}/{total_enc}")

    # Department coverage
    with_dept = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters "
        "WHERE department IS NOT NULL AND department <> ''"
    ).fetchone()[0]
    log.info(f"Encounters with department: {with_dept}/{total_enc}")

    # Encounter type distribution
    log.info("Encounter type distribution:")
    for etype, cnt in conn.execute(
        "SELECT encounter_type, COUNT(*) FROM longitudinal_encounters "
        "GROUP BY encounter_type ORDER BY COUNT(*) DESC"
    ).fetchall():
        log.info(f"  {etype}: {cnt}")

    # Generation method distribution
    log.info("Generation method distribution:")
    for method, cnt in conn.execute(
        "SELECT generation_method, COUNT(*) FROM longitudinal_encounters "
        "GROUP BY generation_method ORDER BY COUNT(*) DESC"
    ).fetchall():
        log.info(f"  {method}: {cnt}")

    # Timeline span stats (only for patients with real dates)
    span_rows = conn.execute(
        "SELECT "
        "  julianday(MAX(encounter_date)) - julianday(MIN(encounter_date)) AS span "
        "FROM longitudinal_encounters "
        "WHERE encounter_date <> '1970-01-01' "
        "GROUP BY patient_id"
    ).fetchall()
    if span_rows:
        spans = [r[0] for r in span_rows if r[0] is not None]
        if spans:
            avg_span = sum(spans) / len(spans)
            log.info(
                f"Timeline span (days): avg={avg_span:.0f}, "
                f"min={min(spans):.0f}, max={max(spans):.0f}"
            )

    # Temporal monotonicity check
    non_monotonic = 0
    for (pid,) in conn.execute(
        "SELECT DISTINCT patient_id FROM longitudinal_encounters"
    ).fetchall():
        dates = [
            d[0] for d in conn.execute(
                "SELECT encounter_date FROM longitudinal_encounters "
                "WHERE patient_id = ? AND encounter_date <> '1970-01-01' "
                "ORDER BY encounter_order",
                (pid,),
            ).fetchall()
        ]
        if dates != sorted(dates):
            non_monotonic += 1
    log.info(f"Temporal monotonicity violations: {non_monotonic} patients")

    # LLM call log stats
    ok_9a = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NULL",
        (STAGE_9A,),
    ).fetchone()[0]
    err_9a = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NOT NULL",
        (STAGE_9A,),
    ).fetchone()[0]
    if ok_9a or err_9a:
        log.info(f"LLM calls (9a): {ok_9a} OK / {err_9a} errors")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


def export_csv_9a(conn: sqlite3.Connection, output_dir: Path):
    """Export longitudinal_encounters to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "s09_longitudinal_encounters.csv"
    rows = conn.execute(
        """SELECT encounter_id, patient_id, encounter_date, encounter_type,
                  chief_complaint, attending_name, department,
                  source_question_ids, encounter_order,
                  note_text, generation_method, created_at
           FROM longitudinal_encounters ORDER BY patient_id, encounter_order"""
    ).fetchall()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "encounter_id", "patient_id", "encounter_date", "encounter_type",
            "chief_complaint", "attending_name", "department",
            "source_question_ids", "encounter_order",
            "note_text", "generation_method", "created_at",
        ])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} encounters to {path}")


# ===========================================================================
# STAGE 9b: Clinical Note Generation (hybrid: template + LLM HPI polish)
# ===========================================================================


# ---------------------------------------------------------------------------
# 9b Helper functions
# ---------------------------------------------------------------------------


def _build_cumulative_pmh(
    source_pmh: str | None,
    prior_dx_entries: list[tuple[str, str]],  # [(dx_name, encounter_date)]
    profile_chronic: list[str],
) -> tuple[str, bool]:
    """Build cumulative PMH by prepending prior diagnoses.

    Returns (pmh_text, was_modified).
    """
    problem_items: list[str] = []
    seen = set()

    for dx_name, dx_date in prior_dx_entries:
        key = dx_name.lower()
        if key not in seen:
            seen.add(key)
            problem_items.append(f"- {dx_name} (diagnosed {dx_date})")

    for cond in profile_chronic:
        key = cond.lower()
        if key not in seen:
            seen.add(key)
            problem_items.append(f"- {cond}")

    if not problem_items:
        return source_pmh or "", False

    parts = ["Active Problem List:"]
    parts.extend(problem_items)
    parts.append("")
    if source_pmh:
        parts.append(source_pmh)
    return "\n".join(parts), True


def _merge_medications(
    source_meds: str | None,
    profile_home_meds: list[dict],
) -> tuple[str, bool]:
    """Merge source medications with patient profile home_medications.

    Returns (meds_text, was_modified).
    """
    if not profile_home_meds:
        return source_meds or "", False

    source_lower = (source_meds or "").lower()
    extra = []
    for med in profile_home_meds:
        name = med.get("name", "")
        if name and name.lower() not in source_lower:
            dose = med.get("dose", "")
            extra.append(f"- {name} {dose}".strip())

    if not extra:
        return source_meds or "", False

    parts = []
    if source_meds:
        parts.append(source_meds)
    parts.append("\nHome Medications:")
    parts.extend(extra)
    return "\n".join(parts), True


def _fill_from_profile(
    section_type: str,
    source_text: str | None,
    profile: dict,
) -> tuple[str | None, bool]:
    """Fill missing section from patient profile. Returns (text, is_profile_derived)."""
    if source_text:
        return source_text, False

    if section_type == "allergies":
        items = profile.get("allergies", [])
        if items:
            return ", ".join(items), True
        return "No known drug allergies (NKDA)", True

    if section_type == "family_history":
        items = profile.get("family_history", [])
        if items:
            return "\n".join(f"- {item}" for item in items), True
        return None, False

    if section_type == "social_history":
        parts = []
        smoking = profile.get("smoking_status")
        if smoking:
            parts.append(f"Smoking: {smoking}")
        alcohol = profile.get("alcohol_use")
        if alcohol:
            parts.append(f"Alcohol: {alcohol}")
        occupation = profile.get("occupation")
        if occupation:
            parts.append(f"Occupation: {occupation}")
        if parts:
            return ". ".join(parts) + ".", True
        return None, False

    return source_text, False


def _adjust_demographics_age(
    source_text: str | None,
    patient_age: int | None,
    encounter_months_offset: int,
) -> tuple[str | None, bool]:
    """Adjust age in demographics text for encounter date offset.

    Returns (adjusted_text, was_modified).
    """
    if not source_text or patient_age is None:
        return source_text, False
    if encounter_months_offset <= 0:
        return source_text, False

    adjusted_age = patient_age + (encounter_months_offset // 12)
    m = RE_AGE_YEARS.search(source_text)
    if m:
        new_text = source_text[: m.start(1)] + str(adjusted_age) + source_text[m.end(1) :]
        return new_text, True
    return source_text, False


def _assemble_note_text(
    encounter_date: str,
    attending_name: str,
    department: str,
    encounter_type: str,
    chief_complaint: str,
    sections: dict[str, str],
) -> str:
    """Assemble full note_text from encounter metadata + sections."""
    parts = [
        "ENCOUNTER NOTE",
        f"Date: {encounter_date}",
        f"Provider: {attending_name}, MD",
        f"Department: {department}",
        f"Visit Type: {encounter_type}",
        "",
    ]

    for stype in NOTE_SECTION_ORDER:
        text = sections.get(stype)
        if not text:
            continue

        # Combine assessment + plan under one header
        if stype == "plan" and "assessment" in sections:
            continue  # already included with assessment
        if stype == "assessment" and "plan" in sections:
            header = "ASSESSMENT AND PLAN"
            combined = text + "\n\n" + sections["plan"]
            parts.append(f"{header}:")
            parts.append(combined)
            parts.append("")
            continue

        header = SECTION_HEADERS.get(stype, stype.upper())
        parts.append(f"{header}:")
        parts.append(text)
        parts.append("")

    return "\n".join(parts).rstrip()


def _compute_input_hash_9b(encounter_id: int) -> str:
    """SHA-256 hash for caching a 9b LLM call."""
    content = f"{STAGE_9B}:{encounter_id}"
    return hashlib.sha256(content.encode()).hexdigest()


def _check_cache_9b(conn: sqlite3.Connection, input_hash: str) -> dict | None:
    """Check llm_call_log for a cached Stage 9b result."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log "
        "WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (STAGE_9B, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _polish_hpi_single(
    encounter_id: int,
    patient_id: int,
    prompt: str,
) -> dict:
    """Thread-safe LLM worker: polish HPI for longitudinal context. No DB access."""
    call_t0 = time.time()
    try:
        parsed, in_tok, out_tok, raw_text = _call_with_retry(prompt)
        latency_ms = int((time.time() - call_t0) * 1000)

        polished = ""
        if isinstance(parsed, dict):
            polished = parsed.get("polished_hpi", "")
        if not polished and isinstance(parsed, str):
            polished = parsed

        if not polished:
            raise ValueError("Empty polished_hpi in response")

        return {
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "polished_hpi": polished,
            "in_tok": in_tok,
            "out_tok": out_tok,
            "raw_text": raw_text,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        latency_ms = int((time.time() - call_t0) * 1000)
        return {
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "polished_hpi": None,
            "in_tok": 0,
            "out_tok": 0,
            "raw_text": None,
            "latency_ms": latency_ms,
            "error": str(e),
        }


def _validate_polished_hpi(original: str, polished: str) -> list[str]:
    """Validate LLM-polished HPI. Returns list of warnings."""
    warnings = []
    if len(polished) < 50:
        warnings.append(f"Polished HPI too short: {len(polished)} chars")
    if original:
        ratio = len(polished) / len(original)
        if ratio < 0.5:
            warnings.append(f"Polished HPI much shorter than original: {ratio:.1%}")
        elif ratio > 2.5:
            warnings.append(f"Polished HPI much longer than original: {ratio:.1%}")
    return warnings


# ---------------------------------------------------------------------------
# 9b Orchestrator
# ---------------------------------------------------------------------------


def run_9b(
    conn: sqlite3.Connection,
    pilot: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Stage 9b: Generate clinical notes (hybrid template + LLM HPI polish).

    Phase 1: Template assembly for all encounters (deterministic)
    Phase 2: Check cache for LLM HPI polish
    Phase 3: Concurrent LLM HPI polish (non-first encounters)
    Phase 4: Apply polished HPIs + reassemble notes
    """
    t0 = time.time()

    # ---- Phase 1: Template assembly ----
    log.info("Phase 1: Loading data and assembling template notes...")

    # Load patient profiles
    profile_lookup: dict[int, dict] = {}
    age_lookup: dict[int, int | None] = {}
    for pid, profile_json, age in conn.execute(
        "SELECT patient_id, profile, age FROM longitudinal_patients"
    ).fetchall():
        try:
            profile_lookup[pid] = json.loads(profile_json) if profile_json else {}
        except (json.JSONDecodeError, TypeError):
            profile_lookup[pid] = {}
        try:
            age_lookup[pid] = int(age) if age is not None else None
        except (ValueError, TypeError):
            age_lookup[pid] = None

    # Load EHR sections by question_id
    ehr_by_qid: dict[int, list[tuple[int, str, str, int]]] = {}
    for sid, qid, stype, stext, sorder in conn.execute(
        "SELECT section_id, question_id, section_type, section_text, section_order "
        "FROM ehr_sections ORDER BY question_id, section_order"
    ).fetchall():
        ehr_by_qid.setdefault(qid, []).append((sid, stype, stext, sorder))

    # Load correct dx per question
    dx_by_qid: dict[int, list[str]] = {}
    for qid, name in conn.execute(
        "SELECT qd.question_id, d.display_name FROM question_diagnoses qd "
        "JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id "
        "WHERE qd.role = 'correct'"
    ).fetchall():
        dx_by_qid.setdefault(qid, []).append(name)

    # Load encounters grouped by patient
    enc_by_patient: dict[int, list[tuple]] = {}
    for row in conn.execute(
        "SELECT encounter_id, patient_id, encounter_date, encounter_type, "
        "       chief_complaint, attending_name, department, "
        "       source_question_ids, encounter_order "
        "FROM longitudinal_encounters ORDER BY patient_id, encounter_order"
    ).fetchall():
        enc_by_patient.setdefault(row[1], []).append(row)

    patient_ids = sorted(enc_by_patient.keys())
    if pilot:
        patient_ids = patient_ids[:pilot]

    # Clear encounter_ehr_sections for idempotent re-run
    if pilot:
        enc_ids_to_clear = []
        for pid in patient_ids:
            for enc in enc_by_patient.get(pid, []):
                enc_ids_to_clear.append(enc[0])
        if enc_ids_to_clear:
            placeholders = ",".join("?" * len(enc_ids_to_clear))
            conn.execute(
                f"DELETE FROM encounter_ehr_sections WHERE encounter_id IN ({placeholders})",
                enc_ids_to_clear,
            )
    else:
        conn.execute("DELETE FROM encounter_ehr_sections")
    conn.commit()

    total_sections_inserted = 0
    total_notes_assembled = 0
    encounters_no_ehr = 0

    # Track encounter metadata for LLM phase
    encounter_meta: dict[int, dict] = {}  # encounter_id -> metadata

    for pid in patient_ids:
        profile = profile_lookup.get(pid, {})
        patient_age = age_lookup.get(pid)
        prior_dx_entries: list[tuple[str, str]] = []  # (dx_name, enc_date)
        encounters = enc_by_patient.get(pid, [])

        for enc_row in encounters:
            (enc_id, _, enc_date, enc_type, cc, att_name,
             dept, sqids_json, enc_order) = enc_row

            qid = json.loads(sqids_json)[0]
            ehr_sections = ehr_by_qid.get(qid, [])
            dx_names = dx_by_qid.get(qid, [])

            if not ehr_sections:
                encounters_no_ehr += 1

            # Build section_map from source: stype -> (section_id, text)
            section_map: dict[str, tuple[int | None, str]] = {}
            for sid, stype, stext, _ in ehr_sections:
                if stype in section_map:
                    # Concatenate duplicate section types
                    old_sid, old_text = section_map[stype]
                    section_map[stype] = (old_sid, old_text + "\n" + stext)
                else:
                    section_map[stype] = (sid, stext)

            # Compute months offset from first encounter date
            first_enc_date = enc_by_patient[pid][0][2]  # first encounter's date
            months_offset = 0
            if first_enc_date and enc_date and first_enc_date != enc_date:
                try:
                    fy, fm = int(first_enc_date[:4]), int(first_enc_date[5:7])
                    ey, em = int(enc_date[:4]), int(enc_date[5:7])
                    months_offset = (ey - fy) * 12 + (em - fm)
                except (ValueError, IndexError):
                    pass

            # Build final sections
            final_sections: dict[str, str] = {}
            sections_to_insert: list[tuple] = []
            section_order_idx = 0

            # Chief complaint from 9a (not source question)
            final_sections["chief_complaint"] = cc or ""
            sections_to_insert.append((
                enc_id, "chief_complaint", cc or "", None, 1, section_order_idx,
            ))
            section_order_idx += 1

            for stype in NOTE_SECTION_ORDER:
                if stype == "chief_complaint":
                    continue  # already handled

                source_sid, source_text = section_map.get(stype, (None, None))
                is_modified = 0
                final_text = source_text
                final_sid = source_sid

                if stype == "demographics":
                    final_text, was_mod = _adjust_demographics_age(
                        source_text, patient_age, months_offset
                    )
                    if was_mod:
                        is_modified = 1

                elif stype == "pmh":
                    profile_chronic = profile.get("chronic_conditions", [])
                    final_text, was_mod = _build_cumulative_pmh(
                        source_text, prior_dx_entries, profile_chronic
                    )
                    if was_mod:
                        is_modified = 1

                elif stype == "psh":
                    if not source_text:
                        surgical = profile.get("surgical_history", [])
                        if surgical:
                            final_text = "\n".join(f"- {s}" for s in surgical)
                            is_modified = 1
                            final_sid = None

                elif stype == "medications":
                    profile_meds = profile.get("home_medications", [])
                    final_text, was_mod = _merge_medications(source_text, profile_meds)
                    if was_mod:
                        is_modified = 1

                elif stype in ("allergies", "family_history", "social_history"):
                    final_text, is_profile = _fill_from_profile(
                        stype, source_text, profile
                    )
                    if is_profile:
                        is_modified = 1
                        final_sid = None

                if final_text:
                    final_sections[stype] = final_text
                    sections_to_insert.append((
                        enc_id, stype, final_text, final_sid,
                        is_modified, section_order_idx,
                    ))
                    section_order_idx += 1

            # INSERT encounter_ehr_sections
            conn.executemany(
                "INSERT INTO encounter_ehr_sections "
                "(encounter_id, section_type, section_text, source_section_id, "
                " is_modified, section_order) VALUES (?, ?, ?, ?, ?, ?)",
                sections_to_insert,
            )
            total_sections_inserted += len(sections_to_insert)

            # Assemble note_text
            note_text = _assemble_note_text(
                enc_date, att_name or "", dept or "",
                enc_type, cc or "", final_sections,
            )
            conn.execute(
                "UPDATE longitudinal_encounters SET note_text = ?, "
                "generation_method = 'template' WHERE encounter_id = ?",
                (note_text, enc_id),
            )
            total_notes_assembled += 1

            # Store metadata for LLM phase
            original_hpi = section_map.get("hpi", (None, None))[1]
            encounter_meta[enc_id] = {
                "patient_id": pid,
                "encounter_order": enc_order,
                "encounter_date": enc_date,
                "encounter_type": enc_type,
                "chief_complaint": cc,
                "department": dept,
                "dx_names": dx_names,
                "original_hpi": original_hpi,
                "note_sections": final_sections,
            }

            # Accumulate prior dx for next encounter
            for dx in dx_names:
                prior_dx_entries.append((dx, enc_date))

    conn.commit()
    log.info(
        f"Phase 1 complete: {total_notes_assembled} notes assembled, "
        f"{total_sections_inserted} sections inserted, "
        f"{encounters_no_ehr} encounters with no EHR sections"
    )

    # ---- Phase 2: Check cache for LLM HPI polish ----
    log.info("Phase 2: Checking cache for HPI polish...")

    hpi_work_queue: list[tuple[int, str, str]] = []  # (encounter_id, prompt, input_hash)
    hpi_cached: dict[int, str] = {}  # encounter_id -> polished_hpi

    for enc_id, meta in encounter_meta.items():
        if meta["encounter_order"] == 0:
            continue
        if not meta["original_hpi"]:
            continue

        input_hash = _compute_input_hash_9b(enc_id)
        cached = _check_cache_9b(conn, input_hash)
        if cached is not None:
            polished = cached.get("polished_hpi", "")
            if polished:
                hpi_cached[enc_id] = polished
                continue

        # Build prompt
        pid = meta["patient_id"]
        profile = profile_lookup.get(pid, {})
        patient_age = age_lookup.get(pid)

        # Build prior encounters summary
        patient_encounters = enc_by_patient.get(pid, [])
        prior_lines = []
        for pe in patient_encounters:
            pe_order = pe[8]
            if pe_order >= meta["encounter_order"]:
                break
            pe_id = pe[0]
            pe_meta = encounter_meta.get(pe_id, {})
            pe_dx = ", ".join(pe_meta.get("dx_names", [])) or "unspecified"
            prior_lines.append(
                f"Encounter {pe_order + 1} ({pe[2]}, {pe[3]}, {pe[6]}):\n"
                f"  Chief Complaint: {pe[4]}\n"
                f"  Diagnosis: {pe_dx}"
            )

        chronic_str = ", ".join(profile.get("chronic_conditions", [])) or "none"
        home_meds = profile.get("home_medications", [])
        meds_str = ", ".join(
            f"{m.get('name', '')} {m.get('dose', '')}".strip() for m in home_meds
        ) or "none"

        encounter_age = patient_age
        if patient_age is not None:
            first_date = patient_encounters[0][2]
            try:
                fy = int(first_date[:4])
                ey = int(meta["encounter_date"][:4])
                encounter_age = patient_age + (ey - fy)
            except (ValueError, IndexError, TypeError):
                pass

        prompt = HPI_POLISH_PROMPT.format(
            age=encounter_age or "unknown",
            sex="male" if profile.get("sex") == "M" else "female",
            chronic_conditions=chronic_str,
            home_medications=meds_str,
            encounter_number=meta["encounter_order"] + 1,
            total_encounters=len(patient_encounters),
            encounter_date=meta["encounter_date"],
            encounter_type=meta["encounter_type"],
            department=meta["department"] or "",
            chief_complaint=meta["chief_complaint"] or "",
            current_dx=", ".join(meta["dx_names"]) or "unspecified",
            prior_encounters_summary="\n\n".join(prior_lines) or "None",
            original_hpi=meta["original_hpi"],
        )

        hpi_work_queue.append((enc_id, prompt, input_hash))

    log.info(
        f"HPI polish: {len(hpi_cached)} cached, {len(hpi_work_queue)} to generate "
        f"({workers} workers)"
    )

    # ---- Phase 3: Concurrent LLM HPI polish ----
    generated = 0
    errors = 0
    hpi_warnings = 0

    if hpi_work_queue:
        completed_count = 0
        pending = len(hpi_work_queue)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for enc_id, prompt, input_hash in hpi_work_queue:
                pid = encounter_meta[enc_id]["patient_id"]
                future = pool.submit(_polish_hpi_single, enc_id, pid, prompt)
                future_map[future] = (enc_id, input_hash)

            for future in as_completed(future_map):
                enc_id, input_hash = future_map[future]
                result = future.result()
                completed_count += 1

                if result["error"] is None:
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'hpi_polish', ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            STAGE_9B, MODEL, result["in_tok"], result["out_tok"],
                            result["latency_ms"], input_hash,
                            json.dumps({"polished_hpi": result["polished_hpi"]}),
                            result["raw_text"], None,
                        ),
                    )
                    hpi_cached[enc_id] = result["polished_hpi"]
                    generated += 1

                    # Validate
                    orig = encounter_meta[enc_id].get("original_hpi", "")
                    warns = _validate_polished_hpi(orig, result["polished_hpi"])
                    for w in warns:
                        log.warning(f"enc_id={enc_id}: {w}")
                        hpi_warnings += 1
                else:
                    log.error(
                        f"enc_id={enc_id} failed: {result['error']}"
                    )
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'hpi_polish', 0, 0, ?, ?, NULL, NULL, ?)""",
                        (
                            STAGE_9B, MODEL, result["latency_ms"],
                            input_hash, result["error"],
                        ),
                    )
                    errors += 1

                if completed_count % 10 == 0:
                    conn.commit()

                if completed_count % 50 == 0 or completed_count == pending:
                    elapsed = time.time() - t0
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = pending - completed_count
                    eta = remaining / rate if rate > 0 else 0
                    log.info(
                        f"HPI polish {completed_count}/{pending} | "
                        f"generated={generated} errors={errors} | "
                        f"{elapsed:.0f}s elapsed | ETA={eta:.0f}s"
                    )

        conn.commit()

    # ---- Phase 4: Apply polished HPIs + reassemble notes ----
    log.info("Phase 4: Applying polished HPIs and reassembling notes...")
    polished_count = 0

    for enc_id, polished_hpi in hpi_cached.items():
        if not polished_hpi:
            continue

        # Update encounter_ehr_sections HPI
        conn.execute(
            "UPDATE encounter_ehr_sections SET section_text = ?, is_modified = 1 "
            "WHERE encounter_id = ? AND section_type = 'hpi'",
            (polished_hpi, enc_id),
        )

        # Reassemble note with polished HPI
        meta = encounter_meta.get(enc_id)
        if meta:
            sections = dict(meta["note_sections"])
            sections["hpi"] = polished_hpi
            note_text = _assemble_note_text(
                meta["encounter_date"], "", "",
                meta["encounter_type"], meta["chief_complaint"] or "",
                sections,
            )
            # Rebuild header with actual attending/dept
            enc_row = None
            for e in enc_by_patient.get(meta["patient_id"], []):
                if e[0] == enc_id:
                    enc_row = e
                    break
            if enc_row:
                note_text = _assemble_note_text(
                    enc_row[2], enc_row[5] or "", enc_row[6] or "",
                    enc_row[3], enc_row[4] or "", sections,
                )

            conn.execute(
                "UPDATE longitudinal_encounters SET note_text = ?, "
                "generation_method = 'hybrid' WHERE encounter_id = ?",
                (note_text, enc_id),
            )
            polished_count += 1

    conn.commit()

    duration = time.time() - t0
    summary = {
        "step": "9b",
        "patients_processed": len(patient_ids),
        "notes_assembled": total_notes_assembled,
        "sections_inserted": total_sections_inserted,
        "encounters_no_ehr": encounters_no_ehr,
        "hpi_cached": len(hpi_cached) - generated,
        "hpi_generated": generated,
        "hpi_errors": errors,
        "hpi_polished": polished_count,
        "hpi_warnings": hpi_warnings,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Stage 9b complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# 9b Verification
# ---------------------------------------------------------------------------


def verify_9b(conn: sqlite3.Connection):
    """Run verification queries for Stage 9b output."""
    log.info("--- Stage 9b Verification ---")

    total_enc = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters"
    ).fetchone()[0]

    # note_text coverage
    with_note = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters "
        "WHERE note_text IS NOT NULL AND note_text <> ''"
    ).fetchone()[0]
    log.info(f"Encounters with note_text: {with_note}/{total_enc}")

    # encounter_ehr_sections stats
    ees_total = conn.execute(
        "SELECT COUNT(*) FROM encounter_ehr_sections"
    ).fetchone()[0]
    ees_encounters = conn.execute(
        "SELECT COUNT(DISTINCT encounter_id) FROM encounter_ehr_sections"
    ).fetchone()[0]
    log.info(
        f"encounter_ehr_sections: {ees_total} rows, "
        f"{ees_encounters} encounters covered"
    )

    # Section type distribution
    log.info("Section type distribution (encounter_ehr_sections):")
    for stype, cnt in conn.execute(
        "SELECT section_type, COUNT(*) FROM encounter_ehr_sections "
        "GROUP BY section_type ORDER BY COUNT(*) DESC"
    ).fetchall():
        log.info(f"  {stype}: {cnt}")

    # is_modified distribution
    modified = conn.execute(
        "SELECT COUNT(*) FROM encounter_ehr_sections WHERE is_modified = 1"
    ).fetchone()[0]
    unmodified = conn.execute(
        "SELECT COUNT(*) FROM encounter_ehr_sections WHERE is_modified = 0"
    ).fetchone()[0]
    log.info(f"Sections modified: {modified}, unmodified: {unmodified}")

    # generation_method distribution
    log.info("generation_method distribution:")
    for method, cnt in conn.execute(
        "SELECT generation_method, COUNT(*) FROM longitudinal_encounters "
        "GROUP BY generation_method ORDER BY COUNT(*) DESC"
    ).fetchall():
        log.info(f"  {method}: {cnt}")

    # Note length stats
    stats = conn.execute(
        "SELECT AVG(length(note_text)), MIN(length(note_text)), "
        "       MAX(length(note_text)) "
        "FROM longitudinal_encounters "
        "WHERE note_text IS NOT NULL AND note_text <> ''"
    ).fetchone()
    if stats[0]:
        log.info(
            f"Note length (chars): avg={stats[0]:.0f}, "
            f"min={stats[1]}, max={stats[2]}"
        )

    # LLM call stats
    ok_9b = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NULL",
        (STAGE_9B,),
    ).fetchone()[0]
    err_9b = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NOT NULL",
        (STAGE_9B,),
    ).fetchone()[0]
    if ok_9b or err_9b:
        log.info(f"LLM calls (9b): {ok_9b} OK / {err_9b} errors")


# ---------------------------------------------------------------------------
# 9b CSV Export
# ---------------------------------------------------------------------------


def export_csv_9b(conn: sqlite3.Connection, output_dir: Path):
    """Export encounter_ehr_sections to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "s09_encounter_ehr_sections.csv"
    rows = conn.execute(
        """SELECT id, encounter_id, section_type, section_text,
                  source_section_id, is_modified, section_order
           FROM encounter_ehr_sections ORDER BY encounter_id, section_order"""
    ).fetchall()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "encounter_id", "section_type", "section_text",
            "source_section_id", "is_modified", "section_order",
        ])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} encounter_ehr_sections to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 9: Encounter Planning & Note Generation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step", choices=["9a", "9b"],
                       help="Which step to run")
    group.add_argument("--all", action="store_true",
                       help="Run 9a then 9b")
    group.add_argument("--verify-only", action="store_true",
                       help="Run verification queries only")
    group.add_argument("--export-csv", action="store_true",
                       help="Export tables to CSV")

    parser.add_argument("--pilot", type=int, default=0,
                        help="Process only first N patients")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of concurrent LLM workers")
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()
    conn = get_connection(args.db)

    if args.verify_only:
        verify_9a(conn)
        verify_9b(conn)
        conn.close()
        return

    if args.export_csv:
        export_csv_9a(conn, DATA_DIR)
        export_csv_9b(conn, DATA_DIR)
        conn.close()
        return

    pilot = args.pilot if args.pilot > 0 else None

    if args.step == "9a" or args.all:
        summary = run_9a(conn, pilot=pilot, workers=args.workers)
        print(json.dumps(summary, indent=2))

    if args.step == "9b" or args.all:
        summary = run_9b(conn, pilot=pilot, workers=args.workers)
        print(json.dumps(summary, indent=2))

    if args.step == "9a" or args.all:
        verify_9a(conn)
    if args.step == "9b" or args.all:
        verify_9b(conn)

    conn.close()


if __name__ == "__main__":
    main()
