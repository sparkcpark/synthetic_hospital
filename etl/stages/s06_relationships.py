"""Stage 6: Relationship Mapping — question↔diagnosis, question↔finding links.

Populates the Layer 3 junction tables from Stage 5a extraction JSON:
  Step 6a: question_diagnoses + question_findings (deterministic, no LLM)
  Step 6b: diagnosis_findings (LLM classification)
  Step 6c: fact_diagnosis_links + fact_finding_links (LLM linking) [future]

Usage:
    python -m etl.stages.s06_relationships --step 6a [--pilot N]
    python -m etl.stages.s06_relationships --step 6b [--pilot N] [--workers N]
    python -m etl.stages.s06_relationships --verify-only
    python -m etl.stages.s06_relationships --export-csv
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from etl.config import DB_PATH, DATA_DIR
from etl.db import get_connection
from etl.ontology.icd10 import ICD10Dictionary
from etl.stages.s05_ontology import (
    STAGE_5A,
    GATEWAY_URL,
    MODEL,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    TIMEOUT_SECS,
    _call_kimi,
    _call_with_retry,
    _normalize_finding_type,
    _resolve_icd10,
    _make_dx_dedup_key,
    _make_finding_dedup_key,
)
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s06_relationships")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_6A = "s06a_relationships"
STAGE_6B = "s06b_dx_findings"

VALID_RELEVANCE = {"key", "supporting", "background", "distractor"}
VALID_ROLES = {"correct", "distractor", "secondary", "predisposing"}
VALID_RELATIONSHIPS = {
    "pathognomonic", "highly_suggestive", "commonly_seen",
    "risk_factor", "protective", "rules_out",
}

MAX_FINDINGS_PER_CALL = 40  # Cap findings sent per LLM call
DEFAULT_WORKERS_6B = 10

# Step 6c constants
STAGE_6C = "s06c_fact_links"
VALID_DX_RELEVANCE = {"defines", "differentiates", "treatment", "epidemiology", "mechanism"}
VALID_CF_RELEVANCE = {"defines", "explains", "interpretation", "normal_variant"}
BATCH_SIZE_6C = 10          # facts per LLM call (spec Pattern B: ≤20)
MAX_FACT_TEXT_CHARS = 300    # truncate long fact text in prompt
DEFAULT_WORKERS_6C = 10

# ---------------------------------------------------------------------------
# Lookup builders
# ---------------------------------------------------------------------------


def _build_diagnosis_lookup(conn: sqlite3.Connection,
                            icd10_dict: ICD10Dictionary) -> dict[str, int]:
    """Build dedup_key → diagnosis_id mapping (mirrors 5b dedup logic).

    Two lookup paths:
      1. icd10:{code} → diagnosis_id  (for ICD-10-resolved diagnoses)
      2. name:{lower_name} → diagnosis_id  (for unresolved diagnoses)
    """
    rows = conn.execute(
        "SELECT diagnosis_id, display_name, icd10_code FROM diagnoses"
    ).fetchall()

    lookup: dict[str, int] = {}
    for diagnosis_id, display_name, icd10_code in rows:
        key = _make_dx_dedup_key(display_name, icd10_code)
        lookup[key] = diagnosis_id
        # Also add name-based lookup as fallback
        name_key = f"name:{display_name.lower().strip()}"
        if name_key not in lookup:
            lookup[name_key] = diagnosis_id

    log.info(f"Built diagnosis lookup: {len(lookup)} keys for {len(rows)} diagnoses")
    return lookup


def _build_finding_lookup(conn: sqlite3.Connection) -> dict[str, int]:
    """Build (lower_name|finding_type) → finding_id mapping."""
    rows = conn.execute(
        "SELECT finding_id, display_name, finding_type FROM clinical_findings"
    ).fetchall()

    lookup: dict[str, int] = {}
    for finding_id, display_name, finding_type in rows:
        key = _make_finding_dedup_key(display_name, finding_type)
        lookup[key] = finding_id

    log.info(f"Built finding lookup: {len(lookup)} keys for {len(rows)} findings")
    return lookup


# ---------------------------------------------------------------------------
# Extraction loader (deduplicated by question_id)
# ---------------------------------------------------------------------------


def _load_extractions(conn: sqlite3.Connection,
                      limit: int | None = None) -> dict[int, dict]:
    """Load Stage 5a extractions from llm_call_log, deduplicated by question_id.

    For duplicate question_ids (3 known), keeps the latest entry (highest call_id).
    Returns dict mapping question_id → extraction JSON.
    """
    sql = """
        SELECT call_id, output_json
        FROM llm_call_log
        WHERE stage = ? AND error IS NULL
        ORDER BY call_id ASC
    """
    rows = conn.execute(sql, (STAGE_5A,)).fetchall()

    extractions: dict[int, dict] = {}
    parse_errors = 0
    for call_id, output_json in rows:
        try:
            parsed = json.loads(output_json, strict=False)
        except (json.JSONDecodeError, TypeError):
            parse_errors += 1
            continue

        qid = parsed.get("_question_id")
        if qid is None:
            parse_errors += 1
            continue

        # Later call_id overwrites earlier (dedup: keep latest)
        extractions[int(qid)] = parsed

    if parse_errors:
        log.warning(f"Skipped {parse_errors} unparseable llm_call_log entries")

    # Apply limit if specified (for pilot testing)
    if limit and limit < len(extractions):
        qids = sorted(extractions.keys())[:limit]
        extractions = {qid: extractions[qid] for qid in qids}

    log.info(f"Loaded {len(extractions)} unique question extractions")
    return extractions


# ---------------------------------------------------------------------------
# Diagnosis resolution helper
# ---------------------------------------------------------------------------


def _resolve_diagnosis_id(dx_name: str, icd10_suggestion: str | None,
                          icd10_dict: ICD10Dictionary,
                          dx_lookup: dict[str, int]) -> int | None:
    """Resolve a diagnosis name from extraction to a diagnosis_id.

    Mirrors the 5b resolution + dedup logic with fallbacks:
    1. Resolve ICD-10 suggestion → compute dedup key → look up
    2. Try original LLM-suggested code (5b may have stored it as llm_unvalidated)
    3. Fall back to name-based lookup
    """
    # Step 1: Resolve ICD-10 to get the code that 5b would have stored
    code, _, method = _resolve_icd10(dx_name, icd10_suggestion, icd10_dict)

    # Step 2: Compute same dedup key as 5b
    key = _make_dx_dedup_key(dx_name, code)
    diagnosis_id = dx_lookup.get(key)
    if diagnosis_id is not None:
        return diagnosis_id

    # Step 3: Try original LLM suggestion directly (5b stores unvalidated codes)
    if icd10_suggestion and icd10_suggestion != code:
        llm_key = f"icd10:{icd10_suggestion}"
        diagnosis_id = dx_lookup.get(llm_key)
        if diagnosis_id is not None:
            return diagnosis_id

    # Step 4: Fall back to name-only lookup
    name_key = f"name:{dx_name.lower().strip()}"
    return dx_lookup.get(name_key)


# ---------------------------------------------------------------------------
# Value parsing helpers
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_value_numeric(value_text: str | None) -> float | None:
    """Extract leading numeric value from text like '7.2 g/dL' → 7.2."""
    if not value_text:
        return None
    m = _NUMERIC_RE.search(value_text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def _normalize_relevance(rel: str | None) -> str | None:
    """Normalize relevance to valid CHECK constraint values."""
    if not rel:
        return None
    rel_lower = rel.lower().strip()
    if rel_lower in VALID_RELEVANCE:
        return rel_lower
    return None


# ---------------------------------------------------------------------------
# Step 6a: Populate question_diagnoses
# ---------------------------------------------------------------------------


def populate_question_diagnoses(conn: sqlite3.Connection,
                                extractions: dict[int, dict],
                                dx_lookup: dict[str, int],
                                icd10_dict: ICD10Dictionary) -> dict:
    """Insert into question_diagnoses from Stage 5a extractions.

    Returns summary with counts.
    """
    inserted = 0
    skipped_no_match = 0
    skipped_dup = 0

    for qid, ext in extractions.items():
        # Primary diagnosis → role='correct'
        pd = ext.get("primary_diagnosis", {})
        if pd and pd.get("name"):
            dx_id = _resolve_diagnosis_id(
                pd["name"].strip(), pd.get("icd10_suggestion"),
                icd10_dict, dx_lookup
            )
            if dx_id:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO question_diagnoses
                       (question_id, diagnosis_id, role, confidence, source)
                       VALUES (?, ?, 'correct', ?, 'llm')""",
                    (qid, dx_id, pd.get("confidence", 1.0)),
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped_dup += 1
            else:
                skipped_no_match += 1

        # Differential diagnoses → role='distractor'
        for dx in ext.get("differential_diagnoses", []):
            if not dx or not dx.get("name"):
                continue
            dx_id = _resolve_diagnosis_id(
                dx["name"].strip(), dx.get("icd10_suggestion"),
                icd10_dict, dx_lookup
            )
            if dx_id:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO question_diagnoses
                       (question_id, diagnosis_id, role, confidence, source)
                       VALUES (?, ?, 'distractor', ?, 'llm')""",
                    (qid, dx_id, dx.get("confidence", 1.0)),
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped_dup += 1
            else:
                skipped_no_match += 1

        # Secondary diagnoses → role='secondary'
        for dx in ext.get("secondary_diagnoses", []):
            if not dx or not dx.get("name"):
                continue
            dx_id = _resolve_diagnosis_id(
                dx["name"].strip(), dx.get("icd10_suggestion"),
                icd10_dict, dx_lookup
            )
            if dx_id:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO question_diagnoses
                       (question_id, diagnosis_id, role, confidence, source)
                       VALUES (?, ?, 'secondary', ?, 'llm')""",
                    (qid, dx_id, dx.get("confidence", 1.0)),
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped_dup += 1
            else:
                skipped_no_match += 1

    conn.commit()
    summary = {
        "table": "question_diagnoses",
        "inserted": inserted,
        "skipped_no_match": skipped_no_match,
        "skipped_dup": skipped_dup,
    }
    log.info(f"question_diagnoses: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Step 6a: Populate question_findings
# ---------------------------------------------------------------------------


def populate_question_findings(conn: sqlite3.Connection,
                               extractions: dict[int, dict],
                               finding_lookup: dict[str, int]) -> dict:
    """Insert into question_findings from Stage 5a extractions.

    Returns summary with counts.
    """
    inserted = 0
    skipped_no_match = 0
    skipped_dup = 0

    for qid, ext in extractions.items():
        for f in ext.get("clinical_findings", []):
            if not f or not f.get("name"):
                continue

            finding_type = _normalize_finding_type(f.get("finding_type"))
            key = _make_finding_dedup_key(f["name"].strip(), finding_type)
            finding_id = finding_lookup.get(key)

            if not finding_id:
                skipped_no_match += 1
                continue

            present = 1 if f.get("present", True) else 0
            value_text = f.get("value")
            value_numeric = _parse_value_numeric(value_text)
            relevance = _normalize_relevance(f.get("relevance"))

            cur = conn.execute(
                """INSERT OR IGNORE INTO question_findings
                   (question_id, finding_id, present, value_text,
                    value_numeric, relevance, source)
                   VALUES (?, ?, ?, ?, ?, ?, 'llm')""",
                (qid, finding_id, present, value_text,
                 value_numeric, relevance),
            )
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped_dup += 1

    conn.commit()
    summary = {
        "table": "question_findings",
        "inserted": inserted,
        "skipped_no_match": skipped_no_match,
        "skipped_dup": skipped_dup,
    }
    log.info(f"question_findings: {summary}")
    return summary


# ---------------------------------------------------------------------------
# L1/L2/L3 Layer Views
# ---------------------------------------------------------------------------

LAYER_VIEWS_DDL = """
-- L1 Benchmark: 1 primary diagnosis per question (for scoring diagnosis accuracy)
CREATE VIEW IF NOT EXISTS v_benchmark_diagnoses AS
SELECT qd.question_id, qd.diagnosis_id, d.icd10_code, d.display_name,
       qd.confidence
FROM question_diagnoses qd
JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
WHERE qd.role = 'correct';

-- L2 Encounter: All diagnoses per question (for clinical reasoning / fact retrieval)
CREATE VIEW IF NOT EXISTS v_encounter_diagnoses AS
SELECT qd.question_id, qd.diagnosis_id, d.icd10_code, d.display_name,
       qd.role, qd.confidence
FROM question_diagnoses qd
JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id;

-- L3 Concept: ICD-10 3-char categories per question (for guideline logic)
CREATE VIEW IF NOT EXISTS v_concept_diagnoses AS
SELECT DISTINCT qd.question_id,
       SUBSTR(d.icd10_code, 1, 3) AS icd10_category,
       d.category AS diagnosis_category
FROM question_diagnoses qd
JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
WHERE d.icd10_code IS NOT NULL;
"""


def create_layer_views(conn: sqlite3.Connection):
    """Create L1/L2/L3 diagnosis layer views."""
    conn.executescript(LAYER_VIEWS_DDL)
    conn.commit()
    log.info("Created L1/L2/L3 layer views")


# ---------------------------------------------------------------------------
# Step 6a: Main orchestrator
# ---------------------------------------------------------------------------


def run_6a(conn: sqlite3.Connection, pilot: int | None = None) -> dict:
    """Run Step 6a: populate question_diagnoses + question_findings."""
    t0 = time.time()

    # Clear existing data (6a reprocesses all, so start fresh)
    old_qd = conn.execute("SELECT COUNT(*) FROM question_diagnoses").fetchone()[0]
    old_qf = conn.execute("SELECT COUNT(*) FROM question_findings").fetchone()[0]
    if old_qd or old_qf:
        conn.execute("DELETE FROM question_findings")
        conn.execute("DELETE FROM question_diagnoses")
        conn.commit()
        log.info(f"Cleared {old_qd} question_diagnoses + {old_qf} question_findings")

    # Load data
    icd10_dict = ICD10Dictionary()
    dx_lookup = _build_diagnosis_lookup(conn, icd10_dict)
    finding_lookup = _build_finding_lookup(conn)
    extractions = _load_extractions(conn, limit=pilot)

    # Populate tables
    qd_summary = populate_question_diagnoses(conn, extractions, dx_lookup, icd10_dict)
    qf_summary = populate_question_findings(conn, extractions, finding_lookup)

    # Create layer views
    create_layer_views(conn)

    duration = time.time() - t0
    summary = {
        "step": "6a",
        "questions_processed": len(extractions),
        "question_diagnoses": qd_summary,
        "question_findings": qf_summary,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Step 6a complete: {duration:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# Step 6b: diagnosis_findings (LLM classification)
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """You are a board-certified physician classifying clinical finding relationships for a medical knowledge base.

For the diagnosis "{diagnosis_name}" ({icd10_code}), classify each clinical finding's relationship to this diagnosis.

Relationship types:
- pathognomonic: Finding is virtually diagnostic of this condition alone
- highly_suggestive: Strongly associated, high positive predictive value
- commonly_seen: Frequently present but not specific
- risk_factor: Predisposes to or increases probability of this diagnosis
- protective: Decreases probability of this diagnosis
- rules_out: Presence argues strongly against this diagnosis

Findings to classify:
{findings_list}

For each finding, return the relationship type and estimated frequency (0.0-1.0 = how often this finding is present when this diagnosis is present).

Return ONLY valid JSON array (no markdown, no explanation):
[
  {{"finding_id": 123, "relationship": "commonly_seen", "frequency": 0.7}},
  ...
]

Only include findings that have a meaningful clinical relationship to the diagnosis. Skip findings that are unrelated (demographics, unrelated medications, etc.) — do NOT include them in the output.
"""


def _aggregate_cooccurrences(conn: sqlite3.Connection,
                              pilot: int | None = None) -> dict[int, list[dict]]:
    """Aggregate diagnosis↔finding co-occurrences from question_diagnoses + question_findings.

    Returns {diagnosis_id: [{"finding_id", "display_name", "finding_type", "co_count"}, ...]}.
    Only includes correct-role diagnoses for clean signal.
    """
    sql = """
        SELECT qd.diagnosis_id, qf.finding_id,
               cf.display_name, cf.finding_type,
               COUNT(*) as co_count
        FROM question_diagnoses qd
        JOIN question_findings qf ON qd.question_id = qf.question_id
        JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
        WHERE qd.role = 'correct'
        GROUP BY qd.diagnosis_id, qf.finding_id
        ORDER BY qd.diagnosis_id, co_count DESC
    """
    rows = conn.execute(sql).fetchall()

    dx_findings: dict[int, list[dict]] = {}
    for dx_id, finding_id, display_name, finding_type, co_count in rows:
        dx_findings.setdefault(dx_id, []).append({
            "finding_id": finding_id,
            "display_name": display_name,
            "finding_type": finding_type,
            "co_count": co_count,
        })

    if pilot and pilot < len(dx_findings):
        dx_ids = sorted(dx_findings.keys())[:pilot]
        dx_findings = {k: dx_findings[k] for k in dx_ids}

    log.info(f"Aggregated co-occurrences: {len(dx_findings)} diagnoses, "
             f"{sum(len(v) for v in dx_findings.values())} total pairs")
    return dx_findings


def _compute_input_hash_6b(diagnosis_id: int, finding_ids: list[int]) -> str:
    """SHA-256 hash for caching a single diagnosis classification."""
    content = f"{STAGE_6B}:{diagnosis_id}:{','.join(str(f) for f in sorted(finding_ids))}"
    return hashlib.sha256(content.encode()).hexdigest()


def _format_6b_prompt(dx_name: str, icd10_code: str | None,
                      findings: list[dict]) -> str:
    """Format the classification prompt for one diagnosis."""
    findings_text = []
    for i, f in enumerate(findings, 1):
        findings_text.append(
            f"{i}. [id={f['finding_id']}] {f['display_name']} "
            f"(type: {f['finding_type']}, seen in {f['co_count']} question(s))"
        )

    return CLASSIFICATION_PROMPT.format(
        diagnosis_name=dx_name,
        icd10_code=icd10_code or "no ICD-10",
        findings_list="\n".join(findings_text),
    )


def _classify_single(dx_id: int, dx_name: str, prompt: str) -> dict:
    """Worker: classify findings for one diagnosis (thread-safe, no DB)."""
    call_t0 = time.time()
    try:
        parsed, in_tok, out_tok, raw_text = _call_with_retry(prompt)
        latency_ms = int((time.time() - call_t0) * 1000)

        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

        return {"dx_id": dx_id, "dx_name": dx_name, "parsed": parsed,
                "in_tok": in_tok, "out_tok": out_tok, "raw_text": raw_text,
                "latency_ms": latency_ms, "error": None}
    except Exception as e:
        latency_ms = int((time.time() - call_t0) * 1000)
        return {"dx_id": dx_id, "dx_name": dx_name, "parsed": None,
                "in_tok": 0, "out_tok": 0, "raw_text": None,
                "latency_ms": latency_ms, "error": str(e)}


def _normalize_relationship(rel: str | None) -> str | None:
    """Normalize relationship to valid CHECK constraint values."""
    if not rel:
        return None
    rel_lower = rel.lower().strip().replace(" ", "_").replace("-", "_")
    if rel_lower in VALID_RELATIONSHIPS:
        return rel_lower
    # Common LLM variants
    _map = {
        "diagnostic": "pathognomonic",
        "suggestive": "highly_suggestive",
        "strong_association": "highly_suggestive",
        "common": "commonly_seen",
        "frequent": "commonly_seen",
        "risk": "risk_factor",
        "predisposing": "risk_factor",
        "exclusion": "rules_out",
        "excludes": "rules_out",
    }
    return _map.get(rel_lower)


def _check_cache_6b(conn: sqlite3.Connection, input_hash: str) -> dict | None:
    """Check llm_call_log for a cached 6b result."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (STAGE_6B, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def run_6b(conn: sqlite3.Connection, pilot: int | None = None,
           workers: int = DEFAULT_WORKERS_6B) -> dict:
    """Run Step 6b: classify diagnosis↔finding relationships via LLM."""
    t0 = time.time()

    # Clear existing diagnosis_findings
    old_df = conn.execute("SELECT COUNT(*) FROM diagnosis_findings").fetchone()[0]
    if old_df:
        conn.execute("DELETE FROM diagnosis_findings")
        conn.commit()
        log.info(f"Cleared {old_df} diagnosis_findings")

    # Aggregate co-occurrences
    dx_findings = _aggregate_cooccurrences(conn, pilot=pilot)

    # Load diagnosis names/codes
    dx_info = {}
    for row in conn.execute("SELECT diagnosis_id, display_name, icd10_code FROM diagnoses").fetchall():
        dx_info[row[0]] = {"name": row[1], "icd10": row[2]}

    # Build valid finding_id set for validation
    valid_finding_ids = set(
        r[0] for r in conn.execute("SELECT finding_id FROM clinical_findings").fetchall()
    )

    # Phase 1: Check cache, build work queue
    work_queue = []
    cached_results = {}
    for dx_id, findings in dx_findings.items():
        # Cap findings per call
        top_findings = findings[:MAX_FINDINGS_PER_CALL]
        finding_ids = [f["finding_id"] for f in top_findings]
        input_hash = _compute_input_hash_6b(dx_id, finding_ids)

        cached = _check_cache_6b(conn, input_hash)
        if cached is not None:
            cached_results[dx_id] = cached
            continue

        info = dx_info.get(dx_id, {"name": f"dx_{dx_id}", "icd10": None})
        prompt = _format_6b_prompt(info["name"], info["icd10"], top_findings)
        work_queue.append((dx_id, info["name"], prompt, input_hash))

    log.info(f"Step 6b: {len(dx_findings)} diagnoses, "
             f"{len(cached_results)} cached, {len(work_queue)} to classify ({workers} workers)")

    # Phase 2: Concurrent LLM calls
    extracted = 0
    errors = 0
    error_dxs = []

    if work_queue:
        completed_count = 0
        pending = len(work_queue)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for dx_id, dx_name, prompt, input_hash in work_queue:
                future = pool.submit(_classify_single, dx_id, dx_name, prompt)
                future_map[future] = (dx_id, input_hash)

            for future in as_completed(future_map):
                dx_id, input_hash = future_map[future]
                result = future.result()
                completed_count += 1

                if result["error"] is None:
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens, output_tokens,
                            latency_ms, input_hash, output_json, raw_response, error)
                           VALUES (?, ?, 'dx_finding_classify', ?, ?, ?, ?, ?, ?, ?)""",
                        (STAGE_6B, MODEL, result["in_tok"], result["out_tok"],
                         result["latency_ms"], input_hash,
                         json.dumps(result["parsed"]), result["raw_text"], None),
                    )
                    cached_results[dx_id] = result["parsed"]
                    extracted += 1
                else:
                    log.error(f"dx_id={dx_id} ({result['dx_name']}) failed: {result['error']}")
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens, output_tokens,
                            latency_ms, input_hash, output_json, raw_response, error)
                           VALUES (?, ?, 'dx_finding_classify', 0, 0, ?, ?, NULL, NULL, ?)""",
                        (STAGE_6B, MODEL, result["latency_ms"], input_hash, result["error"]),
                    )
                    errors += 1
                    error_dxs.append(dx_id)

                if completed_count % 10 == 0:
                    conn.commit()

                if completed_count % 50 == 0 or completed_count == pending:
                    elapsed = time.time() - t0
                    remaining = pending - completed_count
                    eta = remaining / (completed_count / elapsed) if completed_count > 0 else 0
                    log.info(
                        f"Progress {completed_count}/{pending} | extracted={extracted} "
                        f"errors={errors} | {elapsed:.0f}s elapsed | ETA={eta:.0f}s"
                    )

        conn.commit()

    # Phase 3: Insert into diagnosis_findings
    inserted = 0
    skipped = 0
    for dx_id, classifications in cached_results.items():
        if not isinstance(classifications, list):
            continue
        for item in classifications:
            if not isinstance(item, dict):
                continue
            finding_id = item.get("finding_id")
            if not finding_id or finding_id not in valid_finding_ids:
                skipped += 1
                continue
            relationship = _normalize_relationship(item.get("relationship"))
            if not relationship:
                skipped += 1
                continue
            frequency = item.get("frequency")
            if isinstance(frequency, (int, float)):
                frequency = max(0.0, min(1.0, float(frequency)))
            else:
                frequency = None

            cur = conn.execute(
                """INSERT OR IGNORE INTO diagnosis_findings
                   (diagnosis_id, finding_id, relationship, frequency, evidence_source)
                   VALUES (?, ?, ?, ?, 'board_question')""",
                (dx_id, finding_id, relationship, frequency),
            )
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

    conn.commit()

    duration = time.time() - t0
    summary = {
        "step": "6b",
        "diagnoses_total": len(dx_findings),
        "cached": len(cached_results) - extracted,
        "extracted": extracted,
        "errors": errors,
        "error_dxs": error_dxs[:20],
        "df_inserted": inserted,
        "df_skipped": skipped,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Step 6b complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Step 6c: fact_diagnosis_links + fact_finding_links (LLM extraction)
# ---------------------------------------------------------------------------

FACT_LINKING_PROMPT = """You are a board-certified physician building a medical knowledge graph. For each flashcard fact below, identify which medical diagnoses and clinical findings it describes or relates to.

For each fact card, return:
1. **diagnoses**: Medical diagnoses/conditions the fact is about. For each:
   - name: Standard clinical name (e.g., "Type 2 diabetes mellitus", "Myocardial infarction")
   - relevance: One of: defines, differentiates, treatment, epidemiology, mechanism

2. **findings**: Clinical findings (symptoms, signs, labs, imaging) the fact describes. For each:
   - name: Standard clinical term (e.g., "Elevated troponin", "ST elevation", "Chest pain")
   - finding_type: One of: symptom, sign, lab_value, vital_sign, imaging_finding, procedure_result, history_item, medication, demographic
   - relevance: One of: defines, explains, interpretation, normal_variant

Relevance definitions for diagnoses:
- defines: Fact describes what the diagnosis IS (definition, criteria, classification)
- differentiates: Fact helps distinguish from similar conditions
- treatment: Fact describes management, therapy, or interventions
- epidemiology: Fact describes prevalence, demographics, risk factors
- mechanism: Fact describes pathophysiology or disease mechanism

Relevance definitions for findings:
- defines: Fact defines what this finding is
- explains: Fact explains the mechanism behind this finding
- interpretation: Fact describes how to interpret this finding
- normal_variant: Fact describes when this finding is not pathologic

Return ONLY valid JSON (no markdown, no explanation):
{{
  "results": {{
    "<fact_id>": {{
      "diagnoses": [
        {{"name": "...", "relevance": "..."}}
      ],
      "findings": [
        {{"name": "...", "finding_type": "...", "relevance": "..."}}
      ]
    }}
  }}
}}

Rules:
- Use standard medical terminology for diagnosis/finding names
- Only include genuinely relevant diagnoses/findings (not tangential mentions)
- Return empty arrays if a fact has no relevant diagnoses or findings
- Include the fact_id exactly as given (as a string key)

---
FACTS:
{facts_block}
"""


def _format_facts_block(facts: list[dict]) -> str:
    """Format a batch of fact cards for the 6c prompt."""
    lines = []
    for f in facts:
        text = (f["fact_text"] or "")[:MAX_FACT_TEXT_CHARS]
        context = (f.get("fact_context") or "")[:150]
        topic = f.get("topic") or "Unknown"
        parts = [f"[fact_id: {f['fact_id']}] Topic: {topic}"]
        parts.append(text)
        if context:
            parts.append(f"Context: {context}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _compute_input_hash_6c(fact_ids: list[int]) -> str:
    """SHA-256 hash for caching a 6c batch."""
    content = f"{STAGE_6C}:{','.join(str(fid) for fid in sorted(fact_ids))}"
    return hashlib.sha256(content.encode()).hexdigest()


def _check_cache_6c(conn: sqlite3.Connection, input_hash: str) -> dict | None:
    """Check llm_call_log for a cached 6c result."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log "
        "WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (STAGE_6C, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _extract_links_single(fact_ids: list[int], prompt: str) -> dict:
    """Worker: extract dx/finding links for a batch of facts (thread-safe, no DB)."""
    call_t0 = time.time()
    try:
        parsed, in_tok, out_tok, raw_text = _call_with_retry(prompt)
        latency_ms = int((time.time() - call_t0) * 1000)

        # Accept both {"results": {...}} and bare dict
        results = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(results, dict):
            raise ValueError(f"Expected JSON object with results, got {type(results).__name__}")

        return {"fact_ids": fact_ids, "results": results,
                "in_tok": in_tok, "out_tok": out_tok, "raw_text": raw_text,
                "latency_ms": latency_ms, "error": None}
    except Exception as e:
        latency_ms = int((time.time() - call_t0) * 1000)
        return {"fact_ids": fact_ids, "results": None,
                "in_tok": 0, "out_tok": 0, "raw_text": None,
                "latency_ms": latency_ms, "error": str(e)}


def _normalize_dx_relevance(rel: str | None) -> str | None:
    """Normalize LLM-returned diagnosis relevance to valid CHECK constraint values."""
    if not rel:
        return None
    rel_lower = rel.lower().strip().replace(" ", "_").replace("-", "_")
    if rel_lower in VALID_DX_RELEVANCE:
        return rel_lower
    _map = {
        "definition": "defines",
        "diagnostic_criteria": "defines",
        "classification": "defines",
        "differential": "differentiates",
        "distinction": "differentiates",
        "therapy": "treatment",
        "management": "treatment",
        "pharmacotherapy": "treatment",
        "intervention": "treatment",
        "prevalence": "epidemiology",
        "risk": "epidemiology",
        "demographics": "epidemiology",
        "risk_factor": "epidemiology",
        "pathophysiology": "mechanism",
        "etiology": "mechanism",
        "pathogenesis": "mechanism",
    }
    return _map.get(rel_lower)


def _normalize_cf_relevance(rel: str | None) -> str | None:
    """Normalize LLM-returned finding relevance to valid CHECK constraint values."""
    if not rel:
        return None
    rel_lower = rel.lower().strip().replace(" ", "_").replace("-", "_")
    if rel_lower in VALID_CF_RELEVANCE:
        return rel_lower
    _map = {
        "definition": "defines",
        "mechanism": "explains",
        "pathophysiology": "explains",
        "clinical_significance": "interpretation",
        "significance": "interpretation",
        "normal": "normal_variant",
        "benign": "normal_variant",
        "physiologic": "normal_variant",
    }
    return _map.get(rel_lower)


def _build_dx_name_index(conn: sqlite3.Connection) -> dict[str, int]:
    """Build lowercased display_name → diagnosis_id index for name resolution."""
    rows = conn.execute("SELECT diagnosis_id, display_name FROM diagnoses").fetchall()
    index: dict[str, int] = {}
    for dx_id, name in rows:
        key = name.lower().strip()
        if key not in index:
            index[key] = dx_id
    return index


def _build_cf_name_index(conn: sqlite3.Connection) -> dict[str, list[tuple[int, str]]]:
    """Build lowercased display_name → [(finding_id, finding_type)] index."""
    rows = conn.execute(
        "SELECT finding_id, display_name, finding_type FROM clinical_findings"
    ).fetchall()
    index: dict[str, list[tuple[int, str]]] = {}
    for fid, name, ftype in rows:
        key = name.lower().strip()
        index.setdefault(key, []).append((fid, ftype))
    return index


def _resolve_dx_name(name: str, dx_name_index: dict[str, int],
                     dx_lookup: dict[str, int]) -> int | None:
    """Resolve an LLM-returned diagnosis name to a diagnosis_id.

    Strategy: exact → substring contains → fuzzy (≥0.80) → None.
    """
    key = name.lower().strip()

    # 1. Exact match via name-based lookup (from _build_diagnosis_lookup)
    name_key = f"name:{key}"
    dx_id = dx_lookup.get(name_key)
    if dx_id is not None:
        return dx_id

    # 2. Exact match in name index
    dx_id = dx_name_index.get(key)
    if dx_id is not None:
        return dx_id

    if len(key) < 4:
        return None

    # 3. Substring match: LLM name contained in DB name or vice versa
    #    e.g., "IgA deficiency" matches "Selective IgA deficiency"
    #    e.g., "Patau syndrome" matches "Trisomy 13 (Patau syndrome)"
    best_substr_len = 0
    best_substr_id = None
    for candidate_name, candidate_id in dx_name_index.items():
        if key in candidate_name or candidate_name in key:
            # Prefer the shortest containing match (most specific)
            shorter = min(len(key), len(candidate_name))
            if shorter > best_substr_len:
                best_substr_len = shorter
                best_substr_id = candidate_id
    if best_substr_id is not None:
        return best_substr_id

    # 4. Fuzzy match (SequenceMatcher ≥ 0.80)
    best_ratio = 0.0
    best_id = None
    for candidate_name, candidate_id in dx_name_index.items():
        ratio = SequenceMatcher(None, key, candidate_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = candidate_id
    if best_ratio >= 0.80:
        return best_id
    return None


def _resolve_cf_name(name: str, finding_type_hint: str | None,
                     cf_name_index: dict[str, list[tuple[int, str]]],
                     finding_lookup: dict[str, int]) -> int | None:
    """Resolve an LLM-returned finding name to a finding_id.

    Strategy: dedup key → exact name → substring contains → fuzzy (≥0.80) → None.
    """
    key_lower = name.lower().strip()

    # 1. Try exact dedup key match (name|type) if we have a finding_type
    if finding_type_hint:
        ft = _normalize_finding_type(finding_type_hint)
        if ft:
            dedup_key = _make_finding_dedup_key(name.strip(), ft)
            fid = finding_lookup.get(dedup_key)
            if fid is not None:
                return fid

    # 2. Try name-only match in name index
    candidates = cf_name_index.get(key_lower, [])
    if len(candidates) == 1:
        return candidates[0][0]
    if len(candidates) > 1:
        if finding_type_hint:
            ft = _normalize_finding_type(finding_type_hint)
            for fid, ftype in candidates:
                if ftype == ft:
                    return fid
        return candidates[0][0]

    if len(key_lower) < 4:
        return None

    # 3. Substring match: LLM name contained in DB name or vice versa
    best_substr_len = 0
    best_substr_id = None
    for candidate_name, cand_list in cf_name_index.items():
        if key_lower in candidate_name or candidate_name in key_lower:
            shorter = min(len(key_lower), len(candidate_name))
            if shorter > best_substr_len:
                best_substr_len = shorter
                best_substr_id = cand_list[0][0]
    if best_substr_id is not None:
        return best_substr_id

    # 4. Fuzzy match (SequenceMatcher ≥ 0.80)
    best_ratio = 0.0
    best_id = None
    for candidate_name, cand_list in cf_name_index.items():
        ratio = SequenceMatcher(None, key_lower, candidate_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = cand_list[0][0]
    if best_ratio >= 0.80:
        return best_id
    return None


def run_6c(conn: sqlite3.Connection, pilot: int | None = None,
           workers: int = DEFAULT_WORKERS_6C) -> dict:
    """Run Step 6c: link fact_cards to diagnoses and clinical_findings via LLM."""
    t0 = time.time()

    # Clear existing links (6c deletes + re-inserts from cache, like 6b)
    old_fdl = conn.execute("SELECT COUNT(*) FROM fact_diagnosis_links").fetchone()[0]
    old_ffl = conn.execute("SELECT COUNT(*) FROM fact_finding_links").fetchone()[0]
    if old_fdl or old_ffl:
        conn.execute("DELETE FROM fact_finding_links")
        conn.execute("DELETE FROM fact_diagnosis_links")
        conn.commit()
        log.info(f"Cleared {old_fdl} fact_diagnosis_links + {old_ffl} fact_finding_links")

    # ---- Build lookup maps ----
    icd10_dict = ICD10Dictionary()
    dx_lookup = _build_diagnosis_lookup(conn, icd10_dict)
    finding_lookup = _build_finding_lookup(conn)
    dx_name_index = _build_dx_name_index(conn)
    cf_name_index = _build_cf_name_index(conn)

    log.info(f"Name indexes: {len(dx_name_index)} dx names, {len(cf_name_index)} cf names")

    # ---- Fetch and group fact_cards ----
    sql = """
        SELECT fact_id, fact_text, fact_context, fact_summary,
               subject, organ_system, topic
        FROM fact_cards
        ORDER BY subject, organ_system, fact_id
    """
    all_facts = [
        {"fact_id": r[0], "fact_text": r[1], "fact_context": r[2],
         "fact_summary": r[3], "subject": r[4], "organ_system": r[5], "topic": r[6]}
        for r in conn.execute(sql).fetchall()
    ]

    if pilot and pilot < len(all_facts):
        all_facts = all_facts[:pilot]

    # Group by (subject, organ_system) then split into batches of BATCH_SIZE_6C
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in all_facts:
        key = (f["subject"] or "Unknown", f["organ_system"] or "Unknown")
        groups.setdefault(key, []).append(f)

    batches: list[list[dict]] = []
    for facts in groups.values():
        for i in range(0, len(facts), BATCH_SIZE_6C):
            batches.append(facts[i:i + BATCH_SIZE_6C])

    log.info(f"Step 6c: {len(all_facts)} facts in {len(groups)} groups "
             f"-> {len(batches)} batches")

    # ---- Phase 1: Check cache, build work queue ----
    work_queue: list[tuple[list[int], str, str]] = []
    cached_results: dict[str, dict] = {}

    for batch in batches:
        fact_ids = [f["fact_id"] for f in batch]
        input_hash = _compute_input_hash_6c(fact_ids)

        cached = _check_cache_6c(conn, input_hash)
        if cached is not None:
            cached_results[input_hash] = {"fact_ids": fact_ids, "results": cached}
            continue

        prompt = FACT_LINKING_PROMPT.format(
            facts_block=_format_facts_block(batch)
        )
        work_queue.append((fact_ids, prompt, input_hash))

    log.info(f"  {len(cached_results)} cached, {len(work_queue)} to extract "
             f"({workers} workers)")

    # ---- Phase 2: Concurrent LLM calls ----
    extracted = 0
    errors = 0

    if work_queue:
        completed_count = 0
        pending = len(work_queue)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for fact_ids, prompt, input_hash in work_queue:
                future = pool.submit(_extract_links_single, fact_ids, prompt)
                future_map[future] = (fact_ids, input_hash)

            for future in as_completed(future_map):
                fact_ids, input_hash = future_map[future]
                result = future.result()
                completed_count += 1

                if result["error"] is None:
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens, output_tokens,
                            latency_ms, input_hash, output_json, raw_response, error)
                           VALUES (?, ?, 'fact_link_extract', ?, ?, ?, ?, ?, ?, ?)""",
                        (STAGE_6C, MODEL, result["in_tok"], result["out_tok"],
                         result["latency_ms"], input_hash,
                         json.dumps(result["results"]), result["raw_text"], None),
                    )
                    cached_results[input_hash] = {
                        "fact_ids": fact_ids, "results": result["results"],
                    }
                    extracted += 1
                else:
                    log.error(f"Batch {fact_ids[:3]}... failed: {result['error']}")
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens, output_tokens,
                            latency_ms, input_hash, output_json, raw_response, error)
                           VALUES (?, ?, 'fact_link_extract', 0, 0, ?, ?, NULL, NULL, ?)""",
                        (STAGE_6C, MODEL, result["latency_ms"], input_hash,
                         result["error"]),
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
                        f"Progress {completed_count}/{pending} | extracted={extracted} "
                        f"errors={errors} | {elapsed:.0f}s elapsed | ETA={eta:.0f}s"
                    )

        conn.commit()

    # ---- Phase 3: Name resolution + INSERT ----
    fdl_inserted = 0
    ffl_inserted = 0
    fdl_skipped = 0
    ffl_skipped = 0
    unmatched_dx_names: dict[str, int] = {}
    unmatched_cf_names: dict[str, int] = {}

    for cache_entry in cached_results.values():
        results = cache_entry["results"]
        if not isinstance(results, dict):
            continue

        for fact_id_str, extractions in results.items():
            try:
                fact_id = int(fact_id_str)
            except (ValueError, TypeError):
                continue

            if not isinstance(extractions, dict):
                continue

            # Process diagnosis links
            for dx_item in extractions.get("diagnoses", []):
                if not isinstance(dx_item, dict) or not dx_item.get("name"):
                    continue
                dx_name = dx_item["name"].strip()
                relevance = _normalize_dx_relevance(dx_item.get("relevance"))
                if not relevance:
                    fdl_skipped += 1
                    continue

                dx_id = _resolve_dx_name(dx_name, dx_name_index, dx_lookup)
                if dx_id is None:
                    unmatched_dx_names[dx_name] = unmatched_dx_names.get(dx_name, 0) + 1
                    fdl_skipped += 1
                    continue

                cur = conn.execute(
                    """INSERT OR IGNORE INTO fact_diagnosis_links
                       (fact_id, diagnosis_id, relevance) VALUES (?, ?, ?)""",
                    (fact_id, dx_id, relevance),
                )
                if cur.rowcount > 0:
                    fdl_inserted += 1
                else:
                    fdl_skipped += 1

            # Process finding links
            for cf_item in extractions.get("findings", []):
                if not isinstance(cf_item, dict) or not cf_item.get("name"):
                    continue
                cf_name = cf_item["name"].strip()
                relevance = _normalize_cf_relevance(cf_item.get("relevance"))
                if not relevance:
                    ffl_skipped += 1
                    continue

                finding_type_hint = cf_item.get("finding_type")
                cf_id = _resolve_cf_name(
                    cf_name, finding_type_hint, cf_name_index, finding_lookup
                )
                if cf_id is None:
                    unmatched_cf_names[cf_name] = unmatched_cf_names.get(cf_name, 0) + 1
                    ffl_skipped += 1
                    continue

                cur = conn.execute(
                    """INSERT OR IGNORE INTO fact_finding_links
                       (fact_id, finding_id, relevance) VALUES (?, ?, ?)""",
                    (fact_id, cf_id, relevance),
                )
                if cur.rowcount > 0:
                    ffl_inserted += 1
                else:
                    ffl_skipped += 1

    conn.commit()

    # Log top unmatched names for diagnostics
    if unmatched_dx_names:
        top_dx = sorted(unmatched_dx_names.items(), key=lambda x: -x[1])[:20]
        log.warning(f"Top unmatched diagnoses ({len(unmatched_dx_names)} unique): "
                    f"{top_dx}")
    if unmatched_cf_names:
        top_cf = sorted(unmatched_cf_names.items(), key=lambda x: -x[1])[:20]
        log.warning(f"Top unmatched findings ({len(unmatched_cf_names)} unique): "
                    f"{top_cf}")

    duration = time.time() - t0
    summary = {
        "step": "6c",
        "facts_total": len(all_facts),
        "batches_total": len(batches),
        "cached": len(cached_results) - extracted,
        "extracted": extracted,
        "errors": errors,
        "fdl_inserted": fdl_inserted,
        "fdl_skipped": fdl_skipped,
        "ffl_inserted": ffl_inserted,
        "ffl_skipped": ffl_skipped,
        "unmatched_dx_unique": len(unmatched_dx_names),
        "unmatched_cf_unique": len(unmatched_cf_names),
        "duration_sec": round(duration, 1),
    }
    log.info(f"Step 6c complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(conn: sqlite3.Connection):
    """Run verification queries for Stage 6 output."""
    log.info("--- Stage 6 Verification ---")

    # question_diagnoses
    total_qd = conn.execute("SELECT COUNT(*) FROM question_diagnoses").fetchone()[0]
    log.info(f"question_diagnoses: {total_qd} rows")

    rows = conn.execute(
        "SELECT role, COUNT(*) FROM question_diagnoses GROUP BY role ORDER BY COUNT(*) DESC"
    ).fetchall()
    for role, cnt in rows:
        log.info(f"  role={role}: {cnt}")

    distinct_q = conn.execute(
        "SELECT COUNT(DISTINCT question_id) FROM question_diagnoses"
    ).fetchone()[0]
    log.info(f"  distinct questions: {distinct_q}")

    # question_findings
    total_qf = conn.execute("SELECT COUNT(*) FROM question_findings").fetchone()[0]
    log.info(f"question_findings: {total_qf} rows")

    rows = conn.execute(
        "SELECT relevance, COUNT(*) FROM question_findings GROUP BY relevance ORDER BY COUNT(*) DESC"
    ).fetchall()
    for rel, cnt in rows:
        log.info(f"  relevance={rel}: {cnt}")

    distinct_q = conn.execute(
        "SELECT COUNT(DISTINCT question_id) FROM question_findings"
    ).fetchone()[0]
    log.info(f"  distinct questions: {distinct_q}")

    # L1/L2/L3 views
    for view in ("v_benchmark_diagnoses", "v_encounter_diagnoses", "v_concept_diagnoses"):
        try:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            log.info(f"  {view}: {cnt} rows")
        except sqlite3.OperationalError:
            log.warning(f"  {view}: not created yet")

    # diagnosis_findings
    total_df = conn.execute("SELECT COUNT(*) FROM diagnosis_findings").fetchone()[0]
    if total_df:
        log.info(f"diagnosis_findings: {total_df} rows")
        rows = conn.execute(
            "SELECT relationship, COUNT(*) FROM diagnosis_findings "
            "GROUP BY relationship ORDER BY COUNT(*) DESC"
        ).fetchall()
        for rel, cnt in rows:
            log.info(f"  {rel}: {cnt}")

    # fact links
    total_fdl = conn.execute("SELECT COUNT(*) FROM fact_diagnosis_links").fetchone()[0]
    total_ffl = conn.execute("SELECT COUNT(*) FROM fact_finding_links").fetchone()[0]
    if total_fdl:
        log.info(f"fact_diagnosis_links: {total_fdl} rows")
        rows = conn.execute(
            "SELECT relevance, COUNT(*) FROM fact_diagnosis_links "
            "GROUP BY relevance ORDER BY COUNT(*) DESC"
        ).fetchall()
        for rel, cnt in rows:
            log.info(f"  {rel}: {cnt}")
        distinct_facts = conn.execute(
            "SELECT COUNT(DISTINCT fact_id) FROM fact_diagnosis_links"
        ).fetchone()[0]
        log.info(f"  distinct facts with dx links: {distinct_facts}")
    if total_ffl:
        log.info(f"fact_finding_links: {total_ffl} rows")
        rows = conn.execute(
            "SELECT relevance, COUNT(*) FROM fact_finding_links "
            "GROUP BY relevance ORDER BY COUNT(*) DESC"
        ).fetchall()
        for rel, cnt in rows:
            log.info(f"  {rel}: {cnt}")
        distinct_facts = conn.execute(
            "SELECT COUNT(DISTINCT fact_id) FROM fact_finding_links"
        ).fetchone()[0]
        log.info(f"  distinct facts with cf links: {distinct_facts}")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


def export_csv(conn: sqlite3.Connection, output_dir: Path):
    """Export Stage 6 tables to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # question_diagnoses
    path = output_dir / "s06_question_diagnoses.csv"
    rows = conn.execute(
        """SELECT qd.id, qd.question_id, qd.diagnosis_id,
                  d.display_name, d.icd10_code, qd.role, qd.confidence
           FROM question_diagnoses qd
           JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
           ORDER BY qd.question_id, qd.role"""
    ).fetchall()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question_id", "diagnosis_id", "display_name",
                         "icd10_code", "role", "confidence"])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} question_diagnoses to {path}")

    # question_findings
    path = output_dir / "s06_question_findings.csv"
    rows = conn.execute(
        """SELECT qf.id, qf.question_id, qf.finding_id,
                  cf.display_name, cf.finding_type, qf.present,
                  qf.value_text, qf.value_numeric, qf.relevance
           FROM question_findings qf
           JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
           ORDER BY qf.question_id, qf.relevance"""
    ).fetchall()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question_id", "finding_id", "display_name",
                         "finding_type", "present", "value_text", "value_numeric",
                         "relevance"])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} question_findings to {path}")

    # fact_diagnosis_links
    total_fdl = conn.execute("SELECT COUNT(*) FROM fact_diagnosis_links").fetchone()[0]
    if total_fdl:
        path = output_dir / "s06_fact_diagnosis_links.csv"
        rows = conn.execute(
            """SELECT fdl.id, fdl.fact_id, fdl.diagnosis_id,
                      d.display_name, d.icd10_code, fdl.relevance
               FROM fact_diagnosis_links fdl
               JOIN diagnoses d ON fdl.diagnosis_id = d.diagnosis_id
               ORDER BY fdl.fact_id"""
        ).fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "fact_id", "diagnosis_id", "display_name",
                             "icd10_code", "relevance"])
            writer.writerows(rows)
        log.info(f"Exported {len(rows)} fact_diagnosis_links to {path}")

    # fact_finding_links
    total_ffl = conn.execute("SELECT COUNT(*) FROM fact_finding_links").fetchone()[0]
    if total_ffl:
        path = output_dir / "s06_fact_finding_links.csv"
        rows = conn.execute(
            """SELECT ffl.id, ffl.fact_id, ffl.finding_id,
                      cf.display_name, cf.finding_type, ffl.relevance
               FROM fact_finding_links ffl
               JOIN clinical_findings cf ON ffl.finding_id = cf.finding_id
               ORDER BY ffl.fact_id"""
        ).fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "fact_id", "finding_id", "display_name",
                             "finding_type", "relevance"])
            writer.writerows(rows)
        log.info(f"Exported {len(rows)} fact_finding_links to {path}")


# ---------------------------------------------------------------------------
# run() for main.py integration
# ---------------------------------------------------------------------------


def run(conn: sqlite3.Connection):
    """Run the full Stage 6 pipeline."""
    summary = run_6a(conn)
    verify(conn)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Relationship Mapping — question↔diagnosis/finding links"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step", choices=["6a", "6b", "6c"],
                       help="Which step to run")
    group.add_argument("--all", action="store_true",
                       help="Run all implemented steps")
    group.add_argument("--verify-only", action="store_true",
                       help="Run verification queries only")
    group.add_argument("--export-csv", action="store_true",
                       help="Export tables to CSV")

    parser.add_argument("--pilot", type=int, default=0,
                        help="Process only first N questions (for testing)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent LLM workers (for 6b/6c)")
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()
    conn = get_connection(args.db)

    if args.verify_only:
        verify(conn)
        conn.close()
        return

    if args.export_csv:
        export_csv(conn, DATA_DIR)
        conn.close()
        return

    pilot = args.pilot if args.pilot > 0 else None

    if args.step == "6a" or args.all:
        summary = run_6a(conn, pilot=pilot)
        print(json.dumps(summary, indent=2))

    if args.step == "6b" or args.all:
        summary = run_6b(conn, pilot=pilot, workers=args.workers)
        print(json.dumps(summary, indent=2))

    if args.step == "6c" or args.all:
        summary = run_6c(conn, pilot=pilot, workers=args.workers)
        print(json.dumps(summary, indent=2))

    if args.step or args.all:
        verify(conn)

    conn.close()


if __name__ == "__main__":
    main()
