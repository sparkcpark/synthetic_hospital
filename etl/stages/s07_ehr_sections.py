"""Stage 7: Parse EHR Sections — segment clinical vignettes into labeled EHR sections.

For each board question with a clinical vignette, the LLM segments it into standard
EHR sections (demographics, chief_complaint, hpi, vitals, labs, etc.). Rule-based
regex validates demographics, vitals, and labs as a secondary check.

Usage:
    python -m etl.stages.s07_ehr_sections --run [--pilot N] [--workers N] [--db PATH]
    python -m etl.stages.s07_ehr_sections --verify-only [--db PATH]
    python -m etl.stages.s07_ehr_sections --export-csv [--db PATH]
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

log = get_logger("etl.stages.s07_ehr_sections")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_7 = "s07_ehr_sections"
DEFAULT_WORKERS = 5
MIN_VIGNETTE_CHARS = 50

VALID_SECTION_TYPES = {
    "demographics",
    "chief_complaint",
    "hpi",
    "pmh",
    "psh",
    "medications",
    "allergies",
    "family_history",
    "social_history",
    "ros",
    "vitals",
    "physical_exam",
    "labs",
    "imaging",
    "pathology",
    "other_studies",
    "assessment",
    "plan",
}

# Map common LLM variations to canonical section types
SECTION_TYPE_ALIASES = {
    "history_of_present_illness": "hpi",
    "present_illness": "hpi",
    "history of present illness": "hpi",
    "past_medical_history": "pmh",
    "past medical history": "pmh",
    "medical_history": "pmh",
    "past_surgical_history": "psh",
    "past surgical history": "psh",
    "surgical_history": "psh",
    "review_of_systems": "ros",
    "review of systems": "ros",
    "vital_signs": "vitals",
    "vital signs": "vitals",
    "physical_examination": "physical_exam",
    "physical examination": "physical_exam",
    "exam": "physical_exam",
    "pe": "physical_exam",
    "laboratory": "labs",
    "laboratory_values": "labs",
    "laboratory values": "labs",
    "lab_values": "labs",
    "lab values": "labs",
    "lab": "labs",
    "imaging_studies": "imaging",
    "imaging studies": "imaging",
    "radiology": "imaging",
    "cc": "chief_complaint",
    "chief complaint": "chief_complaint",
    "drug_allergies": "allergies",
    "allergy": "allergies",
    "fh": "family_history",
    "family history": "family_history",
    "sh": "social_history",
    "social history": "social_history",
    "meds": "medications",
    "current_medications": "medications",
    "current medications": "medications",
    "other studies": "other_studies",
    "special_studies": "other_studies",
    "diagnostic_studies": "other_studies",
}

# ---------------------------------------------------------------------------
# Prompt Template (spec §5.5)
# ---------------------------------------------------------------------------

EHR_PARSING_PROMPT = """You are a clinical documentation specialist. Given a USMLE-style clinical vignette, segment it into standard EHR sections.

Return a JSON array of sections, each with:
- "section_type": one of [demographics, chief_complaint, hpi, pmh, psh, medications, allergies, family_history, social_history, ros, vitals, physical_exam, labs, imaging, pathology, other_studies, assessment, plan]
- "text": the extracted text for this section
- "order": integer position (0-indexed)

Rules:
- Every vignette should produce at minimum: demographics, chief_complaint, and at least one additional section
- If information for a section is implied but not explicitly stated, do not create that section
- Preserve the clinical language exactly as written
- Do not add information not present in the original text

Vignette:
\"\"\"
{vignette_text}
\"\"\""""

# ---------------------------------------------------------------------------
# Rule-based regex patterns (validation, not primary extraction)
# ---------------------------------------------------------------------------

RE_DEMOGRAPHICS = re.compile(
    r"(\d{1,3})[\s\-]year[\s\-]old\s+"
    r"(man|woman|male|female|boy|girl|infant|newborn|child|adolescent|patient)",
    re.IGNORECASE,
)

RE_VITALS_TEMP = re.compile(
    r"temperature\s+(?:of\s+)?(\d{2,3}(?:\.\d+)?)\s*°?\s*[CF]", re.IGNORECASE
)
RE_VITALS_BP = re.compile(
    r"blood\s+pressure\s+(?:is\s+|of\s+)?(\d{2,3}/\d{2,3})", re.IGNORECASE
)
RE_VITALS_PULSE = re.compile(
    r"(?:pulse|heart\s+rate)\s+(?:is\s+|of\s+)?(\d{2,3})\s*(?:/min|bpm|beats)",
    re.IGNORECASE,
)
RE_VITALS_RR = re.compile(
    r"(?:respiratory\s+rate|respirations?)\s+(?:is\s+|of\s+|are\s+)?(\d{1,2})\s*(?:/min|breaths)",
    re.IGNORECASE,
)

RE_LABS = re.compile(
    r"(?:hemoglobin|hematocrit|WBC|white\s+blood\s+cell|platelet|creatinine|BUN|glucose|"
    r"sodium|potassium|chloride|bicarbonate|calcium|albumin|bilirubin|ALT|AST|"
    r"alkaline\s+phosphatase|troponin|BNP|TSH|HbA1c|INR|PT|PTT|ESR|CRP|"
    r"lactate|amylase|lipase|uric\s+acid)\s+"
    r"(?:is\s+|of\s+|level\s+(?:is\s+|of\s+)?)?"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _compute_input_hash(qid: int, vignette_text: str) -> str:
    """SHA-256 hash for caching a Stage 7 LLM call."""
    vignette_hash = hashlib.sha256(vignette_text.encode()).hexdigest()[:16]
    content = f"{STAGE_7}:{qid}:{vignette_hash}"
    return hashlib.sha256(content.encode()).hexdigest()


def _check_cache(conn: sqlite3.Connection, input_hash: str) -> dict | None:
    """Check llm_call_log for a cached Stage 7 result."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log "
        "WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (STAGE_7, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _normalize_section_type(raw: str) -> str | None:
    """Normalize an LLM-returned section type to the canonical enum value."""
    cleaned = raw.strip().lower()
    if cleaned in VALID_SECTION_TYPES:
        return cleaned
    if cleaned in SECTION_TYPE_ALIASES:
        return SECTION_TYPE_ALIASES[cleaned]
    return None


def _validate_sections(sections: list) -> list[dict]:
    """Validate and normalize a list of section dicts from LLM output.

    Returns only valid sections with canonical section_type, non-empty text,
    and sequential order.
    """
    valid = []
    for item in sections:
        if not isinstance(item, dict):
            continue
        raw_type = item.get("section_type", "")
        normalized = _normalize_section_type(str(raw_type))
        if normalized is None:
            log.debug(f"Dropping unknown section_type: {raw_type!r}")
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        valid.append({
            "section_type": normalized,
            "text": text,
        })

    # Re-assign sequential order
    for i, sec in enumerate(valid):
        sec["order"] = i

    return valid


def _rule_based_validate(vignette_text: str, sections: list[dict]) -> list[str]:
    """Check LLM sections against regex patterns. Returns list of warnings."""
    warnings = []
    section_types = {s["section_type"] for s in sections}

    # Demographics check
    if "demographics" not in section_types and RE_DEMOGRAPHICS.search(vignette_text):
        warnings.append("LLM missed demographics (regex found age/sex pattern)")

    # Vitals check
    has_vitals = any([
        RE_VITALS_TEMP.search(vignette_text),
        RE_VITALS_BP.search(vignette_text),
        RE_VITALS_PULSE.search(vignette_text),
        RE_VITALS_RR.search(vignette_text),
    ])
    if "vitals" not in section_types and has_vitals:
        warnings.append("LLM missed vitals (regex found vital sign patterns)")

    # Labs check
    if "labs" not in section_types and RE_LABS.search(vignette_text):
        warnings.append("LLM missed labs (regex found lab value patterns)")

    return warnings


def _parse_single(qid: int, vignette_text: str) -> dict:
    """Thread-safe LLM worker: parse one vignette into EHR sections.

    No database access inside this function.
    """
    prompt = EHR_PARSING_PROMPT.format(vignette_text=vignette_text)
    call_t0 = time.time()
    try:
        parsed, in_tok, out_tok, raw_text = _call_with_retry(prompt)
        latency_ms = int((time.time() - call_t0) * 1000)

        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

        return {
            "qid": qid,
            "sections": parsed,
            "in_tok": in_tok,
            "out_tok": out_tok,
            "raw_text": raw_text,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        latency_ms = int((time.time() - call_t0) * 1000)
        return {
            "qid": qid,
            "sections": None,
            "in_tok": 0,
            "out_tok": 0,
            "raw_text": None,
            "latency_ms": latency_ms,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def run_7(
    conn: sqlite3.Connection,
    pilot: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Run Stage 7: parse EHR sections from clinical vignettes.

    Phase 1: Fetch questions, check cache, build work queue
    Phase 2: Concurrent LLM extraction
    Phase 3: Validate + insert into ehr_sections
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Fetch questions, check cache
    # ------------------------------------------------------------------
    query = (
        "SELECT question_id, vignette_text FROM board_questions "
        "WHERE vignette_text IS NOT NULL "
        "ORDER BY question_id"
    )
    rows = conn.execute(query).fetchall()
    if pilot:
        rows = rows[:pilot]

    # Filter out very short vignettes
    skipped_short = 0
    questions = []
    for qid, vtext in rows:
        if len(vtext) < MIN_VIGNETTE_CHARS:
            log.warning(f"Skipping qid={qid}: vignette too short ({len(vtext)} chars)")
            skipped_short += 1
            continue
        questions.append((qid, vtext))

    # Check cache
    work_queue = []
    cached_hashes = {}  # input_hash → qid
    for qid, vtext in questions:
        input_hash = _compute_input_hash(qid, vtext)
        cached = _check_cache(conn, input_hash)
        if cached is not None:
            cached_hashes[input_hash] = (qid, cached)
            continue
        work_queue.append((qid, vtext, input_hash))

    log.info(
        f"Stage 7: {len(questions)} questions, "
        f"{len(cached_hashes)} cached, {len(work_queue)} to extract "
        f"({workers} workers), {skipped_short} skipped (too short)"
    )

    # ------------------------------------------------------------------
    # Phase 2: Concurrent LLM extraction
    # ------------------------------------------------------------------
    extracted = 0
    errors = 0
    error_qids = []

    if work_queue:
        completed_count = 0
        pending = len(work_queue)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for qid, vtext, input_hash in work_queue:
                future = pool.submit(_parse_single, qid, vtext)
                future_map[future] = (qid, input_hash)

            for future in as_completed(future_map):
                qid, input_hash = future_map[future]
                result = future.result()
                completed_count += 1

                if result["error"] is None:
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'ehr_section_parse', ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            STAGE_7, MODEL, result["in_tok"], result["out_tok"],
                            result["latency_ms"], input_hash,
                            json.dumps(result["sections"]),
                            result["raw_text"], None,
                        ),
                    )
                    cached_hashes[input_hash] = (qid, result["sections"])
                    extracted += 1
                else:
                    log.error(f"qid={qid} failed: {result['error']}")
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'ehr_section_parse', 0, 0, ?, ?, NULL, NULL, ?)""",
                        (
                            STAGE_7, MODEL, result["latency_ms"],
                            input_hash, result["error"],
                        ),
                    )
                    errors += 1
                    error_qids.append(qid)

                if completed_count % 10 == 0:
                    conn.commit()

                if completed_count % 100 == 0 or completed_count == pending:
                    elapsed = time.time() - t0
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = pending - completed_count
                    eta = remaining / rate if rate > 0 else 0
                    log.info(
                        f"Progress {completed_count}/{pending} | "
                        f"extracted={extracted} errors={errors} | "
                        f"{elapsed:.0f}s elapsed | ETA={eta:.0f}s"
                    )

        conn.commit()

    # ------------------------------------------------------------------
    # Phase 3: Validate + insert into ehr_sections
    # ------------------------------------------------------------------
    old_count = conn.execute("SELECT COUNT(*) FROM ehr_sections").fetchone()[0]
    if old_count:
        conn.execute("DELETE FROM ehr_sections")
        conn.commit()
        log.info(f"Cleared {old_count} existing ehr_sections rows")

    inserted = 0
    questions_with_sections = 0
    questions_lt3 = 0
    rule_warnings = 0
    type_dist: dict[str, int] = {}

    for input_hash, (qid, raw_sections) in cached_hashes.items():
        if not isinstance(raw_sections, list):
            continue

        sections = _validate_sections(raw_sections)
        if not sections:
            log.warning(f"qid={qid}: LLM returned 0 valid sections")
            continue

        questions_with_sections += 1
        if len(sections) < 3:
            questions_lt3 += 1
            log.debug(f"qid={qid}: only {len(sections)} sections")

        # Find vignette_text for rule-based validation
        vtext_row = conn.execute(
            "SELECT vignette_text FROM board_questions WHERE question_id = ?",
            (qid,),
        ).fetchone()
        if vtext_row:
            warnings = _rule_based_validate(vtext_row[0], sections)
            for w in warnings:
                log.debug(f"qid={qid}: {w}")
                rule_warnings += 1

        for sec in sections:
            conn.execute(
                """INSERT INTO ehr_sections
                   (question_id, section_type, section_text, section_order,
                    extraction_method)
                   VALUES (?, ?, ?, ?, 'llm')""",
                (qid, sec["section_type"], sec["text"], sec["order"]),
            )
            inserted += 1
            type_dist[sec["section_type"]] = type_dist.get(sec["section_type"], 0) + 1

    conn.commit()

    duration = time.time() - t0
    total_questions = len(questions)
    coverage_pct = (
        questions_with_sections / total_questions * 100 if total_questions else 0
    )
    gte3_pct = (
        (questions_with_sections - questions_lt3) / total_questions * 100
        if total_questions else 0
    )

    summary = {
        "stage": "7",
        "questions_total": total_questions,
        "skipped_short": skipped_short,
        "cached": len(cached_hashes) - extracted,
        "extracted": extracted,
        "errors": errors,
        "error_qids": error_qids[:20],
        "sections_inserted": inserted,
        "questions_with_sections": questions_with_sections,
        "coverage_pct": round(coverage_pct, 1),
        "questions_gte3_sections": questions_with_sections - questions_lt3,
        "gte3_pct": round(gte3_pct, 1),
        "rule_warnings": rule_warnings,
        "section_type_distribution": dict(
            sorted(type_dist.items(), key=lambda x: -x[1])
        ),
        "duration_sec": round(duration, 1),
    }
    log.info(f"Stage 7 complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(conn: sqlite3.Connection):
    """Run verification queries for Stage 7 output."""
    log.info("--- Stage 7 Verification ---")

    total = conn.execute("SELECT COUNT(*) FROM ehr_sections").fetchone()[0]
    log.info(f"ehr_sections: {total} rows")

    if total == 0:
        log.warning("No ehr_sections rows — run Stage 7 first")
        return

    # Section type distribution
    log.info("Section type distribution:")
    rows = conn.execute(
        "SELECT section_type, COUNT(*) FROM ehr_sections "
        "GROUP BY section_type ORDER BY COUNT(*) DESC"
    ).fetchall()
    for stype, cnt in rows:
        log.info(f"  {stype}: {cnt}")

    # Extraction method distribution
    log.info("Extraction method distribution:")
    rows = conn.execute(
        "SELECT extraction_method, COUNT(*) FROM ehr_sections "
        "GROUP BY extraction_method ORDER BY COUNT(*) DESC"
    ).fetchall()
    for method, cnt in rows:
        log.info(f"  {method}: {cnt}")

    # Question coverage
    total_questions = conn.execute(
        "SELECT COUNT(*) FROM board_questions"
    ).fetchone()[0]
    questions_with = conn.execute(
        "SELECT COUNT(DISTINCT question_id) FROM ehr_sections"
    ).fetchone()[0]
    log.info(
        f"Question coverage: {questions_with}/{total_questions} "
        f"({questions_with / total_questions * 100:.1f}%)"
    )

    # Sections per question distribution
    log.info("Sections per question:")
    rows = conn.execute(
        "SELECT sections_per_q, COUNT(*) FROM ("
        "  SELECT question_id, COUNT(*) AS sections_per_q "
        "  FROM ehr_sections GROUP BY question_id"
        ") GROUP BY sections_per_q ORDER BY sections_per_q"
    ).fetchall()
    for n_sections, n_questions in rows:
        log.info(f"  {n_sections} sections: {n_questions} questions")

    # Questions with >=3 sections
    gte3 = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT question_id FROM ehr_sections "
        "  GROUP BY question_id HAVING COUNT(*) >= 3"
        ")"
    ).fetchone()[0]
    log.info(
        f"Questions with >=3 sections: {gte3}/{total_questions} "
        f"({gte3 / total_questions * 100:.1f}%)"
    )

    # Average sections per question
    avg = conn.execute(
        "SELECT AVG(cnt) FROM ("
        "  SELECT COUNT(*) AS cnt FROM ehr_sections GROUP BY question_id"
        ")"
    ).fetchone()[0]
    log.info(f"Average sections per question: {avg:.1f}")

    # LLM call log stats
    ok = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NULL",
        (STAGE_7,),
    ).fetchone()[0]
    err = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NOT NULL",
        (STAGE_7,),
    ).fetchone()[0]
    log.info(f"LLM calls: {ok} OK / {err} errors")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


def export_csv(conn: sqlite3.Connection, output_dir: Path):
    """Export ehr_sections to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "s07_ehr_sections.csv"
    rows = conn.execute(
        """SELECT es.section_id, es.question_id, es.section_type,
                  es.section_text, es.section_order, es.extraction_method
           FROM ehr_sections es
           ORDER BY es.question_id, es.section_order"""
    ).fetchall()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "section_id", "question_id", "section_type",
            "section_text", "section_order", "extraction_method",
        ])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} ehr_sections to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: EHR Section Parsing"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run EHR section parsing")
    group.add_argument("--verify-only", action="store_true", help="Run verification only")
    group.add_argument("--export-csv", action="store_true", help="Export to CSV only")

    parser.add_argument("--pilot", type=int, default=0, help="Limit to first N questions")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="LLM workers")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")

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
    summary = run_7(conn, pilot=pilot, workers=args.workers)
    print(json.dumps(summary, indent=2))

    verify(conn)
    conn.close()


if __name__ == "__main__":
    main()
