"""Stage 5: Ontology Resolution — diagnoses + clinical_findings.

Extracts diagnoses and clinical findings from board questions via LLM (Kimi k2.5),
maps diagnoses to ICD-10-CM codes using a local CMS dictionary, and populates
the `diagnoses` and `clinical_findings` tables.

Step 5a: LLM extraction (1 question per call, cached in llm_call_log)
Step 5b: ICD-10 mapping + deduplication + insertion

Usage:
    python -m etl.stages.s05_ontology --extract [--pilot N]
    python -m etl.stages.s05_ontology --map [--pilot N]
    python -m etl.stages.s05_ontology --all [--pilot N]
    python -m etl.stages.s05_ontology --verify-only
    python -m etl.stages.s05_ontology --export-csv
"""

import os
import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from etl.config import DB_PATH, DATA_DIR
from etl.db import get_connection
from etl.ontology.icd10 import ICD10Dictionary, ICD10Code
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s05_ontology")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic-compatible LLM gateway used to build the benchmark (not released); see README.
GATEWAY_URL = os.environ.get("SH_LLM_GATEWAY_URL", "http://localhost:8080")
MODEL = "kimi-k2.5"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
TIMEOUT_SECS = 360.0

STAGE_5A = "s05a_extract"
STAGE_5B = "s05b_map"
DEFAULT_WORKERS = 30

MAX_VIGNETTE_CHARS = 2000
MAX_EXPLANATION_CHARS = 500
MAX_DISTRACTOR_CHARS = 200

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a board-certified physician reviewing USMLE questions for a clinical AI benchmark.

Given this clinical vignette and question, extract:
1. PRIMARY DIAGNOSIS — what the correct answer points to
2. DIFFERENTIAL DIAGNOSES — what each distractor answer choice represents (include ALL choices)
3. SECONDARY DIAGNOSES — comorbidities, risk factors, or predisposing conditions mentioned in the vignette that are NOT answer choices but are clinically relevant (e.g., "PMH significant for HTN and DM2")
4. CLINICAL FINDINGS — all symptoms, signs, lab values, vitals, imaging findings, medications, and history items mentioned

For each diagnosis provide:
- name: standard clinical name
- icd10_suggestion: best ICD-10-CM code (e.g., "I21.01")
- snomed_suggestion: SNOMED CT concept ID if known, else null
- category: one of: Cardiovascular, Infectious, Neoplastic, Endocrine, Neurological, Psychiatric, Pulmonary, Gastrointestinal, Renal, Hematologic, Musculoskeletal, Dermatologic, Obstetric, Metabolic, Immunologic, Genetic, Traumatic, Toxic, Ophthalmologic, ENT, Other
- acuity: acute | chronic | acute_on_chronic | unspecified
- confidence: 0.0-1.0

For each clinical finding provide:
- name: standard clinical term
- finding_type: one of: symptom | sign | lab_value | vital_sign | imaging_finding | procedure_result | history_item | medication | demographic
- value: the specific value mentioned (e.g., "140/90 mmHg", "hemoglobin 7.2 g/dL", "45-year-old"), or null
- present: true if the finding IS present, false if explicitly absent
- relevance: key | supporting | background | distractor
- snomed_suggestion: SNOMED CT concept ID if known, else null

Return ONLY valid JSON (no markdown, no explanation):
{
  "primary_diagnosis": {
    "name": "...",
    "icd10_suggestion": "...",
    "snomed_suggestion": null,
    "category": "...",
    "acuity": "...",
    "confidence": 0.9
  },
  "differential_diagnoses": [
    {"name": "...", "icd10_suggestion": "...", "snomed_suggestion": null, "category": "...", "acuity": "...", "confidence": 0.8}
  ],
  "secondary_diagnoses": [
    {"name": "...", "icd10_suggestion": "...", "snomed_suggestion": null, "category": "...", "acuity": "...", "confidence": 0.7}
  ],
  "clinical_findings": [
    {"name": "...", "finding_type": "...", "value": null, "present": true, "relevance": "key", "snomed_suggestion": null}
  ]
}

---
"""

# ---------------------------------------------------------------------------
# Normalization maps
# ---------------------------------------------------------------------------

VALID_FINDING_TYPES = {
    "symptom", "sign", "lab_value", "vital_sign", "imaging_finding",
    "procedure_result", "history_item", "medication", "demographic",
}

_FINDING_TYPE_NORMALIZE: dict[str, str] = {
    "lab": "lab_value", "labs": "lab_value", "laboratory": "lab_value",
    "lab_result": "lab_value", "laboratory_value": "lab_value",
    "vital": "vital_sign", "vitals": "vital_sign", "vital_signs": "vital_sign",
    "imaging": "imaging_finding", "radiology": "imaging_finding",
    "radiologic_finding": "imaging_finding", "xray": "imaging_finding",
    "procedure": "procedure_result", "pathology": "procedure_result",
    "biopsy": "procedure_result",
    "history": "history_item", "pmh": "history_item",
    "social_history": "history_item", "family_history": "history_item",
    "past_medical_history": "history_item",
    "med": "medication", "drug": "medication", "medications": "medication",
    "demographics": "demographic", "age": "demographic", "sex": "demographic",
    "gender": "demographic",
}

_ACUITY_NORMALIZE: dict[str, str] = {
    "acute": "acute", "chronic": "chronic",
    "acute on chronic": "acute_on_chronic",
    "acute_on_chronic": "acute_on_chronic",
    "acute-on-chronic": "acute_on_chronic",
    "subacute": "acute",
    "unspecified": "unspecified", "": "unspecified",
}

CANONICAL_CATEGORIES = {
    "cardiovascular", "infectious", "neoplastic", "endocrine", "neurological",
    "psychiatric", "pulmonary", "gastrointestinal", "renal", "hematologic",
    "musculoskeletal", "dermatologic", "obstetric", "metabolic", "immunologic",
    "genetic", "traumatic", "toxic", "ophthalmologic", "ent", "other",
}

_CATEGORY_NORMALIZE: dict[str, str] = {
    "cardiac": "cardiovascular", "heart": "cardiovascular",
    "cardiology": "cardiovascular",
    "infection": "infectious", "infectious disease": "infectious",
    "cancer": "neoplastic", "oncologic": "neoplastic", "tumor": "neoplastic",
    "oncology": "neoplastic",
    "lung": "pulmonary", "respiratory": "pulmonary",
    "gi": "gastrointestinal", "digestive": "gastrointestinal",
    "kidney": "renal", "urologic": "renal", "nephrology": "renal",
    "blood": "hematologic", "hematology": "hematologic",
    "bone": "musculoskeletal", "orthopedic": "musculoskeletal",
    "skin": "dermatologic", "dermatology": "dermatologic",
    "ob/gyn": "obstetric", "reproductive": "obstetric", "gynecologic": "obstetric",
    "mental": "psychiatric", "behavioral": "psychiatric", "psychiatry": "psychiatric",
    "neuro": "neurological", "neurology": "neurological",
    "autoimmune": "immunologic", "allergy": "immunologic", "immune": "immunologic",
    "eye": "ophthalmologic", "ophthalmic": "ophthalmologic",
    "ear": "ent", "otolaryngology": "ent",
    "poisoning": "toxic", "overdose": "toxic", "toxicology": "toxic",
    "congenital": "genetic", "hereditary": "genetic",
    "injury": "traumatic", "trauma": "traumatic",
}


def _normalize_finding_type(ft: str | None) -> str:
    """Normalize finding_type to valid CHECK constraint values."""
    if not ft:
        return "sign"
    ft_lower = ft.lower().strip()
    if ft_lower in VALID_FINDING_TYPES:
        return ft_lower
    normalized = _FINDING_TYPE_NORMALIZE.get(ft_lower)
    if normalized:
        return normalized
    return "sign"


def _normalize_acuity(acuity: str | None) -> str:
    """Normalize acuity to CHECK constraint values."""
    if not acuity:
        return "unspecified"
    normalized = _ACUITY_NORMALIZE.get(acuity.lower().strip())
    return normalized if normalized else "unspecified"


def _normalize_category(category: str | None) -> str | None:
    """Normalize category to canonical values."""
    if not category:
        return None
    cat_lower = category.lower().strip()
    if cat_lower in CANONICAL_CATEGORIES:
        return cat_lower.capitalize() if cat_lower != "ent" else "ENT"
    normalized = _CATEGORY_NORMALIZE.get(cat_lower)
    if normalized:
        return normalized.capitalize() if normalized != "ent" else "ENT"
    return category.strip()


# ---------------------------------------------------------------------------
# LLM client (reuse s04c patterns)
# ---------------------------------------------------------------------------

def _format_question_input(q: dict) -> str:
    """Format a single board question for the extraction prompt."""
    parts = []

    if q.get("question_stem"):
        parts.append(f"Question: {q['question_stem']}")

    if q.get("vignette_text"):
        vignette = q["vignette_text"][:MAX_VIGNETTE_CHARS]
        parts.append(f"\nClinical Vignette:\n{vignette}")

    if q.get("answer_choices"):
        try:
            choices = json.loads(q["answer_choices"])
            correct = q.get("correct_answer", "")
            parts.append("\nAnswer Choices:")
            for c in choices:
                marker = " [CORRECT]" if c.get("letter") == correct else ""
                parts.append(f"  {c.get('letter', '?')}. {c.get('text', '')}{marker}")
        except (json.JSONDecodeError, TypeError):
            pass

    if q.get("correct_explanation"):
        expl = q["correct_explanation"][:MAX_EXPLANATION_CHARS]
        parts.append(f"\nCorrect Answer Explanation:\n{expl}")

    if q.get("distractor_explanations"):
        try:
            distractors = json.loads(q["distractor_explanations"])
            if isinstance(distractors, dict) and distractors:
                parts.append("\nDistractor Explanations:")
                for letter, expl in list(distractors.items())[:4]:
                    parts.append(f"  {letter}: {str(expl)[:MAX_DISTRACTOR_CHARS]}")
        except (json.JSONDecodeError, TypeError):
            pass

    return "\n".join(parts)


def _compute_input_hash_5a(question_id: int, question_text: str) -> str:
    """SHA-256 hash for caching a single question extraction."""
    content = f"{STAGE_5A}:{question_id}:{question_text[:200]}"
    return hashlib.sha256(content.encode()).hexdigest()


def _check_cache(conn, input_hash: str, stage: str) -> dict | None:
    """Check llm_call_log for a cached result."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (stage, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _call_kimi(prompt: str, timeout: float = TIMEOUT_SECS) -> tuple[dict, int, int, str]:
    """Send a prompt to Kimi k2.5 and return (parsed_json, input_tokens, output_tokens, raw_text)."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{GATEWAY_URL}/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

    resp_json = json.loads(resp.text, strict=False)

    # Extract text block (skip thinking blocks)
    raw_text = ""
    for item in resp_json.get("content", []):
        if item.get("type") == "text":
            raw_text = item["text"]
            break

    if not raw_text:
        raise ValueError("No text block found in Kimi response")

    usage = resp_json.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Parse JSON (strip markdown code fences if present)
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_text = "\n".join(lines)

    parsed = json.loads(json_text, strict=False)
    return parsed, input_tokens, output_tokens, raw_text


def _call_with_retry(prompt: str) -> tuple[dict, int, int, str]:
    """Call Kimi with exponential backoff retry."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_kimi(prompt, timeout=TIMEOUT_SECS + (attempt * 60))
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code in (429, 529) or e.response.status_code >= 500:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(f"HTTP {e.response.status_code}, retrying in {delay:.0f}s (attempt {attempt + 1})")
                time.sleep(delay)
            else:
                raise
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Parse error: {e}, retrying (attempt {attempt + 1})")
                time.sleep(RETRY_BASE_DELAY)
            else:
                raise
        except httpx.TimeoutException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Timeout, retrying with longer timeout (attempt {attempt + 1})")
                time.sleep(RETRY_BASE_DELAY)
            else:
                raise
    raise last_error


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _log_llm_call(conn, stage: str, input_hash: str, input_tokens: int,
                   output_tokens: int, latency_ms: int,
                   output_json: str | None, raw_response: str | None,
                   error: str | None = None):
    """Log an LLM call to llm_call_log."""
    conn.execute(
        """INSERT INTO llm_call_log
           (stage, model, prompt_template, input_tokens, output_tokens,
            latency_ms, input_hash, output_json, raw_response, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (stage, MODEL, "ontology_extract" if stage == STAGE_5A else "icd10_disambiguate",
         input_tokens, output_tokens, latency_ms, input_hash,
         output_json, raw_response, error),
    )


def _log_processing(conn, stage: str, status: str, records_in: int = 0,
                     records_out: int = 0, records_error: int = 0,
                     duration_sec: float = 0.0):
    """Log to processing_log."""
    if status == "started":
        conn.execute(
            "INSERT INTO processing_log (stage, status, records_in) VALUES (?, 'started', ?)",
            (stage, records_in),
        )
    else:
        conn.execute(
            """UPDATE processing_log
               SET status = ?, records_out = ?, records_error = ?,
                   duration_sec = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
               WHERE log_id = (
                   SELECT log_id FROM processing_log
                   WHERE stage = ? AND status = 'started'
                   ORDER BY log_id DESC LIMIT 1
               )""",
            (status, records_out, records_error, duration_sec, stage),
        )
    conn.commit()


def _fetch_questions(conn, limit: int | None = None) -> list[dict]:
    """Fetch board_questions for extraction."""
    sql = """
        SELECT question_id, question_format, vignette_text, question_stem,
               answer_choices, correct_answer, correct_explanation,
               distractor_explanations, subject, organ_system, topic
        FROM board_questions
        ORDER BY question_id
    """
    if limit:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    cols = ["question_id", "question_format", "vignette_text", "question_stem",
            "answer_choices", "correct_answer", "correct_explanation",
            "distractor_explanations", "subject", "organ_system", "topic"]
    return [dict(zip(cols, row)) for row in rows]


# ---------------------------------------------------------------------------
# Step 5a: LLM Extraction
# ---------------------------------------------------------------------------

def _validate_extraction(parsed: dict) -> bool:
    """Validate the structure of an LLM extraction result. Returns True if valid."""
    if "primary_diagnosis" not in parsed:
        return False
    pd = parsed["primary_diagnosis"]
    if not isinstance(pd, dict) or not pd.get("name"):
        return False
    if "clinical_findings" not in parsed or not isinstance(parsed["clinical_findings"], list):
        return False
    return True


def _extract_single(qid: int, prompt: str) -> dict:
    """Worker function: extract one question via LLM (thread-safe, no DB access).

    Returns a dict with keys: qid, parsed, in_tok, out_tok, raw_text, latency_ms, error.
    """
    call_t0 = time.time()
    try:
        parsed, in_tok, out_tok, raw_text = _call_with_retry(prompt)
        latency_ms = int((time.time() - call_t0) * 1000)

        if not _validate_extraction(parsed):
            raise ValueError("Invalid extraction: missing primary_diagnosis or clinical_findings")

        parsed["_question_id"] = qid
        parsed.setdefault("differential_diagnoses", [])
        parsed.setdefault("secondary_diagnoses", [])
        parsed.setdefault("clinical_findings", [])

        return {"qid": qid, "parsed": parsed, "in_tok": in_tok, "out_tok": out_tok,
                "raw_text": raw_text, "latency_ms": latency_ms, "error": None}
    except Exception as e:
        latency_ms = int((time.time() - call_t0) * 1000)
        return {"qid": qid, "parsed": None, "in_tok": 0, "out_tok": 0,
                "raw_text": None, "latency_ms": latency_ms, "error": str(e)}


def extract_all(conn, questions: list[dict], dry_run: bool = False,
                workers: int = DEFAULT_WORKERS) -> dict:
    """Step 5a: Extract diagnoses and findings from all board questions.

    Uses ThreadPoolExecutor for concurrent LLM calls. DB writes happen on the
    main thread only (SQLite is not thread-safe for writes).
    """
    total = len(questions)
    extracted = 0
    cached = 0
    errors = 0
    error_qids = []

    log.info(f"Step 5a: Extracting from {total} board questions ({workers} workers)")
    _log_processing(conn, STAGE_5A, "started", records_in=total)
    t0 = time.time()

    # Phase 1: Check cache and build work queue
    work_queue = []  # (qid, prompt, input_hash)
    for q in questions:
        qid = q["question_id"]
        question_text = _format_question_input(q)
        input_hash = _compute_input_hash_5a(qid, question_text)

        cached_result = _check_cache(conn, input_hash, STAGE_5A)
        if cached_result:
            cached += 1
            continue

        if not dry_run:
            prompt = EXTRACTION_PROMPT + question_text
            work_queue.append((qid, prompt, input_hash))

    log.info(f"Cache hits: {cached}, remaining to extract: {len(work_queue)}")

    if dry_run or not work_queue:
        duration = time.time() - t0
        _log_processing(conn, STAGE_5A, "completed", records_out=0,
                         records_error=0, duration_sec=duration)
        return {"step": "5a", "total": total, "extracted": 0, "cached": cached,
                "errors": 0, "error_qids": [], "duration_sec": round(duration, 1),
                "dry_run": dry_run}

    # Phase 2: Concurrent LLM calls
    pending = len(work_queue)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit all tasks — map future → (qid, input_hash)
        future_map = {}
        for qid, prompt, input_hash in work_queue:
            future = pool.submit(_extract_single, qid, prompt)
            future_map[future] = (qid, input_hash)

        # Collect results as they complete — write to DB on main thread
        for future in as_completed(future_map):
            qid, input_hash = future_map[future]
            result = future.result()
            completed_count += 1

            if result["error"] is None:
                _log_llm_call(conn, STAGE_5A, input_hash,
                              result["in_tok"], result["out_tok"],
                              result["latency_ms"],
                              json.dumps(result["parsed"]),
                              result["raw_text"])
                extracted += 1
            else:
                log.error(f"qid={qid} failed: {result['error']}")
                _log_llm_call(conn, STAGE_5A, input_hash, 0, 0,
                              result["latency_ms"], None, None, result["error"])
                errors += 1
                error_qids.append(qid)

            # Commit in batches of 10
            if completed_count % 10 == 0:
                conn.commit()

            if completed_count % 50 == 0 or completed_count == pending:
                elapsed = time.time() - t0
                rate = (extracted + cached) / elapsed * 3600 if elapsed > 0 else 0
                remaining = pending - completed_count
                eta_hrs = remaining / (completed_count / elapsed) / 3600 if completed_count > 0 else 0
                log.info(
                    f"Progress {completed_count}/{pending} | extracted={extracted} "
                    f"errors={errors} | {elapsed:.0f}s elapsed | ETA={eta_hrs:.1f}h"
                )

    conn.commit()
    duration = time.time() - t0
    status = "completed" if errors < total * 0.5 else "failed"
    _log_processing(conn, STAGE_5A, status, records_out=extracted,
                     records_error=errors, duration_sec=duration)

    summary = {
        "step": "5a",
        "total": total, "extracted": extracted, "cached": cached,
        "errors": errors, "error_qids": error_qids[:20],
        "duration_sec": round(duration, 1),
        "workers": workers,
    }
    log.info(f"Step 5a complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Step 5b: ICD-10 Mapping + Deduplication + Insertion
# ---------------------------------------------------------------------------

def _load_all_extractions(conn) -> list[dict]:
    """Load all successful Step 5a extractions from llm_call_log."""
    rows = conn.execute(
        "SELECT output_json FROM llm_call_log WHERE stage = ? AND error IS NULL",
        (STAGE_5A,),
    ).fetchall()
    extractions = []
    for row in rows:
        try:
            parsed = json.loads(row[0], strict=False)
            extractions.append(parsed)
        except (json.JSONDecodeError, TypeError):
            continue
    log.info(f"Loaded {len(extractions)} extractions from llm_call_log")
    return extractions


def _resolve_icd10(display_name: str, llm_suggestion: str | None,
                    icd10_dict: ICD10Dictionary) -> tuple[str | None, str | None, str]:
    """Resolve ICD-10 code for a diagnosis name.

    Returns (icd10_code, icd10_desc, resolution_method).
    """
    # 1. Validate LLM-suggested code
    if llm_suggestion:
        entry = icd10_dict.lookup_code(llm_suggestion)
        if entry and not entry.is_header:
            return entry.code, entry.description, "llm_validated"

    # 2. Exact description match
    exact = icd10_dict.exact_match(display_name)
    for e in exact:
        if not e.is_header:
            return e.code, e.description, "exact_desc"

    # 3. Fuzzy match (high confidence only)
    fuzzy = icd10_dict.fuzzy_match(display_name, top_k=1, min_score=0.7)
    if fuzzy:
        entry, score = fuzzy[0]
        return entry.code, entry.description, "fuzzy"

    # 4. Fall back to LLM suggestion (unvalidated)
    if llm_suggestion:
        return llm_suggestion, None, "llm_unvalidated"

    return None, None, "unresolved"


def _make_dx_dedup_key(name: str, icd10_code: str | None) -> str:
    """Create a dedup key for a diagnosis."""
    if icd10_code:
        return f"icd10:{icd10_code}"
    return f"name:{name.lower().strip()}"


def _make_finding_dedup_key(name: str, finding_type: str) -> str:
    """Create a dedup key for a clinical finding."""
    return f"{name.lower().strip()}|{finding_type}"


def map_and_insert(conn, icd10_dict: ICD10Dictionary,
                    extractions: list[dict] | None = None) -> dict:
    """Step 5b: Map to ICD-10 codes, deduplicate, and insert into DB."""
    if extractions is None:
        extractions = _load_all_extractions(conn)

    log.info(f"Step 5b: Processing {len(extractions)} extractions")
    _log_processing(conn, STAGE_5B, "started", records_in=len(extractions))
    t0 = time.time()

    # Clear existing data (5b reprocesses ALL extractions, so start fresh)
    old_dx = conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0]
    old_cf = conn.execute("SELECT COUNT(*) FROM clinical_findings").fetchone()[0]
    if old_dx or old_cf:
        conn.execute("DELETE FROM clinical_findings")
        conn.execute("DELETE FROM diagnoses")
        conn.commit()
        log.info(f"Cleared {old_dx} diagnoses + {old_cf} findings for fresh insert")

    # --- Collect all diagnoses ---
    raw_diagnoses: list[dict] = []
    for ext in extractions:
        qid = ext.get("_question_id")

        # Primary diagnosis
        pd = ext.get("primary_diagnosis", {})
        if pd and pd.get("name"):
            raw_diagnoses.append({
                "display_name": pd["name"].strip(),
                "icd10_suggestion": pd.get("icd10_suggestion"),
                "snomed_suggestion": pd.get("snomed_suggestion"),
                "category": pd.get("category"),
                "acuity": pd.get("acuity"),
                "role": "correct",
                "question_id": qid,
            })

        # Differential diagnoses
        for dx in ext.get("differential_diagnoses", []):
            if dx and dx.get("name"):
                raw_diagnoses.append({
                    "display_name": dx["name"].strip(),
                    "icd10_suggestion": dx.get("icd10_suggestion"),
                    "snomed_suggestion": dx.get("snomed_suggestion"),
                    "category": dx.get("category"),
                    "acuity": dx.get("acuity"),
                    "role": "distractor",
                    "question_id": qid,
                })

        # Secondary diagnoses
        for dx in ext.get("secondary_diagnoses", []):
            if dx and dx.get("name"):
                raw_diagnoses.append({
                    "display_name": dx["name"].strip(),
                    "icd10_suggestion": dx.get("icd10_suggestion"),
                    "snomed_suggestion": dx.get("snomed_suggestion"),
                    "category": dx.get("category"),
                    "acuity": dx.get("acuity"),
                    "role": "secondary",
                    "question_id": qid,
                })

    log.info(f"Collected {len(raw_diagnoses)} raw diagnosis mentions")

    # --- Resolve ICD-10 codes ---
    resolution_counts = {"llm_validated": 0, "exact_desc": 0, "fuzzy": 0,
                          "llm_unvalidated": 0, "unresolved": 0}

    for dx in raw_diagnoses:
        code, desc, method = _resolve_icd10(
            dx["display_name"], dx.get("icd10_suggestion"), icd10_dict
        )
        dx["icd10_code"] = code
        dx["icd10_desc"] = desc
        dx["resolution_method"] = method
        resolution_counts[method] += 1

    log.info(f"ICD-10 resolution: {resolution_counts}")

    # --- Deduplicate diagnoses ---
    unique_dx: dict[str, dict] = {}
    for dx in raw_diagnoses:
        key = _make_dx_dedup_key(dx["display_name"], dx.get("icd10_code"))
        if key not in unique_dx:
            unique_dx[key] = dx
        else:
            # Keep the one with better resolution
            existing = unique_dx[key]
            method_priority = {"llm_validated": 0, "exact_desc": 1, "fuzzy": 2,
                               "llm_unvalidated": 3, "unresolved": 4}
            if method_priority.get(dx.get("resolution_method"), 5) < \
               method_priority.get(existing.get("resolution_method"), 5):
                unique_dx[key] = dx

    log.info(f"Deduplicated to {len(unique_dx)} unique diagnoses")

    # --- Insert diagnoses ---
    dx_inserted = 0
    for key, dx in unique_dx.items():
        acuity = _normalize_acuity(dx.get("acuity"))
        category = _normalize_category(dx.get("category"))
        snomed_id = dx.get("snomed_suggestion")
        if snomed_id and not isinstance(snomed_id, str):
            snomed_id = str(snomed_id)

        conn.execute(
            """INSERT INTO diagnoses
               (icd10_code, icd10_desc, snomed_id, snomed_desc,
                display_name, category, acuity)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (dx.get("icd10_code"), dx.get("icd10_desc"),
             snomed_id, None,
             dx["display_name"], category, acuity),
        )
        dx_inserted += 1

    conn.commit()
    log.info(f"Inserted {dx_inserted} diagnoses")

    # --- Collect and deduplicate clinical findings ---
    raw_findings: list[dict] = []
    for ext in extractions:
        for f in ext.get("clinical_findings", []):
            if f and f.get("name"):
                raw_findings.append({
                    "display_name": f["name"].strip(),
                    "finding_type": _normalize_finding_type(f.get("finding_type")),
                    "snomed_suggestion": f.get("snomed_suggestion"),
                    "value": f.get("value"),
                    "present": f.get("present", True),
                    "relevance": f.get("relevance"),
                    "question_id": ext.get("_question_id"),
                })

    log.info(f"Collected {len(raw_findings)} raw finding mentions")

    unique_findings: dict[str, dict] = {}
    for f in raw_findings:
        key = _make_finding_dedup_key(f["display_name"], f["finding_type"])
        if key not in unique_findings:
            unique_findings[key] = f

    log.info(f"Deduplicated to {len(unique_findings)} unique findings")

    # --- Insert clinical findings ---
    cf_inserted = 0
    for key, f in unique_findings.items():
        snomed_id = f.get("snomed_suggestion")
        if snomed_id and not isinstance(snomed_id, str):
            snomed_id = str(snomed_id)

        conn.execute(
            """INSERT INTO clinical_findings
               (snomed_id, snomed_desc, display_name, finding_type)
               VALUES (?, ?, ?, ?)""",
            (snomed_id, None, f["display_name"], f["finding_type"]),
        )
        cf_inserted += 1

    conn.commit()

    duration = time.time() - t0
    _log_processing(conn, STAGE_5B, "completed",
                     records_out=dx_inserted + cf_inserted,
                     duration_sec=duration)

    summary = {
        "step": "5b",
        "raw_diagnoses": len(raw_diagnoses),
        "unique_diagnoses": len(unique_dx),
        "dx_inserted": dx_inserted,
        "resolution": resolution_counts,
        "raw_findings": len(raw_findings),
        "unique_findings": len(unique_findings),
        "cf_inserted": cf_inserted,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Step 5b complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Step 5d: SNOMED CT Validation & Mapping
# ---------------------------------------------------------------------------


_LAB_VALUE_RE = re.compile(
    r"\s+[<>≤≥]?\s*[\d.,]+\s*.*$"
)
_HISTORY_SUFFIX_RE = re.compile(r"\s+history$", re.IGNORECASE)
_AGE_RE = re.compile(r"^Age\s+\d+\s+(?:years?|months?|days?|weeks?).*$", re.IGNORECASE)


def _normalize_finding_name(name: str) -> list[str]:
    """Generate candidate search names for a clinical finding.

    Returns list of names to try (original first, then variants).
    """
    candidates = [name]
    # Strip lab values: "Sodium 140 mEq/L" → "Sodium"
    stripped = _LAB_VALUE_RE.sub("", name).strip()
    if stripped and stripped != name and len(stripped) >= 2:
        candidates.append(stripped)
    # Strip "history" suffix: "Gastroenteritis history" → "Gastroenteritis"
    stripped2 = _HISTORY_SUFFIX_RE.sub("", name).strip()
    if stripped2 and stripped2 != name:
        candidates.append(stripped2)
    # Age demographics: skip (no useful SNOMED match for "Age 42 years")
    # Sex demographics: map common patterns
    name_lower = name.lower().strip()
    if name_lower in ("female sex", "female gender"):
        candidates.append("Female")
    elif name_lower in ("male sex", "male gender"):
        candidates.append("Male")
    return candidates


def _pick_best_snomed(candidates, display_name):
    """Pick the best SNOMED concept from a list by fuzzy matching display_name."""
    from difflib import SequenceMatcher
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    name_lower = display_name.lower().strip()
    best = None
    best_score = -1
    for c in candidates:
        score = SequenceMatcher(None, name_lower, c.preferred_term.lower()).ratio()
        if score > best_score:
            best_score = score
            best = c
    return best


def run_snomed_mapping(conn, snomed_dict, pilot=None):
    """Step 5d: Wipe LLM-suggested SNOMED IDs and re-match from dictionary.

    Strategy for diagnoses:
      1. Clear all snomed_id/snomed_desc
      2. Try exact_match(display_name) against SNOMED descriptions
      3. Try icd10_to_snomed(icd10_code) reverse map, pick best fuzzy match
      4. Try fuzzy_match(display_name, min_score=0.85) for remaining

    Strategy for clinical_findings:
      1. Clear all snomed_id/snomed_desc
      2. Try exact_match(display_name)
      3. Try fuzzy_match(display_name, min_score=0.80) for remaining
    """
    import time
    t0 = time.time()

    # ---- Phase 1: Diagnoses ----
    log.info("Step 5d: SNOMED mapping for diagnoses...")

    # Clear existing SNOMED data
    conn.execute("UPDATE diagnoses SET snomed_id = NULL, snomed_desc = NULL")
    conn.commit()
    log.info("Cleared all diagnosis SNOMED IDs")

    rows = conn.execute(
        "SELECT diagnosis_id, display_name, icd10_code FROM diagnoses"
    ).fetchall()
    if pilot:
        rows = rows[:pilot]

    dx_exact = 0
    dx_icd10_map = 0
    dx_fuzzy = 0
    dx_unmatched = 0
    dx_updates = []

    for did, name, icd10 in rows:
        snomed_id = None
        snomed_desc = None
        method = None

        # Step 1: Exact match against SNOMED descriptions
        matches = snomed_dict.exact_match(name)
        if matches:
            best = _pick_best_snomed(matches, name)
            snomed_id = best.concept_id
            snomed_desc = best.preferred_term
            method = "exact"
            dx_exact += 1
        else:
            # Step 2: ICD-10 → SNOMED reverse map
            if icd10:
                candidates = snomed_dict.icd10_to_snomed(icd10)
                if candidates:
                    best = _pick_best_snomed(candidates, name)
                    snomed_id = best.concept_id
                    snomed_desc = best.preferred_term
                    method = "icd10_map"
                    dx_icd10_map += 1

            # Step 3: Fuzzy match (very high threshold to avoid false positives)
            if not snomed_id:
                fuzzy = snomed_dict.fuzzy_match(name, top_k=1, min_score=0.95)
                if fuzzy:
                    concept, score = fuzzy[0]
                    snomed_id = concept.concept_id
                    snomed_desc = concept.preferred_term
                    method = "fuzzy"
                    dx_fuzzy += 1

        if snomed_id:
            dx_updates.append((snomed_id, snomed_desc, did))
        else:
            dx_unmatched += 1

    # Batch update
    conn.executemany(
        "UPDATE diagnoses SET snomed_id = ?, snomed_desc = ? WHERE diagnosis_id = ?",
        dx_updates,
    )
    conn.commit()

    dx_total = len(rows)
    dx_matched = dx_exact + dx_icd10_map + dx_fuzzy
    log.info(f"Diagnoses: {dx_matched}/{dx_total} mapped "
             f"(exact={dx_exact}, icd10_map={dx_icd10_map}, fuzzy={dx_fuzzy}, "
             f"unmatched={dx_unmatched})")

    # ---- Phase 2: Clinical Findings ----
    log.info("Step 5d: SNOMED mapping for clinical findings...")

    conn.execute("UPDATE clinical_findings SET snomed_id = NULL, snomed_desc = NULL")
    conn.commit()
    log.info("Cleared all clinical_findings SNOMED IDs")

    rows = conn.execute(
        "SELECT finding_id, display_name FROM clinical_findings"
    ).fetchall()
    if pilot:
        rows = rows[:pilot]

    cf_exact = 0
    cf_fuzzy = 0
    cf_unmatched = 0
    cf_updates = []

    for fid, name in rows:
        snomed_id = None
        snomed_desc = None

        # Try each normalized variant of the name
        name_variants = _normalize_finding_name(name)
        for variant in name_variants:
            if snomed_id:
                break
            # Step 1: Exact match
            matches = snomed_dict.exact_match(variant)
            if matches:
                best = _pick_best_snomed(matches, name)
                snomed_id = best.concept_id
                snomed_desc = best.preferred_term
                cf_exact += 1
                break

        if not snomed_id:
            # Step 2: Fuzzy match (very high threshold — near-exact only)
            for variant in name_variants:
                fuzzy = snomed_dict.fuzzy_match(variant, top_k=1, min_score=0.90)
                if fuzzy:
                    concept, score = fuzzy[0]
                    snomed_id = concept.concept_id
                    snomed_desc = concept.preferred_term
                    cf_fuzzy += 1
                    break

        if snomed_id:
            cf_updates.append((snomed_id, snomed_desc, fid))
        else:
            cf_unmatched += 1

    conn.executemany(
        "UPDATE clinical_findings SET snomed_id = ?, snomed_desc = ? WHERE finding_id = ?",
        cf_updates,
    )
    conn.commit()

    cf_total = len(rows)
    cf_matched = cf_exact + cf_fuzzy
    log.info(f"Clinical findings: {cf_matched}/{cf_total} mapped "
             f"(exact={cf_exact}, fuzzy={cf_fuzzy}, unmatched={cf_unmatched})")

    duration = time.time() - t0
    summary = {
        "step": "5d_snomed",
        "diagnoses": {
            "total": dx_total,
            "exact": dx_exact,
            "icd10_map": dx_icd10_map,
            "fuzzy": dx_fuzzy,
            "unmatched": dx_unmatched,
            "coverage_pct": round(100 * dx_matched / dx_total, 1) if dx_total else 0,
        },
        "clinical_findings": {
            "total": cf_total,
            "exact": cf_exact,
            "fuzzy": cf_fuzzy,
            "unmatched": cf_unmatched,
            "coverage_pct": round(100 * cf_matched / cf_total, 1) if cf_total else 0,
        },
        "duration_sec": round(duration, 1),
    }
    log.info(f"Step 5d complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# Step 5e: SapBERT Semantic SNOMED Matching
# ---------------------------------------------------------------------------

def run_snomed_embedding(conn, pilot=None):
    """Step 5e: Match unmatched diagnoses/findings using SapBERT embeddings.

    Only processes items where snomed_id IS NULL (i.e., items that
    dictionary matching in step 5d couldn't resolve).
    """
    from etl.ontology.sapbert_embedder import SapBERTEmbedder, DEFAULT_INDEX_PATH
    from etl.ontology.snomed import SNOMEDDictionary

    import time
    t0 = time.time()

    # Load embedder + index
    embedder = SapBERTEmbedder()
    embedder.load_index(DEFAULT_INDEX_PATH)

    # Load SNOMED dictionary for preferred term lookup
    snomed_dict = SNOMEDDictionary()

    # ---- Phase 1: Diagnoses ----
    rows = conn.execute(
        "SELECT diagnosis_id, display_name FROM diagnoses WHERE snomed_id IS NULL"
    ).fetchall()
    if pilot:
        rows = rows[:pilot]

    dx_total = len(rows)
    log.info(f"Step 5e: {dx_total} unmatched diagnoses to embed")

    if dx_total > 0:
        dx_ids = [r[0] for r in rows]
        dx_names = [r[1] for r in rows]

        results = embedder.find_closest(dx_names, top_k=1, min_score=0.70)

        dx_matched = 0
        dx_updates = []
        for i, matches in enumerate(results):
            if matches:
                cid, score = matches[0]
                pterm = snomed_dict.get_preferred_term(cid)
                dx_updates.append((cid, pterm, dx_ids[i]))
                dx_matched += 1

        conn.executemany(
            "UPDATE diagnoses SET snomed_id = ?, snomed_desc = ? WHERE diagnosis_id = ?",
            dx_updates,
        )
        conn.commit()
        log.info(f"Diagnoses: {dx_matched}/{dx_total} newly matched via SapBERT")
    else:
        dx_matched = 0

    # ---- Phase 2: Clinical Findings ----
    rows = conn.execute(
        "SELECT finding_id, display_name FROM clinical_findings WHERE snomed_id IS NULL"
    ).fetchall()
    if pilot:
        rows = rows[:pilot]

    cf_total = len(rows)
    log.info(f"Step 5e: {cf_total} unmatched clinical findings to embed")

    if cf_total > 0:
        cf_ids = [r[0] for r in rows]
        cf_names = [r[1] for r in rows]

        # Use normalized names for better matching
        search_names = []
        for name in cf_names:
            variants = _normalize_finding_name(name)
            # Use shortest meaningful variant (strip lab values etc.)
            best = variants[-1] if len(variants) > 1 else variants[0]
            search_names.append(best)

        results = embedder.find_closest(search_names, top_k=1, min_score=0.65)

        cf_matched = 0
        cf_updates = []
        for i, matches in enumerate(results):
            if matches:
                cid, score = matches[0]
                pterm = snomed_dict.get_preferred_term(cid)
                cf_updates.append((cid, pterm, cf_ids[i]))
                cf_matched += 1

        conn.executemany(
            "UPDATE clinical_findings SET snomed_id = ?, snomed_desc = ? WHERE finding_id = ?",
            cf_updates,
        )
        conn.commit()
        log.info(f"Clinical findings: {cf_matched}/{cf_total} newly matched via SapBERT")
    else:
        cf_matched = 0

    # ---- Summary ----
    # Total coverage after 5d + 5e
    total_dx = conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0]
    dx_with_snomed = conn.execute(
        "SELECT COUNT(*) FROM diagnoses WHERE snomed_id IS NOT NULL"
    ).fetchone()[0]
    total_cf = conn.execute("SELECT COUNT(*) FROM clinical_findings").fetchone()[0]
    cf_with_snomed = conn.execute(
        "SELECT COUNT(*) FROM clinical_findings WHERE snomed_id IS NOT NULL"
    ).fetchone()[0]

    duration = time.time() - t0
    summary = {
        "step": "5e_sapbert",
        "diagnoses": {
            "unmatched_input": dx_total,
            "newly_matched": dx_matched,
            "total_coverage": f"{dx_with_snomed}/{total_dx} ({100*dx_with_snomed/total_dx:.1f}%)",
        },
        "clinical_findings": {
            "unmatched_input": cf_total,
            "newly_matched": cf_matched,
            "total_coverage": f"{cf_with_snomed}/{total_cf} ({100*cf_with_snomed/total_cf:.1f}%)",
        },
        "duration_sec": round(duration, 1),
    }
    log.info(f"Step 5e complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# Step 5f: LOINC mapping for lab_value findings (deterministic, no LLM)
# ---------------------------------------------------------------------------

# Regex to strip qualitative prefixes from lab finding names
_QUAL_PREFIX_RE = re.compile(
    r"^(elevated|increased|decreased|low|high|normal|abnormal|positive|negative|"
    r"reduced|raised|rising|falling|prolonged|shortened|absent|present|"
    r"very high|very low|markedly elevated|mildly elevated|severely elevated|"
    r"borderline|subnormal|supranormal)\s+",
    re.IGNORECASE,
)
# Regex to strip trailing qualifiers like "level", "levels", "measurement", "test"
_TRAIL_QUALIFIER_RE = re.compile(
    r"\s+(level|levels|measurement|test|result|results|values?|concentration|ratio|count|"
    r"within reference range|above reference range|below reference range|"
    r"finding|assessment)s?$",
    re.IGNORECASE,
)
# Regex to strip specimen prefixes: "Serum creatinine" → "creatinine"
_SPECIMEN_PREFIX_RE = re.compile(
    r"^(serum|blood|urine|urinary|plasma|arterial|venous|whole blood|fasting|"
    r"fasting blood|total|free|ionized|conjugated|unconjugated)\s+",
    re.IGNORECASE,
)


def _normalize_lab_name(name: str) -> list[str]:
    """Generate candidate search names for a lab finding.

    Returns list of names to try (original first, then stripped variants).
    """
    candidates = [name]

    # Strip numeric values: "Hemoglobin 7.2 g/dL" → "Hemoglobin"
    stripped = _LAB_VALUE_RE.sub("", name).strip()
    if stripped and stripped != name and len(stripped) >= 2:
        candidates.append(stripped)

    # Strip qualitative prefixes: "Elevated aminotransferases" → "aminotransferases"
    for candidate in list(candidates):
        no_prefix = _QUAL_PREFIX_RE.sub("", candidate).strip()
        if no_prefix and no_prefix != candidate and no_prefix not in candidates:
            candidates.append(no_prefix)

    # Strip trailing qualifiers: "Follicle-stimulating hormone level" → "Follicle-stimulating hormone"
    for candidate in list(candidates):
        no_trail = _TRAIL_QUALIFIER_RE.sub("", candidate).strip()
        if no_trail and no_trail != candidate and no_trail not in candidates:
            candidates.append(no_trail)

    # Strip specimen prefixes: "Serum creatinine" → "creatinine"
    for candidate in list(candidates):
        no_spec = _SPECIMEN_PREFIX_RE.sub("", candidate).strip()
        if no_spec and no_spec != candidate and no_spec not in candidates:
            candidates.append(no_spec)

    # Filter out overly short candidates (avoid "CA" from "CA 15-3 tumor marker")
    return [c for c in candidates if len(c) >= 3]


def run_loinc_mapping(conn, pilot=None):
    """Step 5f: Map lab_value clinical findings to LOINC codes.

    Deterministic — no LLM calls. Uses exact + fuzzy dictionary matching.
    """
    from etl.db import migrate_add_loinc_columns
    from etl.ontology.loinc import LOINCDictionary

    t0 = time.time()

    # Ensure columns exist on existing databases
    migrate_add_loinc_columns(conn)

    # Load LOINC dictionary
    loinc_dict = LOINCDictionary()
    if not loinc_dict.codes:
        log.error("LOINC dictionary is empty — cannot map")
        return {"error": "LOINC dictionary empty"}

    # Clear existing LOINC mappings
    conn.execute(
        "UPDATE clinical_findings SET loinc_code = NULL, loinc_desc = NULL "
        "WHERE finding_type = 'lab_value'"
    )
    conn.commit()

    # Fetch lab_value findings
    query = "SELECT finding_id, display_name, snomed_desc FROM clinical_findings WHERE finding_type = 'lab_value'"
    if pilot:
        query += f" LIMIT {int(pilot)}"
    findings = conn.execute(query).fetchall()
    log.info(f"Step 5f: Mapping {len(findings)} lab_value findings to LOINC")

    # Counters
    exact_count = 0
    component_count = 0
    fuzzy_count = 0
    unmatched_count = 0
    updates = []

    for fid, display_name, snomed_desc in findings:
        matched = None
        method = None

        # Generate candidate names
        candidates = _normalize_lab_name(display_name)
        # Also try snomed_desc if available
        if snomed_desc and snomed_desc not in candidates:
            candidates.append(snomed_desc)
            # Also strip snomed_desc qualifiers
            for variant in _normalize_lab_name(snomed_desc):
                if variant not in candidates:
                    candidates.append(variant)
            # Strip trailing ", serum", ", arterial", etc. from SNOMED descriptions
            stripped_snomed = re.sub(r",\s*(serum|arterial|venous|plasma|urine|blood|fasting)$",
                                     "", snomed_desc, flags=re.IGNORECASE).strip()
            if stripped_snomed and stripped_snomed != snomed_desc and stripped_snomed not in candidates:
                candidates.append(stripped_snomed)
                for variant in _normalize_lab_name(stripped_snomed):
                    if variant not in candidates:
                        candidates.append(variant)

        # Try exact match on LONG_COMMON_NAME / COMPONENT
        for cand in candidates:
            matches = loinc_dict.exact_match(cand)
            if matches:
                matched = loinc_dict.disambiguate(matches)
                method = "exact"
                break

        # Try component-only match
        if not matched:
            for cand in candidates:
                matches = loinc_dict.component_match(cand)
                if matches:
                    matched = loinc_dict.disambiguate(matches)
                    method = "component"
                    break

        # Try fuzzy match (high threshold — 0.90 to avoid false positives)
        if not matched:
            for cand in candidates:
                results = loinc_dict.fuzzy_match(cand, top_k=5, min_score=0.90)
                if results:
                    # Disambiguate among top matches
                    top_entries = [entry for entry, _score in results]
                    matched = loinc_dict.disambiguate(top_entries)
                    method = "fuzzy"
                    break

        if matched:
            updates.append((matched.loinc_num, matched.long_common_name, fid))
            if method == "exact":
                exact_count += 1
            elif method == "component":
                component_count += 1
            else:
                fuzzy_count += 1
        else:
            unmatched_count += 1

    # Batch update
    if updates:
        conn.executemany(
            "UPDATE clinical_findings SET loinc_code = ?, loinc_desc = ? WHERE finding_id = ?",
            updates,
        )
        conn.commit()

    total = len(findings)
    mapped = exact_count + component_count + fuzzy_count
    duration = time.time() - t0

    summary = {
        "step": "5f_loinc",
        "total_lab_findings": total,
        "mapped": mapped,
        "coverage_pct": round(mapped / total * 100, 1) if total > 0 else 0,
        "exact": exact_count,
        "component": component_count,
        "fuzzy": fuzzy_count,
        "unmatched": unmatched_count,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Step 5f complete: {json.dumps(summary, indent=2)}")

    # Show some sample mappings
    sample = conn.execute(
        "SELECT display_name, loinc_code, loinc_desc FROM clinical_findings "
        "WHERE loinc_code IS NOT NULL ORDER BY RANDOM() LIMIT 10"
    ).fetchall()
    if sample:
        log.info("Sample LOINC mappings:")
        for name, code, desc in sample:
            log.info(f"  {name} → {code}: {desc}")

    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(conn):
    """Run verification queries for Stage 5 output."""
    log.info("--- Stage 5 Verification ---")

    # Diagnosis counts
    total_dx = conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0]
    dx_with_icd = conn.execute(
        "SELECT COUNT(*) FROM diagnoses WHERE icd10_code IS NOT NULL"
    ).fetchone()[0]
    pct_icd = (dx_with_icd / total_dx * 100) if total_dx > 0 else 0
    log.info(f"Diagnoses: {total_dx} total, {dx_with_icd} with ICD-10 ({pct_icd:.1f}%) [target: >=90%]")

    # Finding counts
    total_cf = conn.execute("SELECT COUNT(*) FROM clinical_findings").fetchone()[0]
    log.info(f"Clinical findings: {total_cf} total")

    # LOINC coverage (lab_value findings)
    total_lab = conn.execute(
        "SELECT COUNT(*) FROM clinical_findings WHERE finding_type = 'lab_value'"
    ).fetchone()[0]
    lab_with_loinc = conn.execute(
        "SELECT COUNT(*) FROM clinical_findings WHERE finding_type = 'lab_value' AND loinc_code IS NOT NULL"
    ).fetchone()[0]
    pct_loinc = (lab_with_loinc / total_lab * 100) if total_lab > 0 else 0
    log.info(f"Lab findings with LOINC: {lab_with_loinc}/{total_lab} ({pct_loinc:.1f}%)")

    # Finding type distribution
    rows = conn.execute(
        "SELECT finding_type, COUNT(*) FROM clinical_findings GROUP BY finding_type ORDER BY COUNT(*) DESC"
    ).fetchall()
    log.info("Finding type distribution:")
    for r in rows:
        log.info(f"  {r[0]}: {r[1]}")

    # Diagnosis category distribution
    rows = conn.execute(
        "SELECT category, COUNT(*) FROM diagnoses WHERE category IS NOT NULL "
        "GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    log.info("Diagnosis category distribution:")
    for r in rows:
        log.info(f"  {r[0]}: {r[1]}")

    # Acuity distribution
    rows = conn.execute(
        "SELECT acuity, COUNT(*) FROM diagnoses GROUP BY acuity ORDER BY COUNT(*) DESC"
    ).fetchall()
    log.info("Acuity distribution:")
    for r in rows:
        log.info(f"  {r[0]}: {r[1]}")

    # LLM call stats
    for stage in (STAGE_5A, STAGE_5B):
        row = conn.execute(
            """SELECT COUNT(*) as calls,
                      COALESCE(SUM(input_tokens), 0) as total_in,
                      COALESCE(SUM(output_tokens), 0) as total_out,
                      COALESCE(AVG(latency_ms), 0) as avg_latency,
                      SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
               FROM llm_call_log WHERE stage = ?""",
            (stage,),
        ).fetchone()
        log.info(
            f"LLM [{stage}]: {row[0]} calls, {row[1]} in_tok, {row[2]} out_tok, "
            f"{row[3]:.0f}ms avg, {row[4]} errors"
        )

    # Extraction coverage
    total_q = conn.execute("SELECT COUNT(*) FROM board_questions").fetchone()[0]
    extracted_q = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NULL",
        (STAGE_5A,),
    ).fetchone()[0]
    log.info(f"Extraction coverage: {extracted_q}/{total_q} questions "
             f"({extracted_q/total_q*100:.1f}%)" if total_q > 0 else "No questions")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(conn, output_dir: Path):
    """Export diagnoses and clinical_findings to CSV for review."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Diagnoses
    dx_path = output_dir / "s05_diagnoses.csv"
    rows = conn.execute(
        "SELECT diagnosis_id, icd10_code, icd10_desc, snomed_id, display_name, "
        "category, acuity FROM diagnoses ORDER BY diagnosis_id"
    ).fetchall()
    with open(dx_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["diagnosis_id", "icd10_code", "icd10_desc", "snomed_id",
                         "display_name", "category", "acuity"])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} diagnoses to {dx_path}")

    # Clinical findings
    cf_path = output_dir / "s05_clinical_findings.csv"
    rows = conn.execute(
        "SELECT finding_id, snomed_id, display_name, finding_type, normal_range "
        "FROM clinical_findings ORDER BY finding_id"
    ).fetchall()
    with open(cf_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["finding_id", "snomed_id", "display_name", "finding_type", "normal_range"])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} clinical findings to {cf_path}")


# ---------------------------------------------------------------------------
# ICD-10 download helper
# ---------------------------------------------------------------------------

def download_icd10():
    """Download and parse CMS ICD-10-CM 2025 code descriptions."""
    import zipfile
    import tempfile

    url = "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
    output_path = Path(__file__).resolve().parent.parent.parent / "data" / "ontology" / "icd10cm_2025.csv"

    if output_path.exists():
        log.info(f"ICD-10-CM file already exists: {output_path}")
        return

    log.info(f"Downloading ICD-10-CM 2025 from {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60) as client:
        resp = client.get(url, follow_redirects=True)
        resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    with zipfile.ZipFile(tmp_path) as zf:
        order_file = [n for n in zf.namelist() if "order" in n.lower() and n.endswith(".txt")]
        if not order_file:
            raise FileNotFoundError("No order file found in ZIP")

        with zf.open(order_file[0]) as fin:
            lines = fin.read().decode("utf-8").splitlines()

    total = 0
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["code", "description", "is_header"])
        for line in lines:
            if len(line.strip()) < 17:
                continue
            code_raw = line[6:13].strip()
            is_header = line[14].strip()
            long_desc = line[77:].strip() if len(line) > 77 else line[16:77].strip()
            code = code_raw[:3] + "." + code_raw[3:] if len(code_raw) > 3 else code_raw
            writer.writerow([code, long_desc, is_header])
            total += 1

    Path(tmp_path).unlink(missing_ok=True)
    log.info(f"Parsed {total} ICD-10-CM codes to {output_path}")


# ---------------------------------------------------------------------------
# run() for main.py integration
# ---------------------------------------------------------------------------

def run(conn, workers: int = DEFAULT_WORKERS):
    """Run the full Stage 5 pipeline (5a + 5b)."""
    questions = _fetch_questions(conn)
    summary_5a = extract_all(conn, questions, workers=workers)

    icd10_dict = ICD10Dictionary()
    summary_5b = map_and_insert(conn, icd10_dict)

    verify(conn)
    return {"5a": summary_5a, "5b": summary_5b}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Ontology Resolution — diagnoses + clinical_findings"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--extract", action="store_true",
                       help="Step 5a: Extract diagnoses/findings via LLM")
    group.add_argument("--map", action="store_true",
                       help="Step 5b: Map to ICD-10 and insert into DB")
    group.add_argument("--all", action="store_true",
                       help="Run both 5a and 5b sequentially")
    group.add_argument("--verify-only", action="store_true",
                       help="Run verification queries only")
    group.add_argument("--export-csv", action="store_true",
                       help="Export diagnoses/findings to CSV")
    group.add_argument("--snomed", action="store_true",
                       help="Step 5d: SNOMED CT validation & mapping")
    group.add_argument("--snomed-build-index", action="store_true",
                       help="Step 5e: Build SapBERT embedding index for SNOMED CT (~2-3 hrs)")
    group.add_argument("--snomed-embed", action="store_true",
                       help="Step 5e: Semantic SNOMED mapping via SapBERT embeddings")
    group.add_argument("--loinc", action="store_true",
                       help="Step 5f: LOINC mapping for lab_value findings (deterministic)")
    group.add_argument("--download-icd10", action="store_true",
                       help="Download CMS ICD-10-CM 2025 data")

    parser.add_argument("--pilot", type=int, default=0,
                        help="Process only first N questions (for testing)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of concurrent LLM workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count operations without calling LLM")
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()

    if args.download_icd10:
        download_icd10()
        return

    conn = get_connection(args.db)

    if args.verify_only:
        verify(conn)
        conn.close()
        return

    if args.export_csv:
        export_csv(conn, DATA_DIR)
        conn.close()
        return

    limit = args.pilot if args.pilot > 0 else None

    if args.extract or args.all:
        questions = _fetch_questions(conn, limit)
        if not questions:
            log.info("No board questions found")
        else:
            summary_5a = extract_all(conn, questions, dry_run=args.dry_run,
                                       workers=args.workers)
            print(json.dumps(summary_5a, indent=2))

    if args.map or args.all:
        if args.dry_run:
            log.info("Dry run — skipping Step 5b mapping")
        else:
            icd10_dict = ICD10Dictionary()
            summary_5b = map_and_insert(conn, icd10_dict)
            print(json.dumps(summary_5b, indent=2))

    if args.snomed:
        from etl.ontology.snomed import SNOMEDDictionary
        snomed_dict = SNOMEDDictionary()
        limit = args.pilot if args.pilot > 0 else None
        summary_5d = run_snomed_mapping(conn, snomed_dict, pilot=limit)
        print(json.dumps(summary_5d, indent=2))
        verify(conn)

    if args.snomed_build_index:
        from etl.ontology.snomed import SNOMEDDictionary
        from etl.ontology.sapbert_embedder import SapBERTEmbedder, DEFAULT_INDEX_PATH
        snomed_dict = SNOMEDDictionary()
        embedder = SapBERTEmbedder()
        embedder.build_snomed_index(snomed_dict, DEFAULT_INDEX_PATH)
        log.info("SNOMED SapBERT index built successfully")

    if args.snomed_embed:
        limit = args.pilot if args.pilot > 0 else None
        summary_5e = run_snomed_embedding(conn, pilot=limit)
        print(json.dumps(summary_5e, indent=2))

    if args.loinc:
        limit = args.pilot if args.pilot > 0 else None
        summary_5f = run_loinc_mapping(conn, pilot=limit)
        print(json.dumps(summary_5f, indent=2))
        verify(conn)

    if (args.extract or args.map or args.all) and not args.dry_run:
        verify(conn)

    conn.close()


if __name__ == "__main__":
    main()
