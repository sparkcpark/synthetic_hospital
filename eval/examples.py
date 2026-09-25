"""Few-shot example selection and formatting.

Selects examples by model agreement: queries evaluation_predictions for all
zero_shot val runs, counts how many models scored each gt_id correctly,
and picks the top N items by agreement count.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from eval.config import get_pg_connection
from eval.report import PRIMARY_METRICS

log = logging.getLogger(__name__)

# Truncation limits for input text in few-shot examples
EHR_TRUNCATION_CHARS = 1500
RETRIEVAL_TRUNCATION_CHARS = 3000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=5)
def get_few_shot_examples(task: str) -> str:
    """Cached wrapper: select examples and format as few-shot block.

    Returns formatted string ready for {few_shot_examples} placeholder.
    Returns "" if no suitable examples found (triggers zero_shot fallback).
    """
    conn = get_pg_connection()
    try:
        examples = select_examples(task, n=3, conn=conn)
        if not examples:
            # The released benchmark ships no stored model runs, so fall back to the frozen
            # example blocks exported from the original database (eval/few_shot_examples.json).
            frozen = _frozen_examples().get(task, "")
            if frozen:
                log.info("Using frozen few-shot examples for task=%s (%d chars)", task, len(frozen))
                return frozen
            log.warning("No few-shot examples found for task=%s", task)
            return ""
        block = format_few_shot_block(examples, task)
        log.info("Selected %d few-shot examples for task=%s (%d chars)",
                 len(examples), task, len(block))
        return block
    finally:
        conn.close()


FROZEN_EXAMPLES_PATH = Path(__file__).with_name("few_shot_examples.json")


@lru_cache(maxsize=1)
def _frozen_examples() -> dict:
    """Few-shot blocks selected on the original database (model agreement over zero-shot runs), frozen for the release."""
    try:
        return json.loads(FROZEN_EXAMPLES_PATH.read_text())
    except (OSError, ValueError):
        return {}


def select_examples(task: str, n: int = 3, conn=None) -> list[dict]:
    """Select top-N few-shot examples by model agreement from zero_shot val runs.

    Returns list of dicts: {gt_id, task, input_text, expected_output}.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_pg_connection()
    try:
        return _select_examples_impl(task, n, conn)
    finally:
        if own_conn:
            conn.close()


def format_few_shot_block(examples: list[dict], task: str) -> str:
    """Format selected examples as a numbered few-shot block."""
    if not examples:
        return ""

    max_chars = RETRIEVAL_TRUNCATION_CHARS if task == "evidence_retrieval" else EHR_TRUNCATION_CHARS
    parts = []

    for i, ex in enumerate(examples, 1):
        input_text = ex["input_text"]
        if len(input_text) > max_chars:
            input_text = input_text[:max_chars] + "\n[...truncated...]"

        expected = json.dumps(ex["expected_output"], indent=2)
        parts.append(
            f"Example {i}:\n"
            f"PATIENT CLINICAL DATA:\n{input_text}\n\n"
            f"Expected output:\n{expected}"
        )

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _select_examples_impl(task: str, n: int, conn) -> list[dict]:
    """Core selection logic."""
    with conn.cursor() as cur:
        # 1. Find all completed zero_shot val runs for this task
        cur.execute("""
            SELECT DISTINCT run_id FROM evaluation_runs
            WHERE task = %s AND split = 'public' AND prompt_strategy = 'zero_shot'
              AND completed_at IS NOT NULL
        """, (task,))
        run_ids = [r[0] for r in cur.fetchall()]

    if len(run_ids) < 2:
        log.info("Only %d zero_shot val runs for task=%s — need ≥2 for agreement",
                 len(run_ids), task)
        return []

    # 2. Load all predictions grouped by gt_id
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ep.run_id, ep.gt_id, ep.prediction, bgt.ground_truth
            FROM evaluation_predictions ep
            JOIN benchmark_ground_truth bgt ON ep.gt_id = bgt.gt_id
            WHERE ep.run_id = ANY(%s)
            ORDER BY ep.gt_id
        """, (run_ids,))
        rows = cur.fetchall()

    if not rows:
        return []

    # 3. For retrieval, pre-load judgments since they're not in GT JSON
    judgments_cache: dict[int, dict] = {}
    if task == "evidence_retrieval":
        gt_ids_in_rows = set(r[1] for r in rows)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT gt_id, passage_id, relevance_grade
                FROM relevance_judgments
                WHERE gt_id = ANY(%s)
            """, (list(gt_ids_in_rows),))
            for gt_id_j, pid, grade in cur.fetchall():
                judgments_cache.setdefault(gt_id_j, {})[pid] = grade

    # 4. Group by gt_id and count correct models
    gt_groups: dict[int, dict] = {}  # gt_id → {gt, correct_count, total_score}
    for run_id, gt_id, pred_json, gt_json in rows:
        pred = pred_json if isinstance(pred_json, dict) else json.loads(pred_json)
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)

        if gt_id not in gt_groups:
            # Inject judgments for retrieval
            if task == "evidence_retrieval" and gt_id in judgments_cache:
                gt["_judgments"] = judgments_cache[gt_id]
            gt_groups[gt_id] = {"gt": gt, "correct_count": 0, "total": 0}

        gt_groups[gt_id]["total"] += 1
        try:
            if _is_correct_fast(task, pred, gt_groups[gt_id]["gt"]):
                gt_groups[gt_id]["correct_count"] += 1
        except Exception:
            pass  # Skip malformed predictions

    # 4. Sort by agreement count desc, then gt_id asc for determinism
    ranked = sorted(
        gt_groups.items(),
        key=lambda kv: (-kv[1]["correct_count"], kv[0]),
    )

    # Filter: at least 1 model must have gotten it correct
    ranked = [(gt_id, info) for gt_id, info in ranked if info["correct_count"] > 0]

    if not ranked:
        return []

    # 5. Take top n and build output dicts
    results = []
    for gt_id, info in ranked[:n]:
        input_text = _assemble_input_text(task, gt_id, conn)
        if not input_text:
            continue

        expected = _format_expected_output(task, info["gt"], conn, gt_id)
        results.append({
            "gt_id": gt_id,
            "task": task,
            "input_text": input_text,
            "expected_output": expected,
        })

    return results


def _is_correct_fast(task: str, pred: dict, gt: dict) -> bool:
    """Fast binary correctness check for example selection.

    Uses cheap heuristics per task — avoids expensive per-call RougeScorer for
    summarization and complex scoring for other tasks.
    """
    if task == "diagnosis_accuracy":
        from eval.scoring import _normalize_icd10
        gt_primary = gt.get("primary_diagnosis", {})
        gt_icd = _normalize_icd10(gt_primary.get("icd10", ""))
        if not gt_icd:
            return False
        acceptable = {gt_icd}
        for alt in gt.get("acceptable_alternatives", []):
            alt_icd = _normalize_icd10(alt.get("icd10", ""))
            if alt_icd:
                acceptable.add(alt_icd)
        for dx in pred.get("diagnoses", [])[:1]:
            pred_icd = _normalize_icd10(dx.get("icd10", ""))
            if pred_icd in acceptable:
                return True
        return False

    elif task == "patient_diagnosis":
        # F1 >= 0.5 at 3-char ICD-10 level
        gt_codes = set()
        for dx in gt.get("active_diagnoses", []) + gt.get("chronic_conditions", []):
            c = dx.get("icd10", "").replace(".", "").upper()[:3]
            if c:
                gt_codes.add(c)
        pred_codes = set()
        for dx in pred.get("active_diagnoses", []) + pred.get("chronic_conditions", []):
            c = dx.get("icd10", "").replace(".", "").upper()[:3]
            if c:
                pred_codes.add(c)
        if not gt_codes:
            return True
        tp = len(gt_codes & pred_codes)
        prec = tp / len(pred_codes) if pred_codes else 0
        rec = tp / len(gt_codes) if gt_codes else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return f1 >= 0.5

    elif task == "context_summarization":
        # Token overlap F1 >= 0.35 (avoids expensive rouge_score instantiation)
        pred_text = pred.get("summary", "")
        ref_text = gt.get("reference_summary", "")
        if not isinstance(pred_text, str) or not isinstance(ref_text, str):
            return False
        if not pred_text or not ref_text:
            return False
        pred_tokens = set(pred_text.lower().split())
        ref_tokens = set(ref_text.lower().split())
        tp = len(pred_tokens & ref_tokens)
        prec = tp / len(pred_tokens) if pred_tokens else 0
        rec = tp / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return f1 >= 0.35

    elif task == "evidence_retrieval":
        rankings = pred.get("rankings", [])
        if rankings:
            top_pid = rankings[0].get("passage_id", "")
            judg = gt.get("_judgments", {})
            return judg.get(top_pid, 0) >= 2
        return False

    elif task == "imaging_indication":
        pred_q = set(pred.get("clinical_question", "").lower().split())
        ref_q = set(gt.get("inferred_clinical_question", "").lower().split())
        if not pred_q or not ref_q:
            return False
        tp = len(pred_q & ref_q)
        prec = tp / len(pred_q) if pred_q else 0
        rec = tp / len(ref_q) if ref_q else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return f1 >= 0.3

    return False


def _assemble_input_text(task: str, gt_id: int, conn) -> str:
    """Assemble the EHR/input text for a gt_id, mirroring what load_inputs() produces."""
    cur = conn.cursor()

    if task == "diagnosis_accuracy":
        cur.execute(
            "SELECT question_id, patient_id, granularity FROM benchmark_ground_truth WHERE gt_id = %s",
            (gt_id,),
        )
        row = cur.fetchone()
        if not row:
            return ""
        qid, pid, gran = row
        if gran == "question" and qid:
            from eval.tasks.diagnosis import _assemble_question_ehr
            return _assemble_question_ehr(cur, qid)
        elif pid:
            from eval.tasks.diagnosis import _assemble_patient_ehr
            return _assemble_patient_ehr(cur, pid)

    elif task == "patient_diagnosis":
        cur.execute("SELECT patient_id FROM benchmark_ground_truth WHERE gt_id = %s", (gt_id,))
        row = cur.fetchone()
        if row and row[0]:
            from eval.tasks.patient_diagnosis import _assemble_patient_ehr
            return _assemble_patient_ehr(cur, row[0])

    elif task == "context_summarization":
        cur.execute("SELECT patient_id FROM benchmark_ground_truth WHERE gt_id = %s", (gt_id,))
        row = cur.fetchone()
        if row and row[0]:
            from eval.tasks.diagnosis import _assemble_patient_ehr
            return _assemble_patient_ehr(cur, row[0])

    elif task == "evidence_retrieval":
        return _assemble_retrieval_input_text(cur, gt_id)

    elif task == "imaging_indication":
        cur.execute("""
            SELECT le.patient_id, le.encounter_order
            FROM benchmark_ground_truth bgt
            JOIN longitudinal_encounters le ON bgt.encounter_id = le.encounter_id
            WHERE bgt.gt_id = %s
        """, (gt_id,))
        row = cur.fetchone()
        if row:
            from eval.tasks.imaging import _assemble_temporal_ehr
            return _assemble_temporal_ehr(cur, row[0], row[1])

    return ""


def _assemble_retrieval_input_text(cur, gt_id: int) -> str:
    """Build abbreviated retrieval input for few-shot example."""
    cur.execute("SELECT ground_truth FROM benchmark_ground_truth WHERE gt_id = %s", (gt_id,))
    row = cur.fetchone()
    if not row:
        return ""
    gt = row[0] if isinstance(row[0], dict) else json.loads(row[0])

    query_dx = gt.get("query_diagnoses", [])
    query_parts = []
    for dx in query_dx:
        if isinstance(dx, dict):
            dx_id = dx.get("diagnosis_id")
            if dx_id:
                cur.execute("SELECT display_name FROM diagnoses WHERE diagnosis_id = %s", (dx_id,))
                r = cur.fetchone()
                if r:
                    query_parts.append(r[0])

    return f"DIAGNOSIS: {', '.join(query_parts)}" if query_parts else "DIAGNOSIS: clinical diagnosis"


def _format_expected_output(task: str, gt: dict, conn=None, gt_id: int | None = None) -> dict:
    """Format GT into model prediction schema for few-shot examples."""
    if task == "diagnosis_accuracy":
        primary = gt.get("primary_diagnosis", {})
        diagnoses = [{
            "rank": 1,
            "icd10": primary.get("icd10", ""),
            "name": primary.get("display_name", ""),
            "confidence": 0.95,
        }]
        for i, dx in enumerate(gt.get("differential", [])[:4], start=2):
            diagnoses.append({
                "rank": i,
                "icd10": dx.get("icd10", ""),
                "name": dx.get("display_name", ""),
                "confidence": round(0.7 - 0.1 * (i - 2), 2),
            })
        return {"diagnoses": diagnoses}

    elif task == "patient_diagnosis":
        # Strip non-prediction fields from GT entries
        active = []
        for dx in gt.get("active_diagnoses", []):
            active.append({
                "icd10": dx.get("icd10", ""),
                "name": dx.get("display_name", ""),
                "acuity": dx.get("acuity", "acute"),
            })
        chronic = []
        for dx in gt.get("chronic_conditions", []):
            chronic.append({
                "icd10": dx.get("icd10", ""),
                "name": dx.get("display_name", ""),
                "acuity": dx.get("acuity", "chronic"),
            })
        return {"active_diagnoses": active, "chronic_conditions": chronic}

    elif task == "context_summarization":
        return {"summary": gt.get("reference_summary", "")}

    elif task == "evidence_retrieval":
        # Judgments are not in GT JSON — load from relevance_judgments table
        rankings = []
        if conn and gt_id:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT passage_id, relevance_grade
                    FROM relevance_judgments
                    WHERE gt_id = %s
                    ORDER BY relevance_grade DESC
                    LIMIT 10
                """, (gt_id,))
                rankings = [{"passage_id": pid, "grade": grade}
                            for pid, grade in cur.fetchall()]
        return {"rankings": rankings}

    elif task == "imaging_indication":
        return {
            "clinical_question": gt.get("inferred_clinical_question", ""),
            "pre_read_summary": gt.get("pre_read_summary", ""),
            "must_include_findings": gt.get("must_include_findings", []),
            "differential": gt.get("differential_context", []),
        }

    return {}
