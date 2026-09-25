"""Ground truth validation for benchmark_ground_truth rows.

Callable as:
    python -m eval.cli validate-gt [--task patient_diagnosis|patient_diagnosis]

Checks schema correctness, task/granularity alignment, and cross-task invariants.
"""

import json
import logging

from eval.tasks.diagnosis import normalize_icd10

log = logging.getLogger(__name__)


def validate_ground_truth(conn, task: str | None = None) -> list[str]:
    """Validate GT rows. Returns list of issue strings (empty = all OK)."""
    issues: list[str] = []

    tasks_to_check = []
    if task:
        tasks_to_check = [task]
    else:
        tasks_to_check = ["patient_diagnosis"]

    with conn.cursor() as cur:
        for t in tasks_to_check:
            cur.execute("""
                SELECT gt_id, task, granularity, question_id, patient_id, ground_truth
                FROM benchmark_ground_truth
                WHERE task = %s
                ORDER BY gt_id
            """, (t,))
            rows = cur.fetchall()

            if not rows:
                issues.append(f"No GT rows found for task={t}")
                continue

            for gt_id, task_val, gran, qid, pid, gt_json in rows:
                gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)

                if t == "diagnosis_accuracy":
                    issues.extend(_check_question_level(gt_id, gran, gt))
                elif t == "patient_diagnosis":
                    issues.extend(_check_patient_level(gt_id, gran, gt))

        # Cross-task: no NULL granularity
        cur.execute("""
            SELECT COUNT(*) FROM benchmark_ground_truth
            WHERE granularity IS NULL
        """)
        null_count = cur.fetchone()[0]
        if null_count > 0:
            issues.append(f"{null_count} rows have NULL granularity")

        # Cross-task: no NULL split
        cur.execute("""
            SELECT COUNT(*) FROM benchmark_ground_truth
            WHERE split IS NULL
        """)
        null_split = cur.fetchone()[0]
        if null_split > 0:
            issues.append(f"{null_split} rows have NULL split")

        # Cross-task: diagnosis_accuracy should be question-level only
        if not task or task == "diagnosis_accuracy":
            cur.execute("""
                SELECT COUNT(*) FROM benchmark_ground_truth
                WHERE task = 'diagnosis_accuracy' AND granularity != 'question'
            """)
            bad = cur.fetchone()[0]
            if bad > 0:
                issues.append(f"{bad} diagnosis_accuracy rows have non-question granularity")

        # Cross-task: patient_diagnosis should be patient-level only
        if not task or task == "patient_diagnosis":
            cur.execute("""
                SELECT COUNT(*) FROM benchmark_ground_truth
                WHERE task = 'patient_diagnosis' AND granularity != 'patient'
            """)
            bad = cur.fetchone()[0]
            if bad > 0:
                issues.append(f"{bad} patient_diagnosis rows have non-patient granularity")

    return issues


def _check_question_level(gt_id: int, gran: str, gt: dict) -> list[str]:
    """Validate a question-level diagnosis GT item."""
    issues = []
    if gran != "question":
        issues.append(f"gt_id={gt_id}: diagnosis_accuracy should have granularity='question', got '{gran}'")

    primary = gt.get("primary_diagnosis")
    if not primary:
        issues.append(f"gt_id={gt_id}: missing primary_diagnosis")
        return issues

    icd10 = primary.get("icd10", "")
    if not icd10:
        issues.append(f"gt_id={gt_id}: primary_diagnosis missing icd10")
    else:
        try:
            normalize_icd10(icd10)
        except Exception:
            issues.append(f"gt_id={gt_id}: invalid icd10 '{icd10}'")

    # Should NOT have patient-level keys
    if "active_diagnoses" in gt and "primary_diagnosis" not in gt:
        issues.append(f"gt_id={gt_id}: has active_diagnoses but no primary_diagnosis — may be mislabeled")

    return issues


def _check_patient_level(gt_id: int, gran: str, gt: dict) -> list[str]:
    """Validate a patient-level diagnosis GT item."""
    issues = []
    if gran != "patient":
        issues.append(f"gt_id={gt_id}: patient_diagnosis should have granularity='patient', got '{gran}'")

    active = gt.get("active_diagnoses")
    chronic = gt.get("chronic_conditions")

    if active is None:
        issues.append(f"gt_id={gt_id}: missing active_diagnoses key")
    if chronic is None:
        issues.append(f"gt_id={gt_id}: missing chronic_conditions key")

    if (active is not None and len(active) == 0) and (chronic is not None and len(chronic) == 0):
        issues.append(f"gt_id={gt_id}: both active_diagnoses and chronic_conditions are empty")

    # Validate each entry
    for label, entries in [("active_diagnoses", active or []), ("chronic_conditions", chronic or [])]:
        seen_categories = set()
        for i, entry in enumerate(entries):
            icd10 = entry.get("icd10", "")
            if not icd10:
                issues.append(f"gt_id={gt_id}: {label}[{i}] missing icd10")
            else:
                norm = normalize_icd10(icd10)
                cat = norm[:3]
                if cat in seen_categories:
                    issues.append(f"gt_id={gt_id}: duplicate 3-char category {cat} in {label}")
                seen_categories.add(cat)

            if not entry.get("display_name"):
                issues.append(f"gt_id={gt_id}: {label}[{i}] missing display_name")

            # Acuity checks
            if label == "active_diagnoses":
                acuity = entry.get("acuity", "")
                if acuity and acuity not in ("acute", "acute_on_chronic", "unspecified"):
                    issues.append(f"gt_id={gt_id}: {label}[{i}] unexpected acuity '{acuity}'")

    # Should NOT have question-level key
    if "primary_diagnosis" in gt:
        issues.append(f"gt_id={gt_id}: patient-level GT has primary_diagnosis key — may be mislabeled")

    return issues
