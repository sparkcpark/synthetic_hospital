"""Error taxonomy and classification for evaluation predictions.

Deterministic, post-scoring step — no LLM calls. Classifies each prediction
into one or more error categories per the spec §E1.1b taxonomy:

  PARSE_FAILURE, SCHEMA_MISMATCH, CODING_ERROR,
  REASONING_ERROR, HALLUCINATION, OMISSION

Usage:
    from eval.errors import classify_errors
    errors = classify_errors(task, gt, prediction, ehr_text, score)
"""

import logging
import re
from enum import Enum

log = logging.getLogger(__name__)

# ============================================================================
# Enums
# ============================================================================


class ErrorCategory(str, Enum):
    PARSE_FAILURE = "parse_failure"
    SCHEMA_MISMATCH = "schema_mismatch"
    CODING_ERROR = "coding_error"
    REASONING_ERROR = "reasoning_error"
    HALLUCINATION = "hallucination"
    OMISSION = "omission"


class ParseFailureSubtype(str, Enum):
    NO_JSON = "no_json"
    INVALID_JSON = "invalid_json"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED = "truncated"
    WRONG_STRUCTURE = "wrong_structure"


class SchemaMismatchSubtype(str, Enum):
    MISSING_REQUIRED_KEY = "missing_required_key"
    WRONG_KEY_NAME = "wrong_key_name"
    WRONG_VALUE_TYPE = "wrong_value_type"
    EXTRA_NESTING = "extra_nesting"


class CodingErrorSubtype(str, Enum):
    WRONG_SPECIFICITY = "wrong_specificity"
    PARENT_CODE = "parent_code"
    ADJACENT_CODE = "adjacent_code"
    WRONG_CHAPTER = "wrong_chapter"
    NONEXISTENT_CODE = "nonexistent_code"
    FORMAT_ERROR = "format_error"


class ReasoningErrorSubtype(str, Enum):
    WRONG_DIAGNOSIS = "wrong_diagnosis"
    RELATED_DIFFERENTIAL = "related_differential"
    ACUITY_MISCLASS = "acuity_misclass"
    MISSED_COMORBIDITY = "missed_comorbidity"
    GRADE_INFLATION = "grade_inflation"
    GRADE_DEFLATION = "grade_deflation"


# ============================================================================
# ICD-10 Dictionary Singleton
# ============================================================================

_icd10_dict = None


def _get_icd10_dict():
    """Lazy-load ICD10Dictionary singleton."""
    global _icd10_dict
    if _icd10_dict is None:
        from etl.ontology.icd10 import ICD10Dictionary
        _icd10_dict = ICD10Dictionary()
    return _icd10_dict


def _normalize_icd10(code: str) -> str:
    """Strip dots, uppercase."""
    return code.replace(".", "").replace(" ", "").upper() if code else ""


# ============================================================================
# Helpers
# ============================================================================

def _err(category: ErrorCategory, subtype=None, detail: str = "") -> dict:
    """Build a single error entry."""
    e = {"category": category.value}
    if subtype is not None:
        e["subtype"] = subtype.value if isinstance(subtype, Enum) else subtype
    if detail:
        e["detail"] = detail
    return e


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter (reuses logic from scoring.py)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]


# ============================================================================
# Dispatcher
# ============================================================================

def classify_errors(task: str, gt: dict, prediction: dict,
                    ehr_text: str = "", score: float | None = None,
                    **kwargs) -> list[dict]:
    """Classify errors for a single prediction.

    Returns list of error dicts. Empty list = correct prediction.
    """
    if task == "diagnosis_accuracy":
        return _classify_diagnosis(gt, prediction)
    elif task == "patient_diagnosis":
        return _classify_patient_diagnosis(gt, prediction, ehr_text)
    elif task == "context_summarization":
        return _classify_summarization(gt, prediction, ehr_text)
    elif task == "evidence_retrieval":
        return _classify_retrieval(gt, prediction, **kwargs)
    elif task == "imaging_indication":
        return _classify_imaging(gt, prediction, ehr_text)
    else:
        log.warning("Unknown task for error classification: %s", task)
        return []


# ============================================================================
# Task 1: Diagnosis Accuracy
# ============================================================================

def _classify_diagnosis(gt: dict, pred: dict) -> list[dict]:
    errors = []

    # 1. Parse failure
    if pred.get("parse_error"):
        return [_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.NO_JSON,
                      pred.get("parse_error", ""))]

    diagnoses = pred.get("diagnoses")
    if not diagnoses:
        return [_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.WRONG_STRUCTURE,
                      "empty or missing diagnoses list")]

    # 2. Schema check
    if not isinstance(diagnoses, list):
        return [_err(ErrorCategory.SCHEMA_MISMATCH, SchemaMismatchSubtype.MISSING_REQUIRED_KEY,
                      "diagnoses is not a list")]

    # Build acceptable set
    gt_primary = gt.get("primary_diagnosis", {})
    gt_icd = _normalize_icd10(gt_primary.get("icd10", ""))
    if not gt_icd:
        return []  # no GT to compare

    acceptable = {gt_icd}
    for alt in gt.get("acceptable_alternatives", []):
        alt_icd = _normalize_icd10(alt.get("icd10", ""))
        if alt_icd:
            acceptable.add(alt_icd)

    # Get top-1 prediction
    top1_dx = diagnoses[0] if diagnoses else {}
    top1_raw = top1_dx.get("icd10", "")
    top1 = _normalize_icd10(top1_raw)

    if not top1:
        return [_err(ErrorCategory.SCHEMA_MISMATCH, SchemaMismatchSubtype.MISSING_REQUIRED_KEY,
                      "top-1 diagnosis missing icd10")]

    # 3. Correct?
    if top1 in acceptable:
        return []

    # 4. Nonexistent code
    icd_dict = _get_icd10_dict()
    entry = icd_dict.lookup_code(top1_raw)
    if entry is None:
        errors.append(_err(ErrorCategory.CODING_ERROR, CodingErrorSubtype.NONEXISTENT_CODE,
                           f"code={top1_raw}"))

    # 5. Header (non-billable) code
    if entry is not None and entry.is_header:
        errors.append(_err(ErrorCategory.CODING_ERROR, CodingErrorSubtype.PARENT_CODE,
                           f"code={top1_raw} is header"))

    # 6. Same 3-char category = wrong specificity
    if entry is not None and not entry.is_header and top1[:3] == gt_icd[:3]:
        errors.append(_err(ErrorCategory.CODING_ERROR, CodingErrorSubtype.WRONG_SPECIFICITY,
                           f"pred={top1} gt={gt_icd}"))
        return errors

    # 7. Distractor match
    differential = gt.get("differential", [])
    distractor_codes = {_normalize_icd10(d.get("icd10", "")) for d in differential}
    # Also include secondary diagnoses as distractors
    for sec in gt.get("secondary_diagnoses", []):
        sec_code = _normalize_icd10(sec.get("icd10", ""))
        if sec_code and sec_code != gt_icd:
            distractor_codes.add(sec_code)

    if top1 in distractor_codes:
        errors.append(_err(ErrorCategory.REASONING_ERROR,
                           ReasoningErrorSubtype.RELATED_DIFFERENTIAL,
                           f"pred={top1} is a distractor"))
        return errors

    # 8. Otherwise: wrong diagnosis
    if not errors:
        errors.append(_err(ErrorCategory.REASONING_ERROR,
                           ReasoningErrorSubtype.WRONG_DIAGNOSIS,
                           f"pred={top1} gt={gt_icd}"))

    return errors


# ============================================================================
# Task 2: Patient Diagnosis
# ============================================================================

def _classify_patient_diagnosis(gt: dict, pred: dict, ehr_text: str) -> list[dict]:
    from eval.scoring import _match_icd10_sets, _normalize_acuity

    errors = []

    # 1. Parse tier check
    parse_tier = pred.get("_parse_tier", "")

    # Check for unparseable (Tier C)
    pred_active = pred.get("active_diagnoses", [])
    pred_chronic = pred.get("chronic_conditions", [])
    flat_dx = pred.get("diagnoses", [])

    if not pred_active and not pred_chronic and not flat_dx:
        if pred.get("parse_error") or parse_tier == "unparseable":
            return [_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.NO_JSON,
                         "unparseable prediction")]
        return [_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.WRONG_STRUCTURE,
                     "no diagnoses in prediction")]

    # 2. Partial match (Tier B — flat list only)
    if not pred_active and not pred_chronic and flat_dx:
        errors.append(_err(ErrorCategory.SCHEMA_MISMATCH,
                           SchemaMismatchSubtype.MISSING_REQUIRED_KEY,
                           "flat diagnoses list, missing active/chronic split"))
        pred_active = flat_dx

    # Build predicted codes
    pred_codes = []
    pred_acuities = []
    for dx in pred_active:
        pred_codes.append(dx.get("icd10", ""))
        pred_acuities.append(_normalize_acuity(dx.get("acuity", "")) or "acute")
    for dx in pred_chronic:
        pred_codes.append(dx.get("icd10", ""))
        pred_acuities.append("chronic")

    # Build GT codes
    gt_codes = []
    gt_acuities = []
    for dx in gt.get("active_diagnoses", []):
        gt_codes.append(dx.get("icd10", ""))
        acuity = dx.get("acuity", "unspecified")
        if acuity == "unspecified":
            acuity = "acute"
        gt_acuities.append(acuity)
    for dx in gt.get("chronic_conditions", []):
        gt_codes.append(dx.get("icd10", ""))
        gt_acuities.append("chronic")

    if not gt_codes:
        return errors  # no GT to compare

    # 3. Check for nonexistent codes
    icd_dict = _get_icd10_dict()
    for code in pred_codes[:20]:  # cap check
        if not code:
            continue
        entry = icd_dict.lookup_code(code)
        if entry is None:
            errors.append(_err(ErrorCategory.CODING_ERROR,
                               CodingErrorSubtype.NONEXISTENT_CODE,
                               f"code={code}"))
            if len([e for e in errors if e.get("subtype") == "nonexistent_code"]) >= 3:
                break

    # 4. ICD-10 set matching
    matched, unmatched_pred, unmatched_gt = _match_icd10_sets(pred_codes, gt_codes)

    # 5. Acuity mismatches on matched pairs
    gt_code_to_acuity = dict(zip(
        [_normalize_icd10(c) for c in gt_codes], gt_acuities
    ))
    pred_code_to_acuity = dict(zip(
        [_normalize_icd10(c) for c in pred_codes], pred_acuities
    ))

    acuity_errors = 0
    for pred_c, gt_c in matched:
        p_acuity = _normalize_acuity(pred_code_to_acuity.get(pred_c, ""))
        g_acuity = _normalize_acuity(gt_code_to_acuity.get(gt_c, ""))
        if p_acuity and g_acuity and p_acuity != g_acuity:
            if acuity_errors < 5:
                gt_name = _icd10_desc(gt_c)
                errors.append(_err(ErrorCategory.REASONING_ERROR,
                                   ReasoningErrorSubtype.ACUITY_MISCLASS,
                                   f"{gt_name}: pred={p_acuity} gt={g_acuity}"))
            acuity_errors += 1

    # 6. Unmatched GT = omissions
    omission_count = 0
    for gt_c in unmatched_gt:
        if omission_count >= 5:
            break
        gt_name = _icd10_desc(gt_c)
        errors.append(_err(ErrorCategory.OMISSION, detail=f"missed {gt_name} ({gt_c})"))
        omission_count += 1

    # 7/8. Unmatched pred: check if hallucination or reasoning error
    ehr_lower = ehr_text.lower() if ehr_text else ""
    halluc_count = 0
    reason_count = 0
    for pred_c in unmatched_pred:
        if halluc_count + reason_count >= 5:
            break
        pred_name = _icd10_desc(pred_c)
        if ehr_lower and pred_name and pred_name.lower() in ehr_lower:
            errors.append(_err(ErrorCategory.REASONING_ERROR,
                               ReasoningErrorSubtype.WRONG_DIAGNOSIS,
                               f"pred={pred_name} ({pred_c}) found in EHR"))
            reason_count += 1
        else:
            errors.append(_err(ErrorCategory.HALLUCINATION,
                               detail=f"pred={pred_name} ({pred_c}) not in EHR"))
            halluc_count += 1

    return errors


def _icd10_desc(code: str) -> str:
    """Get ICD-10 description for a normalized code, or return code."""
    icd_dict = _get_icd10_dict()
    # Try with dot format
    if len(code) > 3 and "." not in code:
        dotted = code[:3] + "." + code[3:]
    else:
        dotted = code
    entry = icd_dict.lookup_code(dotted)
    if entry:
        return entry.description
    return code


# ============================================================================
# Task 3: Context Summarization
# ============================================================================

def _classify_summarization(gt: dict, pred: dict, ehr_text: str) -> list[dict]:
    errors = []

    # 1. Schema check
    summary = pred.get("summary")
    if summary is None or (isinstance(summary, str) and not summary.strip()):
        if pred.get("parse_error"):
            return [_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.EMPTY_RESPONSE)]
        return [_err(ErrorCategory.SCHEMA_MISMATCH,
                     SchemaMismatchSubtype.MISSING_REQUIRED_KEY,
                     "missing or empty summary")]

    if not isinstance(summary, str):
        summary = str(summary)

    # 2. Parse error flag
    if pred.get("parse_error"):
        errors.append(_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.EMPTY_RESPONSE,
                           pred.get("parse_error", "")))

    # 3. Truncated (1 sentence or less)
    sentences = _split_sentences(summary)
    if len(sentences) <= 1:
        errors.append(_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.TRUNCATED,
                           f"only {len(sentences)} sentence(s)"))

    # 4. Hallucination: per-sentence Jaccard overlap with EHR < 0.15
    if ehr_text and sentences:
        ehr_tokens = set(ehr_text.lower().split())
        halluc_count = 0
        for sent in sentences:
            sent_tokens = set(sent.lower().split())
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & ehr_tokens) / len(sent_tokens)
            if overlap < 0.15:
                if halluc_count < 3:
                    errors.append(_err(ErrorCategory.HALLUCINATION,
                                       detail=sent[:120]))
                halluc_count += 1

    # 5. Omission: must_include_findings not in summary
    must_include = gt.get("must_include_findings", [])
    summary_lower = summary.lower()
    omission_count = 0
    for finding in must_include:
        name = finding.get("display_name", "").lower()
        if name and name not in summary_lower:
            if omission_count < 5:
                errors.append(_err(ErrorCategory.OMISSION,
                                   detail=finding.get("display_name", "")))
            omission_count += 1

    return errors


# ============================================================================
# Task 4: Evidence Retrieval
# ============================================================================

def _classify_retrieval(gt: dict, pred: dict, **kwargs) -> list[dict]:
    errors = []
    judgments = kwargs.get("judgments", {})

    # 1. Parse / schema check
    rankings = pred.get("rankings")
    if pred.get("parse_error"):
        return [_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.NO_JSON,
                     pred.get("parse_error", ""))]
    if rankings is None or not isinstance(rankings, list):
        return [_err(ErrorCategory.SCHEMA_MISMATCH,
                     SchemaMismatchSubtype.MISSING_REQUIRED_KEY,
                     "missing or invalid rankings")]

    if not judgments:
        return []  # baseline methods — no grade to compare

    # 2. Invalid passage IDs
    invalid_count = 0
    for item in rankings:
        pid = item.get("passage_id", "")
        if pid and pid not in judgments:
            if invalid_count < 3:
                errors.append(_err(ErrorCategory.SCHEMA_MISMATCH,
                                   SchemaMismatchSubtype.WRONG_KEY_NAME,
                                   f"unknown passage_id={pid}"))
            invalid_count += 1

    # 3/4. Grade inflation / deflation
    inflation_count = 0
    deflation_count = 0
    for item in rankings:
        pid = item.get("passage_id", "")
        pred_grade = item.get("grade", 0)
        if not isinstance(pred_grade, (int, float)):
            try:
                pred_grade = int(pred_grade)
            except (ValueError, TypeError):
                pred_grade = 0
        gt_grade = judgments.get(pid, 0)

        if pred_grade >= 3 and gt_grade <= 0:
            if inflation_count < 5:
                errors.append(_err(ErrorCategory.REASONING_ERROR,
                                   ReasoningErrorSubtype.GRADE_INFLATION,
                                   f"passage={pid} pred={pred_grade} gt={gt_grade}"))
            inflation_count += 1
        elif pred_grade <= 0 and gt_grade >= 3:
            if deflation_count < 5:
                errors.append(_err(ErrorCategory.REASONING_ERROR,
                                   ReasoningErrorSubtype.GRADE_DEFLATION,
                                   f"passage={pid} pred={pred_grade} gt={gt_grade}"))
            deflation_count += 1

    return errors


# ============================================================================
# Task 5: Imaging Indication
# ============================================================================

def _classify_imaging(gt: dict, pred: dict, ehr_text: str) -> list[dict]:
    errors = []

    # 1. Schema check
    cq = pred.get("clinical_question")
    if not cq or (isinstance(cq, str) and not cq.strip()):
        if pred.get("parse_error"):
            return [_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.NO_JSON,
                         pred.get("parse_error", ""))]
        return [_err(ErrorCategory.SCHEMA_MISMATCH,
                     SchemaMismatchSubtype.MISSING_REQUIRED_KEY,
                     "missing clinical_question")]

    # 2. Parse error
    if pred.get("parse_error"):
        errors.append(_err(ErrorCategory.PARSE_FAILURE, ParseFailureSubtype.INVALID_JSON,
                           pred.get("parse_error", "")))

    # 3. Differential ICD-10 codes: check existence
    icd_dict = _get_icd10_dict()
    pred_diff = pred.get("differential", [])
    if isinstance(pred_diff, list):
        coding_err_count = 0
        for dx in pred_diff:
            code = dx.get("icd10", "")
            if code and icd_dict.lookup_code(code) is None:
                if coding_err_count < 3:
                    errors.append(_err(ErrorCategory.CODING_ERROR,
                                       CodingErrorSubtype.NONEXISTENT_CODE,
                                       f"code={code}"))
                coding_err_count += 1

    # 4. Omission: GT must_include_findings not in prediction
    gt_findings = gt.get("must_include_findings", [])
    # Build combined prediction text for searching
    pred_text_parts = [
        pred.get("clinical_question", ""),
        pred.get("pre_read_summary", ""),
    ]
    pred_must_include = pred.get("must_include_findings", [])
    if isinstance(pred_must_include, list):
        for f in pred_must_include:
            if isinstance(f, str):
                pred_text_parts.append(f)
            elif isinstance(f, dict):
                pred_text_parts.append(f.get("display_name", ""))
    combined_pred_lower = " ".join(pred_text_parts).lower()

    omission_count = 0
    for finding in gt_findings:
        if isinstance(finding, str):
            fname = finding.lower()
        elif isinstance(finding, dict):
            fname = finding.get("display_name", "").lower()
        else:
            continue
        if fname and fname not in combined_pred_lower:
            # Check key terms (first 3 words) as relaxed match
            key_terms = fname.split()[:3]
            if not all(t in combined_pred_lower for t in key_terms):
                if omission_count < 5:
                    errors.append(_err(ErrorCategory.OMISSION, detail=fname[:120]))
                omission_count += 1

    # 5. Hallucination: predicted findings not in EHR
    if ehr_text and isinstance(pred_must_include, list):
        ehr_lower = ehr_text.lower()
        halluc_count = 0
        for f in pred_must_include:
            if isinstance(f, str):
                fname = f.lower()
            elif isinstance(f, dict):
                fname = f.get("display_name", "").lower()
            else:
                continue
            if not fname:
                continue
            # Check key terms in EHR
            key_terms = fname.split()[:3]
            if not any(t in ehr_lower for t in key_terms if len(t) > 3):
                if halluc_count < 3:
                    errors.append(_err(ErrorCategory.HALLUCINATION, detail=fname[:120]))
                halluc_count += 1

    return errors
