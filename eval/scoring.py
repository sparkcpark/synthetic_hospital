"""Scoring engine: task-specific and cross-task metrics.

Task-specific:
  - Diagnosis: top-k accuracy, hierarchical F1, MRR
  - Summarization: ROUGE-L, BERTScore, clinical F1, hallucination rate, omission rate
  - Retrieval: NDCG@10, MAP@10, Recall@K, Precision@5, MRR, latency-adjusted NDCG
  - Imaging: clinical question F1, summary ROUGE-L, differential coverage, findings recall

Cross-task:
  - Cost-normalized performance
  - Inter-model agreement (Cohen's kappa)
"""

import json
import logging
import math
import re
from collections import defaultdict

import numpy as np

from eval import semantic_match, value_match

log = logging.getLogger(__name__)


# Acuity weights for acuity-weighted recall (higher-severity findings count more).
_ACUITY_WEIGHTS = {
    "acute": 1.0,
    "acute_on_chronic": 1.0,
    "chronic": 0.6,
    "unspecified": 0.8,
}


def _finding_names(findings: list) -> list[str]:
    """Extract display names from a findings list (dicts or bare strings).

    Entries marked `excluded_nondiagnostic` are skipped: these are non-clinical
    concepts (study design, statistics, ethics) that leaked in from non-diagnostic
    board questions and are not valid targets for chart-grounded reasoning.
    """
    names: list[str] = []
    for f in findings or []:
        if isinstance(f, str):
            if f:
                names.append(f)
        elif isinstance(f, dict):
            if f.get("excluded_nondiagnostic"):
                continue
            n = f.get("display_name") or f.get("name") or ""
            if n:
                names.append(n)
    return names


def _list_recall(summaries: list[str], findings_lists: list[list]) -> float:
    """Mean per-item recall of a findings list within each summary (semantic).

    Items whose findings list is empty are skipped (no denominator).
    """
    recalls = []
    for summary, findings in zip(summaries, findings_lists):
        names = _finding_names(findings)
        if not names:
            continue
        hits = semantic_match.count_present(names, summary or "")
        recalls.append(hits / len(names))
    return float(np.mean(recalls)) if recalls else 0.0


def _cascade_present(names: list[str], summary: str) -> tuple[int, int]:
    """Matcher cascade for one summary: lexical-AND-not-negated first, then the
    polarity-safe value-normalizer on the residual. Returns (n_matched, n_value_added)."""
    summary = summary or ""
    matched = value_added = 0
    for name in names:
        if semantic_match.phrase_in_text(name, summary) and not value_match.negated_mention(name, summary):
            matched += 1
        elif value_match.value_polarity_match(name, summary):
            matched += 1
            value_added += 1
    return matched, value_added


def _list_recall_cascade(summaries: list[str], findings_lists: list[list]) -> tuple[float, int]:
    """Cascade (lexical+negation+value-normalizer) mean recall + total value-normalizer credits."""
    recalls, added = [], 0
    for summary, findings in zip(summaries, findings_lists):
        names = _finding_names(findings)
        if not names:
            continue
        n, a = _cascade_present(names, summary)
        recalls.append(n / len(names))
        added += a
    return (float(np.mean(recalls)) if recalls else 0.0), added


# ============================================================================
# DIAGNOSIS METRICS
# ============================================================================

def _normalize_icd10(code: str) -> str:
    """Strip dots, uppercase."""
    return code.replace(".", "").replace(" ", "").upper() if code else ""


def top_k_accuracy(predictions: list[dict], ground_truths: list[dict], k: int = 1) -> float:
    """Exact ICD-10 match within top-k predicted diagnoses."""
    correct = 0
    total = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_primary = gt.get("primary_diagnosis", {})
        gt_icd = _normalize_icd10(gt_primary.get("icd10", ""))
        if not gt_icd:
            continue
        total += 1

        # Also accept alternatives
        acceptable = {gt_icd}
        for alt in gt.get("acceptable_alternatives", []):
            alt_icd = _normalize_icd10(alt.get("icd10", ""))
            if alt_icd:
                acceptable.add(alt_icd)

        pred_diagnoses = pred.get("diagnoses", [])
        for dx in pred_diagnoses[:k]:
            pred_icd = _normalize_icd10(dx.get("icd10", ""))
            if pred_icd in acceptable:
                correct += 1
                break

    return correct / total if total > 0 else 0.0


def hierarchical_f1(predictions: list[dict], ground_truths: list[dict]) -> float:
    """ICD-10 chapter-level (first 3 chars) match F1."""
    tp = fp = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_primary = gt.get("primary_diagnosis", {})
        gt_chapter = _normalize_icd10(gt_primary.get("icd10", ""))[:3]
        if not gt_chapter:
            continue

        pred_diagnoses = pred.get("diagnoses", [])
        pred_chapters = set()
        for dx in pred_diagnoses[:5]:
            ch = _normalize_icd10(dx.get("icd10", ""))[:3]
            if ch:
                pred_chapters.add(ch)

        if gt_chapter in pred_chapters:
            tp += 1
        else:
            fn += 1
        # FP: predicted chapters not matching GT
        fp += len(pred_chapters - {gt_chapter})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(predictions: list[dict], ground_truths: list[dict]) -> float:
    """MRR: 1/rank of first correct ICD-10 in ranked list."""
    rr_sum = 0.0
    total = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_primary = gt.get("primary_diagnosis", {})
        gt_icd = _normalize_icd10(gt_primary.get("icd10", ""))
        if not gt_icd:
            continue
        total += 1

        acceptable = {gt_icd}
        for alt in gt.get("acceptable_alternatives", []):
            alt_icd = _normalize_icd10(alt.get("icd10", ""))
            if alt_icd:
                acceptable.add(alt_icd)

        for rank, dx in enumerate(pred.get("diagnoses", []), start=1):
            pred_icd = _normalize_icd10(dx.get("icd10", ""))
            if pred_icd in acceptable:
                rr_sum += 1.0 / rank
                break

    return rr_sum / total if total > 0 else 0.0


# ============================================================================
# PATIENT DIAGNOSIS METRICS
# ============================================================================

class SeverityTier:
    CRITICAL = 3.0
    MODERATE = 2.0
    ROUTINE = 1.0


CRITICAL_CODES_3CHAR = {
    "A41", "I21", "I26", "I63", "I60", "I61",
    "J96", "N17", "K72", "E87", "T78",
}

CRITICAL_CHAPTERS = {"A", "B", "C", "D", "I", "J", "S", "T"}


def _normalize_icd10_category(code: str) -> str:
    """Normalize and extract 3-char ICD-10 category."""
    norm = _normalize_icd10(code)
    return norm[:3] if len(norm) >= 3 else norm


def _assign_severity_tier(icd10_code: str, acuity: str) -> float:
    """Assign severity tier weight based on ICD-10 category and acuity."""
    norm = _normalize_icd10(icd10_code)
    cat3 = norm[:3] if len(norm) >= 3 else norm
    first_char = norm[0] if norm else ""

    if cat3 in CRITICAL_CODES_3CHAR:
        return SeverityTier.CRITICAL

    is_active = acuity in ("acute", "acute_on_chronic")
    if is_active and first_char in CRITICAL_CHAPTERS:
        return SeverityTier.CRITICAL
    if is_active and first_char not in CRITICAL_CHAPTERS:
        return SeverityTier.MODERATE
    if not is_active and first_char in CRITICAL_CHAPTERS:
        return SeverityTier.MODERATE

    return SeverityTier.ROUTINE


def _match_icd10_sets(
    pred_codes: list[str], gt_codes: list[str]
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Match predicted to GT codes at 3-char category level.

    Returns (matched_pairs, unmatched_pred, unmatched_gt).
    Greedy matching: prefer exact full-code match, then longest prefix.
    """
    pred_norm = [_normalize_icd10(c) for c in pred_codes]
    gt_norm = [_normalize_icd10(c) for c in gt_codes]

    matched: list[tuple[str, str]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    # Pass 1: exact full-code match
    for gi, gc in enumerate(gt_norm):
        if gi in used_gt:
            continue
        for pi, pc in enumerate(pred_norm):
            if pi in used_pred:
                continue
            if pc == gc:
                matched.append((pc, gc))
                used_pred.add(pi)
                used_gt.add(gi)
                break

    # Pass 2: 3-char category match (longest shared prefix as tiebreaker)
    for gi, gc in enumerate(gt_norm):
        if gi in used_gt:
            continue
        gc_cat = gc[:3]
        # Injury/poisoning: use 5-char dedup
        dedup_len = 5 if gc_cat and gc_cat[0] in ("S", "T") else 3
        gc_key = gc[:dedup_len]

        best_pi = None
        best_prefix_len = -1
        for pi, pc in enumerate(pred_norm):
            if pi in used_pred:
                continue
            pc_key = pc[:dedup_len]
            if pc_key == gc_key:
                prefix_len = len(_shared_prefix(pc, gc))
                if prefix_len > best_prefix_len:
                    best_prefix_len = prefix_len
                    best_pi = pi

        if best_pi is not None:
            matched.append((pred_norm[best_pi], gc))
            used_pred.add(best_pi)
            used_gt.add(gi)

    unmatched_pred = [pred_norm[i] for i in range(len(pred_norm)) if i not in used_pred]
    unmatched_gt = [gt_norm[i] for i in range(len(gt_norm)) if i not in used_gt]

    return matched, unmatched_pred, unmatched_gt


def _shared_prefix(a: str, b: str) -> str:
    """Return the shared prefix of two strings."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def _icd10_specificity_score(matched_pairs: list[tuple[str, str]]) -> float | None:
    """Mean character-level agreement beyond the 3-char prefix."""
    if not matched_pairs:
        return None

    scores = []
    for pred, gt in matched_pairs:
        # If GT ends in 9 at subcategory (unspecified convention), full credit
        if len(gt) >= 4 and gt[3] == "9":
            scores.append(1.0)
            continue
        prefix_len = len(_shared_prefix(pred, gt))
        max_len = max(len(pred), len(gt))
        scores.append(prefix_len / max_len if max_len > 0 else 1.0)

    return float(np.mean(scores))


def problem_list_recall(pred_codes: list[str], gt_codes: list[str]) -> float:
    """Fraction of GT diagnoses found in predicted set at 3-char category level."""
    if not gt_codes:
        return 1.0
    matched, _, _ = _match_icd10_sets(pred_codes, gt_codes)
    return len(matched) / len(gt_codes)


def problem_list_precision(pred_codes: list[str], gt_codes: list[str]) -> float:
    """Fraction of predicted diagnoses matching any GT diagnosis."""
    if not pred_codes:
        return 1.0 if not gt_codes else 0.0
    matched, _, _ = _match_icd10_sets(pred_codes, gt_codes)
    return len(matched) / len(pred_codes)


def problem_list_f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision."""
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def weighted_problem_list_recall(
    matched_gt_codes: list[str], matched_gt_acuities: list[str],
    all_gt_codes: list[str], all_gt_acuities: list[str],
) -> float:
    """Severity-weighted recall."""
    if not all_gt_codes:
        return 1.0
    total_weight = sum(
        _assign_severity_tier(c, a) for c, a in zip(all_gt_codes, all_gt_acuities)
    )
    matched_weight = sum(
        _assign_severity_tier(c, a) for c, a in zip(matched_gt_codes, matched_gt_acuities)
    )
    return matched_weight / total_weight if total_weight > 0 else 0.0


ACUITY_NORMALIZATION = {
    "acute": "acute",
    "chronic": "chronic",
    "acute_on_chronic": "acute_on_chronic",
    "subacute": "acute",
    "resolving": "acute",
    "stable": "chronic",
    "exacerbation": "acute_on_chronic",
    "flare": "acute_on_chronic",
    "acute on chronic": "acute_on_chronic",
    "acute-on-chronic": "acute_on_chronic",
}


def _normalize_acuity(raw: str | None) -> str | None:
    """Normalize an acuity label. Returns None if unrecognized."""
    if not raw:
        return None
    return ACUITY_NORMALIZATION.get(raw.lower().strip())


def acuity_accuracy(
    matched_pairs_with_labels: list[tuple[str, str]]
) -> float | None:
    """Among matched diagnoses with valid acuity labels, fraction correct.

    Each tuple is (predicted_acuity, gt_acuity) already normalized.
    """
    eligible = [(p, g) for p, g in matched_pairs_with_labels if p and g]
    if not eligible:
        return None
    correct = sum(1 for p, g in eligible if p == g)
    return correct / len(eligible)


def acuity_confusion_matrix(
    matched_pairs_with_labels: list[tuple[str, str]]
) -> dict[str, int]:
    """3x3 confusion matrix. Keys are 'gt_label:pred_label'."""
    matrix: dict[str, int] = defaultdict(int)
    for pred_acuity, gt_acuity in matched_pairs_with_labels:
        if pred_acuity and gt_acuity:
            matrix[f"{gt_acuity}:{pred_acuity}"] += 1
    return dict(matrix)


def _compute_patient_diagnosis_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """Compute all metrics for patient_diagnosis task."""
    all_recalls = []
    all_precisions = []
    all_weighted_recalls = []
    all_specificity = []
    all_acuity_pairs: list[tuple[str, str]] = []
    n_scored = 0
    n_tier_c = 0
    # Chart-neutral precision (paper's primary rule): unmatched predictions whose
    # 3-char category lies in the patient's chart-neutral set (gt["_neutral_categories"],
    # see eval/chart_neutral.py) are dropped from the precision denominator.
    all_precisions_neutral: list[float] = []
    n_neutral_predictions = 0

    for pred, gt in zip(predictions, ground_truths):
        # Extract GT codes and acuities
        # Skip entries marked `excluded_nondiagnostic` (non-clinical concepts such as
        # "Odds ratio" or "Berkson bias" that leaked in from non-diagnostic board
        # questions); they are not valid problem-list targets.
        gt_active = [d for d in gt.get("active_diagnoses", [])
                     if not (isinstance(d, dict) and d.get("excluded_nondiagnostic"))]
        gt_chronic = [d for d in gt.get("chronic_conditions", [])
                      if not (isinstance(d, dict) and d.get("excluded_nondiagnostic"))]

        gt_codes = []
        gt_acuities = []
        for dx in gt_active:
            gt_codes.append(dx.get("icd10", ""))
            acuity = dx.get("acuity", "unspecified")
            # unspecified in active list → treat as acute
            if acuity == "unspecified":
                acuity = "acute"
            gt_acuities.append(acuity)
        for dx in gt_chronic:
            gt_codes.append(dx.get("icd10", ""))
            gt_acuities.append("chronic")

        # Extract predicted codes and acuities
        pred_active = pred.get("active_diagnoses", [])
        pred_chronic = pred.get("chronic_conditions", [])

        if not pred_active and not pred_chronic:
            # Check for flat "diagnoses" list (Tier B fallback)
            flat = pred.get("diagnoses", [])
            if flat:
                pred_active = flat
            else:
                n_tier_c += 1
                continue

        pred_codes = []
        pred_acuities = []
        for dx in pred_active:
            pred_codes.append(dx.get("icd10", ""))
            pred_acuities.append(_normalize_acuity(dx.get("acuity", "")) or "acute")
        for dx in pred_chronic:
            pred_codes.append(dx.get("icd10", ""))
            pred_acuities.append("chronic")

        n_scored += 1

        # ICD-10 matching
        matched, unmatched_pred, unmatched_gt = _match_icd10_sets(pred_codes, gt_codes)
        rec = problem_list_recall(pred_codes, gt_codes)
        prec = problem_list_precision(pred_codes, gt_codes)
        all_recalls.append(rec)
        all_precisions.append(prec)

        neutral = set(gt.get("_neutral_categories") or [])
        if neutral:
            neutral_unmatched = [u for u in unmatched_pred if u[:3] in neutral]
            n_neutral_predictions += len(neutral_unmatched)
            denom = len(matched) + len(unmatched_pred) - len(neutral_unmatched)
            # Undefined (every prediction matched-or-neutral with no match at all):
            # excluded from the mean, exactly as in the paper's rescoring.
            all_precisions_neutral.append(len(matched) / denom if denom else float("nan"))
        else:
            all_precisions_neutral.append(prec)

        # Specificity
        spec = _icd10_specificity_score(matched)
        if spec is not None:
            all_specificity.append(spec)

        # Weighted recall: find matched GT entries
        gt_code_to_acuity = dict(zip(
            [_normalize_icd10(c) for c in gt_codes], gt_acuities
        ))
        matched_gt_codes_list = [gt_c for _, gt_c in matched]
        matched_gt_acuities_list = [gt_code_to_acuity.get(gt_c, "chronic") for gt_c in matched_gt_codes_list]
        all_gt_norm = [_normalize_icd10(c) for c in gt_codes]

        w_rec = weighted_problem_list_recall(
            matched_gt_codes_list, matched_gt_acuities_list,
            all_gt_norm, gt_acuities,
        )
        all_weighted_recalls.append(w_rec)

        # Acuity accuracy: for matched pairs, collect acuity labels
        pred_code_to_acuity = dict(zip(
            [_normalize_icd10(c) for c in pred_codes], pred_acuities
        ))
        for pred_c, gt_c in matched:
            p_acuity = _normalize_acuity(pred_code_to_acuity.get(pred_c, ""))
            g_acuity = _normalize_acuity(gt_code_to_acuity.get(gt_c, ""))
            all_acuity_pairs.append((p_acuity or "", g_acuity or ""))

    # Aggregate
    mean_recall = float(np.mean(all_recalls)) if all_recalls else 0.0
    mean_precision = float(np.mean(all_precisions)) if all_precisions else 0.0
    mean_f1 = problem_list_f1(mean_recall, mean_precision)
    mean_w_recall = float(np.mean(all_weighted_recalls)) if all_weighted_recalls else 0.0
    w_f1 = problem_list_f1(mean_w_recall, mean_precision)

    acuity_acc = acuity_accuracy(all_acuity_pairs)
    acuity_cm = acuity_confusion_matrix(all_acuity_pairs)

    defined_neutral = [p for p in all_precisions_neutral if not np.isnan(p)]
    mean_precision_neutral = (float(np.mean(defined_neutral)) if defined_neutral
                              else mean_precision)
    w_f1_neutral = problem_list_f1(mean_w_recall, mean_precision_neutral)

    return {
        "problem_list_recall": mean_recall,
        "problem_list_precision": mean_precision,
        "problem_list_f1": mean_f1,
        "weighted_problem_list_recall": mean_w_recall,
        "weighted_problem_list_f1": w_f1,
        "problem_list_precision_neutral": mean_precision_neutral,
        "problem_list_f1_neutral": problem_list_f1(mean_recall, mean_precision_neutral),
        "weighted_problem_list_f1_neutral": w_f1_neutral,
        "n_neutral_predictions": n_neutral_predictions,
        "icd10_specificity_score": float(np.mean(all_specificity)) if all_specificity else 0.0,
        "acuity_accuracy": acuity_acc if acuity_acc is not None else 0.0,
        "acuity_confusion_matrix": acuity_cm,
        "n_scored": n_scored,
        "n_tier_c": n_tier_c,
    }


# ============================================================================
# SUMMARIZATION METRICS
# ============================================================================

def rouge_l(predictions: list[str], references: list[str]) -> float:
    """ROUGE-L F1 score averaged across items."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        log.warning("rouge-score not installed; skipping ROUGE-L")
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0


def clinical_f1(predictions: list[str], must_include_findings: list[list[dict]]) -> float:
    """Recall of must_include_findings in predicted summaries.

    Uses the abbreviation-aware semantic matcher (eval/semantic_match.py) so that
    physician shorthand and synonym variants are matched, consistent with the
    physician-side scoring (spec_v1.33 §17.9).
    """
    if not predictions:
        return 0.0
    return _list_recall(predictions, must_include_findings)


def omission_rate(predictions: list[str], must_include_findings: list[list[dict]]) -> float:
    """Fraction of must_include_findings NOT mentioned (1 - clinical recall)."""
    return 1.0 - clinical_f1(predictions, must_include_findings)


def hallucination_rate(predictions: list[str], ehr_texts: list[str],
                       jaccard_threshold: float = 0.15) -> float:
    """Fraction of summary sentences not grounded in source EHR text."""
    if not predictions:
        return 0.0

    rates = []
    for pred, ehr in zip(predictions, ehr_texts):
        if not pred:
            rates.append(0.0)
            continue
        sentences = _split_sentences(pred)
        if not sentences:
            rates.append(0.0)
            continue

        ehr_tokens = set(ehr.lower().split())
        hallucinated = 0
        for sent in sentences:
            sent_tokens = set(sent.lower().split())
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & ehr_tokens) / len(sent_tokens)
            if overlap < jaccard_threshold:
                hallucinated += 1
        rates.append(hallucinated / len(sentences))

    return float(np.mean(rates)) if rates else 0.0


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

def ndcg_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
              k: int = 10) -> float:
    """NDCG@k with graded relevance (0-3)."""
    scores = []
    for results, judg in zip(ranked_results, judgments):
        dcg = 0.0
        for i, item in enumerate(results[:k]):
            pid = item.get("passage_id", "")
            grade = judg.get(pid, 0)
            dcg += (2**grade - 1) / math.log2(i + 2)

        # Ideal DCG: sort all judged passages by grade
        ideal_grades = sorted(judg.values(), reverse=True)[:k]
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))

        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def map_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
             k: int = 10, threshold: int = 2) -> float:
    """Mean Average Precision@k with binary relevance (grade >= threshold)."""
    aps = []
    for results, judg in zip(ranked_results, judgments):
        relevant_count = 0
        precision_sum = 0.0
        for i, item in enumerate(results[:k]):
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        total_relevant = sum(1 for g in judg.values() if g >= threshold)
        ap = precision_sum / min(total_relevant, k) if total_relevant > 0 else 0.0
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def recall_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                k: int = 10, threshold: int = 2) -> float:
    """Recall@k with binary relevance."""
    recalls = []
    for results, judg in zip(ranked_results, judgments):
        total_relevant = sum(1 for g in judg.values() if g >= threshold)
        if total_relevant == 0:
            recalls.append(0.0)
            continue
        retrieved_relevant = 0
        for item in results[:k]:
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                retrieved_relevant += 1
        recalls.append(retrieved_relevant / total_relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def precision_at_k(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                   k: int = 5, threshold: int = 2) -> float:
    """Precision@k with binary relevance."""
    precisions = []
    for results, judg in zip(ranked_results, judgments):
        relevant = 0
        for item in results[:k]:
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                relevant += 1
        precisions.append(relevant / min(k, len(results)) if results else 0.0)
    return float(np.mean(precisions)) if precisions else 0.0


def retrieval_mrr(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                  threshold: int = 2) -> float:
    """MRR: 1/rank of first relevant passage (grade >= threshold)."""
    rr_sum = 0.0
    total = len(ranked_results)
    for results, judg in zip(ranked_results, judgments):
        for rank, item in enumerate(results, start=1):
            pid = item.get("passage_id", "")
            if judg.get(pid, 0) >= threshold:
                rr_sum += 1.0 / rank
                break
    return rr_sum / total if total > 0 else 0.0


def latency_adjusted_ndcg(ranked_results: list[list[dict]], judgments: list[dict[str, int]],
                          latencies_ms: list[int], k: int = 10) -> float:
    """NDCG@10 / log2(1 + latency_seconds). Penalizes slow retrieval."""
    base_ndcg = ndcg_at_k(ranked_results, judgments, k)
    if not latencies_ms:
        return base_ndcg
    avg_latency_s = np.mean(latencies_ms) / 1000.0
    penalty = math.log2(1 + avg_latency_s)
    return base_ndcg / penalty if penalty > 0 else base_ndcg


# ============================================================================
# IMAGING METRICS
# ============================================================================

def clinical_question_f1(predicted_questions: list[str], reference_questions: list[str]) -> float:
    """Token-level F1 between predicted and reference clinical questions."""
    f1_scores = []
    for pred, ref in zip(predicted_questions, reference_questions):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        tp = len(pred_tokens & ref_tokens)
        precision = tp / len(pred_tokens)
        recall = tp / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def differential_coverage(predicted_diffs: list[list[dict]],
                          reference_diffs: list[list[dict]]) -> float:
    """Fraction of reference differential diagnoses mentioned in prediction."""
    coverages = []
    for pred_list, ref_list in zip(predicted_diffs, reference_diffs):
        if not ref_list:
            continue
        # Build set of reference ICD-10s and names
        ref_codes = set()
        ref_names = set()
        for dx in ref_list:
            code = _normalize_icd10(dx.get("icd10", ""))
            if code:
                ref_codes.add(code)
            name = dx.get("diagnosis", "").lower()
            if name:
                ref_names.add(name)

        hits = 0
        for ref_dx in ref_list:
            ref_code = _normalize_icd10(ref_dx.get("icd10", ""))
            ref_name = ref_dx.get("diagnosis", "").lower()
            found = False
            for pred_dx in pred_list:
                pred_code = _normalize_icd10(pred_dx.get("icd10", ""))
                pred_name = pred_dx.get("diagnosis", "").lower()
                if (ref_code and pred_code and ref_code == pred_code) or \
                   (ref_name and pred_name and (ref_name in pred_name or pred_name in ref_name)):
                    found = True
                    break
            if found:
                hits += 1
        coverages.append(hits / len(ref_list))
    return float(np.mean(coverages)) if coverages else 0.0


def imaging_findings_recall(predicted_findings: list[list[str]],
                            reference_findings: list[list[str]]) -> float:
    """Recall of must_include_findings for imaging task."""
    recalls = []
    for pred_list, ref_list in zip(predicted_findings, reference_findings):
        if not ref_list:
            continue
        pred_lower = " ".join(pred_list).lower()
        hits = 0
        for ref_finding in ref_list:
            if ref_finding.lower() in pred_lower:
                hits += 1
        recalls.append(hits / len(ref_list))
    return float(np.mean(recalls)) if recalls else 0.0


# ============================================================================
# CROSS-TASK METRICS
# ============================================================================

def cost_normalized_performance(primary_metric: float, model_name: str,
                                input_tokens: list[int], output_tokens: list[int]) -> float:
    """Primary metric / mean cost per item in USD."""
    from eval.config import estimate_cost_usd

    if not input_tokens:
        return primary_metric  # No cost data

    total_cost = sum(
        estimate_cost_usd(model_name, inp, out)
        for inp, out in zip(input_tokens, output_tokens)
    )
    mean_cost = total_cost / len(input_tokens) if input_tokens else 0.0
    if mean_cost <= 0:
        return float("inf") if primary_metric > 0 else 0.0
    return primary_metric / mean_cost


def inter_model_agreement(predictions_by_model: dict[str, list[bool]]) -> dict[str, float]:
    """Pairwise Cohen's kappa between models on binary correct/incorrect.

    Returns dict with 'mean_kappa' and pairwise 'model_a_vs_model_b' keys.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        log.warning("scikit-learn not installed; skipping Cohen's kappa")
        return {"mean_kappa": 0.0}

    models = sorted(predictions_by_model.keys())
    kappas = {}
    all_k = []

    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            p1 = predictions_by_model[m1]
            p2 = predictions_by_model[m2]
            if len(p1) != len(p2):
                continue
            # Convert to int for kappa
            y1 = [int(x) for x in p1]
            y2 = [int(x) for x in p2]
            try:
                k = cohen_kappa_score(y1, y2)
            except Exception:
                k = 0.0
            key = f"{m1}_vs_{m2}"
            kappas[key] = k
            all_k.append(k)

    kappas["mean_kappa"] = float(np.mean(all_k)) if all_k else 0.0
    return kappas


# ============================================================================
# DISPATCHER
# ============================================================================

def compute_all_metrics(task: str, predictions: list[dict], ground_truths: list[dict],
                        **kwargs) -> dict[str, float]:
    """Compute all metrics for a task. Returns flat dict of metric_name → value."""
    if task == "patient_diagnosis":
        return _compute_patient_diagnosis_metrics(predictions, ground_truths)
    elif task == "context_summarization":
        return _compute_summarization_metrics(predictions, ground_truths, **kwargs)
    elif task == "evidence_retrieval":
        return _compute_retrieval_metrics(predictions, ground_truths, **kwargs)
    elif task == "imaging_indication":
        return _compute_imaging_metrics(predictions, ground_truths, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


def _compute_diagnosis_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict:
    return {
        "top_1_accuracy": top_k_accuracy(predictions, ground_truths, k=1),
        "top_3_accuracy": top_k_accuracy(predictions, ground_truths, k=3),
        "top_5_accuracy": top_k_accuracy(predictions, ground_truths, k=5),
        "hierarchical_f1": hierarchical_f1(predictions, ground_truths),
        "mrr": mean_reciprocal_rank(predictions, ground_truths),
    }


def _extract_pred_summaries(predictions: list[dict]) -> list[str]:
    """Pull the summary string from predictions (handles structured sections)."""
    summaries = []
    for p in predictions:
        s = p.get("summary", "")
        if isinstance(s, str) and s:
            summaries.append(s)
            continue
        # Structured current-visit / conditioned output: join section fields.
        parts = []
        for key in ("reason_for_visit", "interval_changes", "active_problems",
                    "context", "summary"):
            v = p.get(key)
            if isinstance(v, str) and v:
                parts.append(v)
            elif isinstance(v, list):
                parts.append(" ".join(str(x) for x in v))
        summaries.append(" ".join(parts) if parts else (json.dumps(s) if s else ""))
    return summaries


def _dominant_variant(ground_truths: list[dict]) -> str:
    """The variant shared by the GT rows in this run (default 'unconditioned')."""
    counts: dict[str, int] = defaultdict(int)
    for gt in ground_truths:
        counts[gt.get("variant") or "unconditioned"] += 1
    return max(counts, key=counts.get) if counts else "unconditioned"


def _acuity_weighted_recall(summaries: list[str],
                            must_include_lists: list[list]) -> float | None:
    """Recall weighted by per-finding acuity. None if findings carry no acuity."""
    has_acuity = any(
        isinstance(f, dict) and f.get("acuity")
        for fl in must_include_lists for f in (fl or [])
    )
    if not has_acuity:
        return None
    num = den = 0.0
    for summary, findings in zip(summaries, must_include_lists):
        for f in findings or []:
            if not isinstance(f, dict):
                continue
            name = f.get("display_name") or f.get("name") or ""
            if not name:
                continue
            w = _ACUITY_WEIGHTS.get(f.get("acuity") or "unspecified", 0.8)
            den += w
            if semantic_match.phrase_in_text(name, summary or ""):
                num += w
    return (num / den) if den else 0.0


def _compute_summarization_metrics(predictions: list[dict], ground_truths: list[dict],
                                    ehr_texts: list[str] | None = None,
                                    **kwargs) -> dict:
    pred_summaries = _extract_pred_summaries(predictions)
    variant = _dominant_variant(ground_truths)

    if variant == "current_visit":
        return _compute_current_visit_metrics(
            pred_summaries, ground_truths, ehr_texts
        )
    if variant == "specialty_conditioned":
        return _compute_specialty_conditioned_metrics(pred_summaries, ground_truths, ehr_texts)

    # --- Unconditioned whole-patient summary (default) ---
    ref_summaries = [gt.get("reference_summary", "") for gt in ground_truths]
    must_include = [gt.get("must_include_findings", []) for gt in ground_truths]

    metrics = {
        "rouge_l": rouge_l(pred_summaries, ref_summaries),
        "clinical_f1": clinical_f1(pred_summaries, must_include),
        "omission_rate": omission_rate(pred_summaries, must_include),
        # Length covariate: leakage/precision metrics are length-sensitive.
        "mean_summary_words": float(np.mean([len((s or "").split()) for s in pred_summaries])) if pred_summaries else 0.0,
    }
    aw = _acuity_weighted_recall(pred_summaries, must_include)
    if aw is not None:
        metrics["acuity_weighted_recall"] = aw
    if ehr_texts:
        metrics["hallucination_rate"] = hallucination_rate(pred_summaries, ehr_texts)
    return metrics


def _compute_current_visit_metrics(pred_summaries: list[str],
                                   ground_truths: list[dict],
                                   ehr_texts: list[str] | None = None) -> dict:
    """Current-visit summarization: recall of active findings + interval deltas.

    The EHR input was truncated to encounters <= the index encounter, so
    hallucination is scored against that same truncated context.
    """
    ref_summaries = [gt.get("reference_summary", "") for gt in ground_truths]
    must_include = [gt.get("must_include_findings", []) for gt in ground_truths]
    new_lists = [(gt.get("deltas") or {}).get("new", []) for gt in ground_truths]
    resolved_lists = [(gt.get("deltas") or {}).get("resolved", []) for gt in ground_truths]
    off_target_lists = [gt.get("off_target_findings", []) for gt in ground_truths]
    future_lists = [gt.get("future_findings", []) for gt in ground_truths]
    # "Should-not-include" set = irrelevant (background/distractor) + future-only
    # (temporal leakage). Both penalize precision.
    neg_lists = [(o or []) + (f or []) for o, f in zip(off_target_lists, future_lists)]

    recall = clinical_f1(pred_summaries, must_include)
    # Fractions of each "should-not-include" set the summary pulled in (lower is
    # better). precision = 1 - (fraction of the combined negative set included).
    off_target_rate = _list_recall(pred_summaries, off_target_lists)
    future_leakage_rate = _list_recall(pred_summaries, future_lists)
    precision = 1.0 - _list_recall(pred_summaries, neg_lists)
    selection_f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0 else 0.0
    )

    metrics = {
        "rouge_l": rouge_l(pred_summaries, ref_summaries),
        "clinical_f1": recall,
        "omission_rate": 1.0 - recall,
        "delta_coverage_new": _list_recall(pred_summaries, new_lists),
        "delta_coverage_resolved": _list_recall(pred_summaries, resolved_lists),
        "off_target_rate": off_target_rate,
        "future_leakage_rate": future_leakage_rate,
        "selection_f1": selection_f1,
        "mean_summary_words": float(np.mean([len((s or "").split()) for s in pred_summaries])) if pred_summaries else 0.0,
    }
    if ehr_texts:
        metrics["hallucination_rate"] = hallucination_rate(pred_summaries, ehr_texts)
    return metrics


# Abstention detection for the specialty-conditioned "absent" stratum.
_ABSTAIN_MAX_WORDS = 12
_ABSTAIN_PHRASES = (
    "no active", "not applicable", "no relevant", "none identified",
    "no involvement", "no significant", "no specialty", "no findings",
)


def _is_abstention(summary: str) -> bool:
    """A summary counts as abstaining if it is essentially empty or explicitly
    states no specialty involvement (correct output for an 'absent' pair)."""
    s = (summary or "").strip()
    if len(s.split()) <= _ABSTAIN_MAX_WORDS:
        return True
    low = s.lower()
    return any(p in low for p in _ABSTAIN_PHRASES)


def _tier(gt: dict, name: str) -> list:
    return (gt.get("tiers") or {}).get(name, []) or []


def _compute_specialty_conditioned_metrics(pred_summaries: list[str],
                                           ground_truths: list[dict],
                                           ehr_texts: list[str] | None = None) -> dict:
    """Specialty-conditioned summarization (spec_v1.33 §17.9).

    Involved items (involvement high/low): primary-tier recall, relevant-tier
    recall, excluded-tier leakage, and conditioned_f1 = HM(primary∪relevant
    recall, 1-leakage). Absent items (correct output = abstain): binary
    abstention accuracy + leakage on the excluded set.
    """
    involvement = [gt.get("involvement") for gt in ground_truths]
    has_content = [bool(_tier(gt, "primary") or _tier(gt, "relevant")) for gt in ground_truths]
    involved = [i for i in range(len(ground_truths))
                if involvement[i] in ("high", "low", "involved") or has_content[i]]
    involved_set = set(involved)
    absent = [i for i in range(len(ground_truths)) if i not in involved_set]

    def sub(idxs, key):
        return [_tier(ground_truths[i], key) for i in idxs]

    def preds(idxs):
        return [pred_summaries[i] for i in idxs]

    metrics = {
        "n_involved": len(involved),
        "n_absent": len(absent),
        "mean_summary_words": float(np.mean([len((s or "").split()) for s in pred_summaries])) if pred_summaries else 0.0,
    }

    if involved:
        ip = preds(involved)
        primary = sub(involved, "primary")
        relevant = sub(involved, "relevant")
        neutral = sub(involved, "neutral")
        excluded = sub(involved, "excluded_sample")
        # Two-number recall: CRITICAL = pathognomonic/highly_suggestive (must-not-miss,
        # the headline faithfulness signal); COMPLETE = all findings (secondary
        # thoroughness). Most tier findings are 'optional' (commonly_seen/risk_factor),
        # which a faithful, focused summary may legitimately omit.
        crit = lambda lists: [[f for f in (lst or [])
                               if isinstance(f, dict) and f.get("importance") == "critical"]
                              for lst in lists]
        primary_c, relevant_c = crit(primary), crit(relevant)
        combined = [(p or []) + (r or []) for p, r in zip(primary, relevant)]
        combined_c = [(p or []) + (r or []) for p, r in zip(primary_c, relevant_c)]
        # Matcher cascade (lexical + negation guard + polarity-safe value-normalizer) for
        # recall AND leakage (negation-aware: "no atrial fibrillation" is not a leak).
        # leakage = the EXCLUDED tier only (truly off-target). The NEUTRAL buffer (Class-2/3
        # associative/residual links) is neither credited nor penalized, de-circularizing the
        # boundary; leakage_strict folds neutral back in (graph-wide) to report its sensitivity.
        leakage, _ = _list_recall_cascade(ip, excluded)
        leakage_strict, _ = _list_recall_cascade(
            ip, [(e or []) + (n or []) for e, n in zip(excluded, neutral)])
        precision = 1.0 - leakage

        def _hm(recall):
            return (2 * recall * precision / (recall + precision)
                    if (recall + precision) > 0 else 0.0)

        pr_c, _ = _list_recall_cascade(ip, primary_c)
        pr_all, _ = _list_recall_cascade(ip, primary)
        rr_c, _ = _list_recall_cascade(ip, relevant_c)
        rr_all, _ = _list_recall_cascade(ip, relevant)
        cond_c, _ = _list_recall_cascade(ip, combined_c)
        cond_all, credits = _list_recall_cascade(ip, combined)
        metrics.update({
            "primary_recall_critical": pr_c,                 # headline (must-not-miss)
            "primary_recall_complete": pr_all,               # secondary (thoroughness)
            "primary_recall_critical_lexical": _list_recall(ip, primary_c),  # conservative lower bound
            "relevant_recall_critical": rr_c,                # decomposition / secondary (novelty)
            "relevant_recall_complete": rr_all,
            "leakage_rate": leakage,                         # excluded-only (de-circularized boundary)
            "leakage_rate_strict": leakage_strict,           # excluded ∪ neutral (boundary sensitivity)
            "conditioned_f1": _hm(cond_c),                   # HEADLINE: HM(primary∪relevant critical, 1-leakage)
            "conditioned_f1_complete": _hm(cond_all),        # secondary (all findings)
            "value_normalizer_credits": credits,             # findings the value-normalizer added (audit)
            # Cross-variant secondaries (match the unconditioned table): omission =
            # fraction of primary findings omitted (1 - primary-tier recall, cascade);
            # hallucination = summary sentences not grounded in the source EHR.
            "omission_rate": 1.0 - pr_all,
        })
        if ehr_texts is not None:
            metrics["hallucination_rate"] = hallucination_rate(
                ip, [ehr_texts[i] for i in involved])

    if absent:
        ap = preds(absent)
        metrics["abstention_accuracy"] = float(np.mean([_is_abstention(s) for s in ap]))
        metrics["absent_leakage_rate"] = _list_recall(ap, sub(absent, "excluded_sample"))

    return metrics


def _normalize_passage_id(pid: str, judg_keys: set[str]) -> str:
    """Normalize passage_id to match judgment keys.

    Agents may submit raw numeric IDs (e.g., '2085') while judgments use
    prefixed IDs (e.g., 'ees_2085'). Try the raw ID first, then common prefixes.
    """
    pid_str = str(pid)
    if pid_str in judg_keys:
        return pid_str
    # Try adding common prefixes
    for prefix in ("ees_", "fc_"):
        prefixed = f"{prefix}{pid_str}"
        if prefixed in judg_keys:
            return prefixed
    # Try stripping prefix if agent included one
    if "_" in pid_str:
        stripped = pid_str.split("_", 1)[1]
        if stripped in judg_keys:
            return stripped
    return pid_str


def _compute_retrieval_metrics(predictions: list[dict], ground_truths: list[dict],
                                latencies_ms: list[int] | None = None,
                                **kwargs) -> dict:
    # predictions should be list of {"rankings": [{"passage_id", "grade"}, ...]}
    ranked_results = []
    judgments_list = []
    for pred, gt in zip(predictions, ground_truths):
        raw_rankings = pred.get("rankings", [])
        judg = gt.get("_judgments", {})
        judg_keys = set(judg.keys())
        # Normalize passage IDs to match judgment keys
        normalized = []
        for item in raw_rankings:
            norm_item = dict(item)
            norm_item["passage_id"] = _normalize_passage_id(
                item.get("passage_id", ""), judg_keys
            )
            normalized.append(norm_item)
        ranked_results.append(normalized)
        judgments_list.append(judg)

    metrics = {
        "ndcg_10": ndcg_at_k(ranked_results, judgments_list, k=10),
        "map_10": map_at_k(ranked_results, judgments_list, k=10),
        "recall_5": recall_at_k(ranked_results, judgments_list, k=5),
        "recall_10": recall_at_k(ranked_results, judgments_list, k=10),
        "recall_20": recall_at_k(ranked_results, judgments_list, k=20),
        "precision_5": precision_at_k(ranked_results, judgments_list, k=5),
        "mrr": retrieval_mrr(ranked_results, judgments_list),
    }

    if latencies_ms:
        metrics["latency_adjusted_ndcg"] = latency_adjusted_ndcg(
            ranked_results, judgments_list, latencies_ms, k=10
        )

    return metrics


def _compute_imaging_metrics(predictions: list[dict], ground_truths: list[dict],
                             concept_extractor=None, **kwargs) -> dict:
    """Imaging-indication metrics. `concept_extractor` (eval.imaging_concepts.ConceptExtractor,
    built from the loaded database) adds the paper's primary metric, concept-level F1 of the
    inferred clinical question; without it only the token-level F1 is reported."""
    pred_questions = [p.get("clinical_question", "") for p in predictions]
    ref_questions = [gt.get("inferred_clinical_question", "") for gt in ground_truths]
    pred_summaries = [p.get("pre_read_summary", "") for p in predictions]
    ref_summaries = [gt.get("pre_read_summary", "") for gt in ground_truths]
    pred_diffs = [p.get("differential", []) for p in predictions]
    ref_diffs = [gt.get("differential_context", []) for gt in ground_truths]
    pred_findings = [p.get("must_include_findings", []) for p in predictions]
    ref_findings = [gt.get("must_include_findings", []) for gt in ground_truths]

    metrics = {
        "clinical_question_f1": clinical_question_f1(pred_questions, ref_questions),
        "summary_rouge_l": rouge_l(pred_summaries, ref_summaries),
        "differential_coverage": differential_coverage(pred_diffs, ref_diffs),
        "findings_recall": imaging_findings_recall(pred_findings, ref_findings),
    }
    if concept_extractor is not None:
        from eval.imaging_concepts import concept_f1_batch
        metrics.update(concept_f1_batch(concept_extractor, pred_questions, ref_questions))
    return metrics
