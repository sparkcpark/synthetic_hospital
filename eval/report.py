"""Results aggregation, comparison tables, CSV/JSON export, terminal leaderboards."""

import csv
import json
import logging
import sys
from io import StringIO

from eval.config import COST_RATES, EVAL_TASKS, get_pg_connection
from eval.scoring import cost_normalized_performance, inter_model_agreement

log = logging.getLogger(__name__)

# Primary metric per task (for leaderboard ranking)
PRIMARY_METRICS = {
    "patient_diagnosis": "weighted_problem_list_f1_neutral",   # chart-neutral rule (paper Table 3)
    "context_summarization": "clinical_f1",                   # must-include finding recall
    "evidence_retrieval": "precision_5",                      # chart sections only
    "imaging_indication": "clinical_question_concept_f1",     # ontology-grounded concept F1
}


# ============================================================================
# CLI entry point
# ============================================================================

def generate_report(task: str | None, split: str, fmt: str, output: str | None,
                    strategy: str | None = None):
    """Generate comparison report. Called by CLI."""
    conn = get_pg_connection()
    try:
        if task:
            tasks = [task]
        else:
            tasks = EVAL_TASKS

        for t in tasks:
            df_rows, col_names = generate_comparison_table(conn, t, split, strategy=strategy)
            if not df_rows:
                log.info("No completed runs for task=%s split=%s", t, split)
                continue

            if fmt == "terminal":
                _print_table(t, df_rows, col_names)
            elif fmt == "csv":
                path = output or f"results/{t}_{split}_comparison.csv"
                _export_csv(df_rows, col_names, path)
                log.info("Exported CSV: %s", path)
            elif fmt == "json":
                path = output or f"results/{t}_{split}_comparison.json"
                _export_json(df_rows, col_names, path)
                log.info("Exported JSON: %s", path)
    finally:
        conn.close()


# ============================================================================
# Comparison table
# ============================================================================

def generate_comparison_table(conn, task: str, split: str,
                              strategy: str | None = None) -> tuple[list[dict], list[str]]:
    """Build rows=models, cols=metrics comparison table.

    Returns (rows, column_names) where each row is a dict.
    """
    with conn.cursor() as cur:
        sql = """
            SELECT run_id, model_name, prompt_strategy, metrics,
                   started_at, completed_at
            FROM evaluation_runs
            WHERE task = %s AND split = %s AND completed_at IS NOT NULL
        """
        params: list = [task, split]
        if strategy:
            sql += " AND prompt_strategy = %s"
            params.append(strategy)
        sql += " ORDER BY model_name, prompt_strategy"
        cur.execute(sql, params)
        runs = cur.fetchall()

    if not runs:
        return [], []

    # Determine metric columns from first run
    sample_metrics = runs[0][3]
    if isinstance(sample_metrics, str):
        sample_metrics = json.loads(sample_metrics)
    metric_keys = [k for k in sample_metrics.keys()
                   if not k.startswith("n_") and not k.startswith("total_")]

    col_names = ["model", "strategy", "run_id"] + metric_keys + [
        "n_predictions", "mean_latency_ms", "total_input_tokens", "total_output_tokens"
    ]

    rows = []
    for run_id, model, strategy, metrics_json, started, completed in runs:
        m = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
        row = {
            "model": model or "-",
            "strategy": strategy or "-",
            "run_id": run_id,
        }
        for k in metric_keys:
            row[k] = m.get(k, 0.0)
        row["n_predictions"] = m.get("n_predictions", 0)
        row["mean_latency_ms"] = m.get("mean_latency_ms", 0.0)
        row["total_input_tokens"] = m.get("total_input_tokens", 0)
        row["total_output_tokens"] = m.get("total_output_tokens", 0)
        rows.append(row)

    # Sort by primary metric descending
    primary = PRIMARY_METRICS.get(task, metric_keys[0] if metric_keys else None)
    if primary and primary in metric_keys:
        rows.sort(key=lambda r: r.get(primary, 0.0), reverse=True)

    return rows, col_names


# ============================================================================
# Cost efficiency table
# ============================================================================

def generate_cost_efficiency_table(conn, task: str, split: str) -> tuple[list[dict], list[str]]:
    """Build cost-normalized performance comparison."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT er.run_id, er.model_name, er.metrics,
                   COALESCE(SUM(ep.input_tokens), 0),
                   COALESCE(SUM(ep.output_tokens), 0),
                   COUNT(ep.prediction_id)
            FROM evaluation_runs er
            LEFT JOIN evaluation_predictions ep ON er.run_id = ep.run_id
            WHERE er.task = %s AND er.split = %s AND er.completed_at IS NOT NULL
            GROUP BY er.run_id, er.model_name, er.metrics
            ORDER BY er.model_name
        """, (task, split))
        runs = cur.fetchall()

    if not runs:
        return [], []

    primary_key = PRIMARY_METRICS.get(task, "top_1_accuracy")
    col_names = ["model", "primary_metric", "primary_value",
                 "total_cost_usd", "cost_per_item_usd", "cost_normalized_perf"]

    rows = []
    for run_id, model, metrics_json, total_inp, total_out, n_pred in runs:
        m = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
        primary_val = m.get(primary_key, 0.0)

        from eval.config import estimate_cost_usd
        total_cost = estimate_cost_usd(model or "", total_inp, total_out)
        cost_per_item = total_cost / n_pred if n_pred > 0 else 0.0
        cnp = primary_val / cost_per_item if cost_per_item > 0 else float("inf")

        rows.append({
            "model": model or "-",
            "primary_metric": primary_key,
            "primary_value": round(primary_val, 4),
            "total_cost_usd": round(total_cost, 4),
            "cost_per_item_usd": round(cost_per_item, 6),
            "cost_normalized_perf": round(cnp, 2),
        })

    rows.sort(key=lambda r: r.get("cost_normalized_perf", 0.0), reverse=True)
    return rows, col_names


# ============================================================================
# Kappa matrix
# ============================================================================

def generate_kappa_matrix(conn, task: str, split: str) -> tuple[list[dict], list[str]]:
    """Build inter-model agreement (Cohen's kappa) matrix."""
    with conn.cursor() as cur:
        # Get all completed runs for this task/split
        cur.execute("""
            SELECT run_id, model_name
            FROM evaluation_runs
            WHERE task = %s AND split = %s AND completed_at IS NOT NULL
            ORDER BY model_name
        """, (task, split))
        runs = cur.fetchall()

    if len(runs) < 2:
        return [], []

    # Load per-item binary correctness for each model
    primary_key = PRIMARY_METRICS.get(task, "top_1_accuracy")
    predictions_by_model = {}

    for run_id, model in runs:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ep.gt_id, ep.prediction, bgt.ground_truth
                FROM evaluation_predictions ep
                JOIN benchmark_ground_truth bgt ON ep.gt_id = bgt.gt_id
                WHERE ep.run_id = %s
                ORDER BY ep.gt_id
            """, (run_id,))
            rows = cur.fetchall()

        correctness = []
        for gt_id, pred_json, gt_json in rows:
            pred = pred_json if isinstance(pred_json, dict) else json.loads(pred_json)
            gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
            correct = _is_correct(task, pred, gt)
            correctness.append(correct)

        predictions_by_model[model or f"run_{run_id}"] = correctness

    kappas = inter_model_agreement(predictions_by_model)

    # Format as table
    models = sorted(predictions_by_model.keys())
    col_names = ["model"] + models + ["mean_kappa"]
    rows_out = []
    for m1 in models:
        row = {"model": m1}
        for m2 in models:
            if m1 == m2:
                row[m2] = 1.0
            else:
                k1 = f"{m1}_vs_{m2}"
                k2 = f"{m2}_vs_{m1}"
                row[m2] = round(kappas.get(k1, kappas.get(k2, 0.0)), 4)
        row["mean_kappa"] = round(kappas.get("mean_kappa", 0.0), 4)
        rows_out.append(row)

    return rows_out, col_names


def _is_correct(task: str, pred: dict, gt: dict) -> bool:
    """Binary correctness check for Cohen's kappa."""
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
        # Binary: problem list F1 >= 0.5
        from eval.scoring import _normalize_icd10, _match_icd10_sets, problem_list_f1
        gt_active = gt.get("active_diagnoses", [])
        gt_chronic = gt.get("chronic_conditions", [])
        gt_codes = [dx.get("icd10", "") for dx in gt_active + gt_chronic]
        pred_active = pred.get("active_diagnoses", [])
        pred_chronic = pred.get("chronic_conditions", [])
        pred_codes = [dx.get("icd10", "") for dx in pred_active + pred_chronic]
        if not gt_codes:
            return True
        from eval.scoring import problem_list_recall, problem_list_precision
        rec = problem_list_recall(pred_codes, gt_codes)
        prec = problem_list_precision(pred_codes, gt_codes)
        return problem_list_f1(rec, prec) >= 0.5

    elif task == "context_summarization":
        # Discretize ROUGE-L: >= 0.4 = "correct"
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            pred_summary = pred.get("summary", "")
            ref_summary = gt.get("reference_summary", "")
            if pred_summary and ref_summary:
                result = scorer.score(ref_summary, pred_summary)
                return result["rougeL"].fmeasure >= 0.4
        except ImportError:
            pass
        return False

    elif task == "evidence_retrieval":
        # Top-1 passage is relevant (grade >= 2)
        rankings = pred.get("rankings", [])
        if rankings:
            top_pid = rankings[0].get("passage_id", "")
            judg = gt.get("_judgments", {})
            return judg.get(top_pid, 0) >= 2
        return False

    elif task == "imaging_indication":
        # Clinical question F1 >= 0.3
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


# ============================================================================
# Per-item export
# ============================================================================

def export_per_item_scores(run_id: int, path: str):
    """Export per-GT-item predictions and scores for error analysis."""
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ep.gt_id, ep.prediction, ep.latency_ms,
                       ep.input_tokens, ep.output_tokens,
                       bgt.ground_truth, bgt.task
                FROM evaluation_predictions ep
                JOIN benchmark_ground_truth bgt ON ep.gt_id = bgt.gt_id
                WHERE ep.run_id = %s
                ORDER BY ep.gt_id
            """, (run_id,))
            rows = cur.fetchall()

        if not rows:
            log.warning("No predictions for run_id=%d", run_id)
            return

        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["gt_id", "task", "latency_ms", "input_tokens",
                             "output_tokens", "prediction_json", "ground_truth_json"])
            for gt_id, pred, lat, inp, out, gt, task in rows:
                writer.writerow([
                    gt_id, task, lat, inp, out,
                    json.dumps(pred) if isinstance(pred, dict) else pred,
                    json.dumps(gt) if isinstance(gt, dict) else gt,
                ])
        log.info("Exported %d per-item scores to %s", len(rows), path)
    finally:
        conn.close()


# ============================================================================
# Output formatters
# ============================================================================

def _print_table(task: str, rows: list[dict], col_names: list[str]):
    """Print a comparison table to terminal."""
    primary = PRIMARY_METRICS.get(task)
    print(f"\n{'='*80}")
    print(f"  {task.upper()} — Leaderboard (sorted by {primary})")
    print(f"{'='*80}")

    # Compute column widths
    widths = {}
    for col in col_names:
        widths[col] = max(len(col), max((len(_fmt_val(r.get(col, ""))) for r in rows), default=0))

    # Header
    header = " | ".join(f"{col:>{widths[col]}}" for col in col_names)
    print(header)
    print("-" * len(header))

    # Rows
    for row in rows:
        line = " | ".join(f"{_fmt_val(row.get(col, '')):>{widths[col]}}" for col in col_names)
        print(line)
    print()


def _fmt_val(val) -> str:
    """Format a value for terminal display."""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _export_csv(rows: list[dict], col_names: list[str], path: str):
    """Export comparison table to CSV."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _export_json(rows: list[dict], col_names: list[str], path: str):
    """Export comparison table to JSON."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"columns": col_names, "rows": rows}, f, indent=2, default=str)
