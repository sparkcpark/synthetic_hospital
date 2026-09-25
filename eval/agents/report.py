"""Phase F comparison report: arm comparisons, paired tests, visualization export.

Generates terminal tables comparing structured vs. bash arms across models
and tasks, including efficiency metrics, statistical tests, and CSV export.
"""

import csv
import io
import json
import logging

from eval.agents.metrics import (
    compute_arm_comparison,
    compute_efficiency_metrics,
    compute_efficiency_ratio,
    compute_info_gathering_metrics,
    compute_workflow_metrics,
)
from eval.agents.runner import AGENT_TASKS
from eval.config import get_pg_connection
from eval.report import PRIMARY_METRICS

log = logging.getLogger(__name__)

# Phase F models
AGENT_MODELS = ["gpt-5.3", "gemini-3.1", "opus-4.6", "kimi-2.5-thinking-improved"]


def generate_agent_report(
    conn=None,
    split: str = "val",
    task: str | None = None,
    model: str | None = None,
) -> str:
    """Generate the Phase F comparison report.

    Returns formatted terminal output.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_pg_connection()

    try:
        tasks = [task] if task else AGENT_TASKS
        models = [model] if model else AGENT_MODELS

        lines = []
        lines.append("=" * 80)
        lines.append("PHASE F: AGENT TOOL-USE EVALUATION — STRUCTURED vs BASH")
        lines.append("=" * 80)
        lines.append("")

        for t in tasks:
            primary_metric = PRIMARY_METRICS.get(t, "score")
            lines.append(f"### {t.upper()} (primary metric: {primary_metric})")
            lines.append("-" * 70)

            # Header
            lines.append(f"{'Model':<28} {'Structured':>10} {'Bash':>10} {'Delta':>10} {'p-value':>10} {'Cohen d':>10}")
            lines.append("-" * 70)

            for m in models:
                structured_data = _load_arm_data(conn, m, "structured", t, split)
                bash_data = _load_arm_data(conn, m, "bash", t, split)

                if not structured_data["scores"] or not bash_data["scores"]:
                    lines.append(f"{m:<28} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
                    continue

                comparison = compute_arm_comparison(
                    structured_data["scores"],
                    bash_data["scores"],
                )

                s_mean = comparison.get("structured_mean", 0)
                b_mean = comparison.get("bash_mean", 0)
                delta = comparison.get("structured_advantage", 0)
                p_val = comparison.get("wilcoxon_p_value", 1.0)
                d = comparison.get("cohens_d", 0)

                sig = "*" if p_val < 0.05 else " "
                lines.append(
                    f"{m:<28} {s_mean:>10.4f} {b_mean:>10.4f} "
                    f"{delta:>+10.4f} {p_val:>9.4f}{sig} {d:>10.3f}"
                )

            lines.append("")

            # Efficiency summary
            lines.append(f"  Efficiency Metrics:")
            lines.append(f"  {'Model':<26} {'Arm':<12} {'Actions':>8} {'Submit%':>8} {'Budget%':>8}")
            lines.append(f"  {'-' * 64}")

            for m in models:
                for arm in ("structured", "bash"):
                    data = _load_arm_data(conn, m, arm, t, split)
                    eff = compute_efficiency_metrics(data.get("results", []))
                    if not eff:
                        continue
                    lines.append(
                        f"  {m:<26} {arm:<12} "
                        f"{eff.get('mean_actions_used', 0):>8.1f} "
                        f"{eff.get('submission_rate', 0):>7.1%} "
                        f"{eff.get('budget_utilization', 0):>7.1%}"
                    )

            lines.append("")

        lines.append("* p < 0.05 (paired Wilcoxon signed-rank test)")
        return "\n".join(lines)

    finally:
        if own_conn:
            conn.close()


def _load_arm_data(conn, model: str, arm: str, task: str, split: str) -> dict:
    """Load scores and result metadata for one (model, arm, task, split)."""
    method_type = f"agent_{arm}"

    with conn.cursor() as cur:
        # Find the run
        cur.execute("""
            SELECT run_id, metrics
            FROM evaluation_runs
            WHERE model_name = %s
              AND method_type = %s
              AND task = %s
              AND split = %s
              AND completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 1
        """, (model, method_type, task, split))
        row = cur.fetchone()

        if not row:
            return {"scores": [], "results": []}

        run_id, metrics_json = row
        metrics = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json or "{}")

        # Load per-item scores
        cur.execute("""
            SELECT ep.gt_id, ep.score, ep.input_tokens, ep.output_tokens,
                   ep.latency_ms, ep.raw_output
            FROM evaluation_predictions ep
            WHERE ep.run_id = %s
        """, (run_id,))

        scores = []
        results = []
        for gt_id, score, inp_tok, out_tok, lat_ms, raw_output in cur.fetchall():
            scores.append(score or 0.0)

            # Parse config from raw_output (we stored config JSON there)
            config = {}
            if raw_output:
                try:
                    config = json.loads(raw_output)
                except (json.JSONDecodeError, TypeError):
                    pass

            results.append({
                "gt_id": gt_id,
                "score": score or 0.0,
                "turns_used": config.get("turns_used", 0),
                "submitted": config.get("submitted", False),
                "budget": config.get("budget", DEFAULT_BUDGET),
                "total_input_tokens": inp_tok or 0,
                "total_output_tokens": out_tok or 0,
                "total_latency_ms": lat_ms or 0,
            })

    return {"scores": scores, "results": results, "metrics": metrics}


DEFAULT_BUDGET = 40


def export_agent_csv(
    output_path: str,
    conn=None,
    split: str = "val",
) -> str:
    """Export Phase F results to CSV."""
    own_conn = conn is None
    if own_conn:
        conn = get_pg_connection()

    try:
        rows = []
        for task in AGENT_TASKS:
            primary_metric = PRIMARY_METRICS.get(task, "score")
            for model in AGENT_MODELS:
                for arm in ("structured", "bash"):
                    data = _load_arm_data(conn, model, arm, task, split)
                    eff = compute_efficiency_metrics(data.get("results", []))
                    metrics = data.get("metrics", {})
                    rows.append({
                        "task": task,
                        "model": model,
                        "arm": arm,
                        "primary_metric": primary_metric,
                        "primary_score": metrics.get(primary_metric, ""),
                        "n_predictions": metrics.get("n_predictions", 0),
                        "mean_actions": eff.get("mean_actions_used", ""),
                        "submission_rate": eff.get("submission_rate", ""),
                        "budget_utilization": eff.get("budget_utilization", ""),
                        "mean_cost_tokens": eff.get("mean_cost_tokens", ""),
                        "mean_time_ms": eff.get("mean_time_ms", ""),
                    })

        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            log.info("Exported Phase F results to %s (%d rows)", output_path, len(rows))

        return output_path

    finally:
        if own_conn:
            conn.close()
