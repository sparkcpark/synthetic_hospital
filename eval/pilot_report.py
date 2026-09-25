"""Pilot report generator with delta-vs-baseline comparison.

Displays metrics for a given strategy alongside zero_shot baseline,
with delta columns showing improvement/regression.
"""

import json
import logging

from eval.config import EVAL_TASKS
from eval.report import PRIMARY_METRICS

log = logging.getLogger(__name__)


def generate_pilot_report(conn, task: str, split: str, strategy: str,
                          baseline_strategy: str = "zero_shot"):
    """Print a terminal report comparing strategy vs baseline for one task."""
    primary_key = PRIMARY_METRICS.get(task)
    if not primary_key:
        log.warning("No primary metric for task=%s", task)
        return

    with conn.cursor() as cur:
        # Load baseline runs
        cur.execute("""
            SELECT model_name, metrics FROM evaluation_runs
            WHERE task = %s AND split = %s AND prompt_strategy = %s
              AND completed_at IS NOT NULL
            ORDER BY model_name
        """, (task, split, baseline_strategy))
        baseline_rows = cur.fetchall()

        # Load strategy runs
        cur.execute("""
            SELECT model_name, metrics FROM evaluation_runs
            WHERE task = %s AND split = %s AND prompt_strategy = %s
              AND completed_at IS NOT NULL
            ORDER BY model_name
        """, (task, split, strategy))
        strategy_rows = cur.fetchall()

    if not strategy_rows:
        log.info("No completed %s runs for task=%s", strategy, task)
        return

    # Build baseline lookup
    baseline_map: dict[str, float] = {}
    for model, metrics_json in baseline_rows:
        m = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
        baseline_map[model] = m.get(primary_key, 0.0)

    # Print table
    print(f"\n  {task.upper()} — {strategy} vs {baseline_strategy} (metric: {primary_key})")
    print(f"  {'=' * 70}")
    header = f"  {'Model':<18} {baseline_strategy:>10} {strategy:>10} {'Delta':>10} {'%Change':>10}"
    print(header)
    print(f"  {'-' * 70}")

    for model, metrics_json in strategy_rows:
        m = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
        strat_val = m.get(primary_key, 0.0)
        base_val = baseline_map.get(model)

        if base_val is not None:
            delta = strat_val - base_val
            pct = (delta / base_val * 100) if base_val > 0 else 0.0
            sign = "+" if delta >= 0 else ""
            print(f"  {model:<18} {base_val:>10.4f} {strat_val:>10.4f} "
                  f"{sign}{delta:>9.4f} {sign}{pct:>9.1f}%")
        else:
            print(f"  {model:<18} {'N/A':>10} {strat_val:>10.4f} "
                  f"{'N/A':>10} {'N/A':>10}")
    print()


def generate_all_pilot_reports(conn, split: str = "public", strategy: str = "structured",
                               baseline_strategy: str = "zero_shot"):
    """Generate pilot reports for all 5 tasks."""
    for task in EVAL_TASKS:
        generate_pilot_report(conn, task, split, strategy, baseline_strategy)
