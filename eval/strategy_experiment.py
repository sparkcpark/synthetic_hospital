"""Strategy experiment orchestrator.

Runs structured → cot → few_shot strategies across 3 representative models.
Each round runs a 75-item pilot first, then optionally scales to full val.

Usage:
    python -m eval.strategy_experiment --round 1              # structured pilot
    python -m eval.strategy_experiment --round 1 --full       # structured full val
    python -m eval.strategy_experiment --round 2              # cot pilot
    python -m eval.strategy_experiment --round 3              # few_shot pilot
    python -m eval.strategy_experiment --compare              # compare all strategies
    python -m eval.strategy_experiment --round 1 --dry-run    # verify preconditions
    python -m eval.strategy_experiment --scale-val            # locked strategies, 3 experiment models, full val
    python -m eval.strategy_experiment --scale-all            # locked strategies, all 11 models, full val
    python -m eval.strategy_experiment --scale-val --dry-run  # verify preconditions only
"""

import argparse
import json
import logging
import sys

from eval.config import EVAL_TASKS, MODEL_REGISTRY, get_pg_connection
from eval.report import PRIMARY_METRICS

log = logging.getLogger(__name__)

EXPERIMENT_MODELS = ["gpt-5.3", "deepseek-v3.2", "mistral-3"]
ALL_MODELS = [
    "gpt-5.3", "opus-4.6", "gemini-3.1", "llama-4", "qwen-3",
    "deepseek-v3.2", "mistral-3", "gemma-3", "glm-5",
    "kimi-2.5-improved", "kimi-2.5-thinking-improved",
]
ROUND_STRATEGIES = {1: "structured", 2: "cot", 3: "few_shot"}
DEFAULT_PILOT = 75

# Winning strategy per task from E1.2 pilot (3 rounds × 3 models × 75 items)
LOCKED_STRATEGIES = {
    "patient_diagnosis": "cot",
    "context_summarization": "few_shot",
    "evidence_retrieval": "zero_shot",
    "imaging_indication": "few_shot",
}


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------

def run_round(round_num: int, pilot: int = DEFAULT_PILOT, full: bool = False,
              workers: int = 5, dry_run: bool = False):
    """Execute one experiment round."""
    strategy = ROUND_STRATEGIES[round_num]
    log.info("=== Round %d: strategy=%s ===", round_num, strategy)

    # Pre-flight checks
    if round_num == 2:
        _check_deepseek_cot_safety()
    if round_num == 3:
        _check_few_shot_examples()

    if dry_run:
        log.info("[DRY RUN] Would run: %d models x %d tasks x strategy=%s (pilot=%s)",
                 len(EXPERIMENT_MODELS), len(EVAL_TASKS), strategy,
                 "full" if full else str(pilot))
        _dry_run_checks(strategy)
        return

    from eval.runner import run_evaluation

    for task in EVAL_TASKS:
        for model in EXPERIMENT_MODELS:
            log.info("--- %s / %s / %s ---", task, model, strategy)
            try:
                run_id = run_evaluation(
                    task=task,
                    model_name=model,
                    split="public",
                    prompt_strategy=strategy,
                    pilot=None if full else pilot,
                    workers=workers,
                )
                log.info("Completed run_id=%d", run_id)
            except Exception as e:
                log.error("Failed %s/%s/%s: %s", task, model, strategy, e)

    # Print comparison after round
    _print_round_summary(strategy)


# ---------------------------------------------------------------------------
# Scale to full val with locked strategies
# ---------------------------------------------------------------------------

def run_scale(models: list[str], workers: int = 5, dry_run: bool = False):
    """Run locked per-task strategies across models on full val split."""
    log.info("=== Scale-val: %d models × %d tasks (locked strategies) ===", len(models), len(EVAL_TASKS))

    # Pre-flight: verify few-shot examples for tasks that need them
    few_shot_tasks = [t for t, s in LOCKED_STRATEGIES.items() if s == "few_shot"]
    if few_shot_tasks:
        _check_few_shot_examples()

    if dry_run:
        log.info("[DRY RUN] Would run:")
        for task in EVAL_TASKS:
            strategy = LOCKED_STRATEGIES[task]
            log.info("  %s → %s (%d models)", task, strategy, len(models))
        _dry_run_checks_models(models)
        return

    from eval.runner import run_evaluation

    for task in EVAL_TASKS:
        strategy = LOCKED_STRATEGIES[task]
        for model in models:
            log.info("--- %s / %s / %s ---", task, model, strategy)
            try:
                run_id = run_evaluation(
                    task=task,
                    model_name=model,
                    split="public",
                    prompt_strategy=strategy,
                    pilot=None,
                    workers=workers,
                )
                log.info("Completed run_id=%d", run_id)
            except Exception as e:
                log.error("Failed %s/%s/%s: %s", task, model, strategy, e)

    # Print summary for each locked strategy
    for strategy in sorted(set(LOCKED_STRATEGIES.values())):
        if strategy != "zero_shot":
            _print_round_summary(strategy)


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def _check_deepseek_cot_safety():
    """Round 2 safety: log DeepSeek V3.2 zero_shot output token baseline."""
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT metrics FROM evaluation_runs
                WHERE model_name = 'deepseek-v3.2' AND prompt_strategy = 'zero_shot'
                  AND split = 'public' AND completed_at IS NOT NULL
                ORDER BY completed_at DESC LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                log.warning("No DeepSeek V3.2 zero_shot baseline found for CoT safety check")
                return
            m = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            baseline_tokens = m.get("total_output_tokens", 0)
            n_preds = m.get("n_predictions", 1)
            avg_tokens = baseline_tokens / n_preds if n_preds > 0 else 0
            log.info("DeepSeek V3.2 zero_shot avg output tokens: %.0f", avg_tokens)
            log.info("CoT budget ceiling (2x): %.0f tokens/prediction", avg_tokens * 2)
    finally:
        conn.close()


def _check_few_shot_examples():
    """Round 3 dependency: verify get_few_shot_examples() returns non-empty for all tasks."""
    from eval.examples import get_few_shot_examples

    missing = []
    for task in EVAL_TASKS:
        block = get_few_shot_examples(task)
        if not block:
            missing.append(task)
        else:
            log.info("Few-shot examples OK for task=%s (%d chars)", task, len(block))

    if missing:
        log.error("No few-shot examples for tasks: %s. Run zero_shot baseline first.", missing)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dry-run verification
# ---------------------------------------------------------------------------

def _dry_run_checks(strategy: str):
    """Verify preconditions without running evaluations."""
    # 1. Verify few-shot examples if needed
    if strategy == "few_shot":
        _check_few_shot_examples()

    # 2. Verify DB connectivity
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM evaluation_runs")
            count = cur.fetchone()[0]
            log.info("Database accessible: %d existing runs", count)
    finally:
        conn.close()

    # 3. Verify all experiment models exist in registry
    for model in EXPERIMENT_MODELS:
        if model not in MODEL_REGISTRY:
            log.error("Model %s not in MODEL_REGISTRY", model)
            sys.exit(1)
        log.info("Model %s: OK (adapter=%s)", model, MODEL_REGISTRY[model].adapter_type.value)

    log.info("[DRY RUN] All checks passed.")


def _dry_run_checks_models(models: list[str]):
    """Verify preconditions for a given model list."""
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM evaluation_runs")
            count = cur.fetchone()[0]
            log.info("Database accessible: %d existing runs", count)
    finally:
        conn.close()

    for model in models:
        if model not in MODEL_REGISTRY:
            log.error("Model %s not in MODEL_REGISTRY", model)
            sys.exit(1)
        log.info("Model %s: OK (adapter=%s)", model, MODEL_REGISTRY[model].adapter_type.value)

    log.info("[DRY RUN] All checks passed.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_round_summary(strategy: str):
    """Print comparison table with deltas vs zero_shot baseline."""
    from eval.pilot_report import generate_all_pilot_reports

    conn = get_pg_connection()
    try:
        generate_all_pilot_reports(conn, "val", strategy, "zero_shot")
    finally:
        conn.close()


def run_compare():
    """Cross-strategy comparison matrix: all strategies for experiment models."""
    conn = get_pg_connection()
    try:
        for task in EVAL_TASKS:
            primary_key = PRIMARY_METRICS.get(task)
            if not primary_key:
                continue

            with conn.cursor() as cur:
                cur.execute("""
                    SELECT model_name, prompt_strategy, metrics
                    FROM evaluation_runs
                    WHERE task = %s AND split = 'public' AND completed_at IS NOT NULL
                      AND model_name = ANY(%s)
                    ORDER BY model_name, prompt_strategy
                """, (task, EXPERIMENT_MODELS))
                rows = cur.fetchall()

            if not rows:
                continue

            # Build matrix: model x strategy -> primary metric
            matrix: dict[str, dict[str, float]] = {}
            for model, strategy, metrics_json in rows:
                m = metrics_json if isinstance(metrics_json, dict) else json.loads(metrics_json)
                matrix.setdefault(model, {})[strategy] = m.get(primary_key, 0.0)

            strategies = sorted(set(s for m_dict in matrix.values() for s in m_dict))

            print(f"\n{'=' * 80}")
            print(f"  {task.upper()} — Cross-Strategy Comparison (metric: {primary_key})")
            print(f"{'=' * 80}")
            header = f"  {'Model':<18}" + "".join(f"{s:>14}" for s in strategies)
            print(header)
            print(f"  {'-' * (18 + 14 * len(strategies))}")

            for model in EXPERIMENT_MODELS:
                if model not in matrix:
                    continue
                line = f"  {model:<18}"
                baseline = matrix[model].get("zero_shot", 0.0)
                for s in strategies:
                    val = matrix[model].get(s)
                    if val is None:
                        line += f"{'---':>14}"
                    elif s == "zero_shot":
                        line += f"{val:>14.4f}"
                    else:
                        delta = val - baseline
                        sign = "+" if delta >= 0 else ""
                        cell = f"{val:.4f}({sign}{delta:.3f})"
                        line += f"{cell:>14}"
                print(line)
            print()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="eval.strategy_experiment",
        description="Strategy experiment orchestrator",
    )
    parser.add_argument("--round", type=int, choices=[1, 2, 3],
                        help="Experiment round: 1=structured, 2=cot, 3=few_shot")
    parser.add_argument("--full", action="store_true",
                        help="Run full val set (not just pilot)")
    parser.add_argument("--compare", action="store_true",
                        help="Print cross-strategy comparison")
    parser.add_argument("--scale-val", action="store_true",
                        help="Run locked strategies on full val with 3 experiment models")
    parser.add_argument("--scale-all", action="store_true",
                        help="Run locked strategies on full val with all 11 models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify preconditions only, do not run models")
    parser.add_argument("--pilot", type=int, default=DEFAULT_PILOT,
                        help=f"Pilot size (default: {DEFAULT_PILOT})")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    if args.compare:
        run_compare()
    elif args.scale_all:
        run_scale(ALL_MODELS, workers=args.workers, dry_run=args.dry_run)
    elif args.scale_val:
        run_scale(EXPERIMENT_MODELS, workers=args.workers, dry_run=args.dry_run)
    elif args.round:
        run_round(args.round, pilot=args.pilot, full=args.full,
                  workers=args.workers, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
