"""CLI entry point for the evaluation pipeline.

Usage:
    python -m eval.cli split [--seed 42] [--dev-patients 200]
    python -m eval.cli verify-split
    python -m eval.cli run --task patient_diagnosis --model kimi-2.5 [--split val] [--strategy zero_shot] [--pilot 10] [--workers 5]
    python -m eval.cli run --task patient_diagnosis --model kimi-2.5 [--split val] [--strategy zero_shot] [--pilot 5]
    python -m eval.cli run --task evidence_retrieval --model bm25 [--split val]
    python -m eval.cli score --run-id 42
    python -m eval.cli report [--split val] [--task patient_diagnosis] [--format csv]
    python -m eval.cli matrix --task patient_diagnosis --split val [--strategy zero_shot]
    python -m eval.cli list-runs [--task patient_diagnosis]
    python -m eval.cli validate-gt [--task patient_diagnosis]
"""

import argparse
import json
import logging
import sys

from eval.config import EVAL_TASKS, MODEL_REGISTRY, PROMPT_STRATEGIES, RETRIEVAL_BASELINES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval.cli")

ALL_MODELS = sorted(MODEL_REGISTRY.keys()) + sorted(RETRIEVAL_BASELINES)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def cmd_split(args):
    """Assign dev/test splits to benchmark_ground_truth rows."""
    from eval.split import run_split
    run_split(seed=args.seed, dev_patients=args.dev_patients)


def cmd_verify_split(args):
    """Verify split integrity: no NULLs, no patient overlap."""
    from eval.split import verify_split
    verify_split()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Execute one model on one task."""
    from eval.runner import run_evaluation, run_retrieval_baseline

    # Redirect legacy --task patient_diagnosis --granularity patient
    if args.task == "diagnosis_accuracy" and args.granularity == "patient":
        log.warning("Patient-level diagnosis has been moved to --task patient_diagnosis. Redirecting.")
        args.task = "patient_diagnosis"
        args.granularity = None

    if args.model in RETRIEVAL_BASELINES:
        run_id = run_retrieval_baseline(
            method=args.model,
            split=args.split,
            pilot=args.pilot,
        )
    else:
        run_id = run_evaluation(
            task=args.task,
            model_name=args.model,
            split=args.split,
            prompt_strategy=args.strategy,
            granularity=args.granularity,
            pilot=args.pilot,
            workers=args.workers,
            resume_run_id=args.resume,
            tier=args.tier,
        )
    log.info("Completed run_id=%d", run_id)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def cmd_score(args):
    """Recompute metrics for an existing run."""
    from eval.runner import rescore_run
    rescore_run(args.run_id)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args):
    """Generate comparison tables."""
    from eval.report import generate_report
    generate_report(
        task=args.task,
        split=args.split,
        fmt=args.format,
        output=args.output,
        strategy=args.strategy,
    )


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------

def cmd_matrix(args):
    """Run all 10 models on one task (or a single model with --model)."""
    from eval.runner import run_evaluation

    # Redirect legacy --task patient_diagnosis --granularity patient
    if args.task == "diagnosis_accuracy" and args.granularity == "patient":
        log.warning("Patient-level diagnosis has been moved to --task patient_diagnosis. Redirecting.")
        args.task = "patient_diagnosis"
        args.granularity = None

    if args.model:
        models = [args.model]
    else:
        models = sorted(MODEL_REGISTRY.keys())
    for model_name in models:
        log.info("=== Running %s on %s ===", model_name, args.task)
        try:
            run_evaluation(
                task=args.task,
                model_name=model_name,
                split=args.split,
                prompt_strategy=args.strategy,
                granularity=args.granularity,
                pilot=args.pilot,
                workers=args.workers,
                tier=args.tier,
            )
        except Exception as e:
            log.error("Failed %s: %s", model_name, e)


# ---------------------------------------------------------------------------
# list-runs
# ---------------------------------------------------------------------------

def cmd_validate_gt(args):
    """Validate ground truth rows for a task."""
    from eval.config import get_pg_connection
    from eval.validate_gt import validate_ground_truth

    conn = get_pg_connection()
    try:
        issues = validate_ground_truth(conn, task=args.task)
        if issues:
            log.error("%d GT validation issues found:", len(issues))
            for issue in issues:
                log.error("  %s", issue)
            sys.exit(1)
        else:
            task_label = args.task or "all tasks"
            log.info("GT validation passed for %s", task_label)
    finally:
        conn.close()


def cmd_pilot_report(args):
    """Show pilot report with delta-vs-baseline."""
    from eval.config import get_pg_connection
    from eval.pilot_report import generate_all_pilot_reports, generate_pilot_report

    conn = get_pg_connection()
    try:
        if args.task:
            generate_pilot_report(conn, args.task, args.split, args.strategy,
                                  args.baseline)
        else:
            generate_all_pilot_reports(conn, args.split, args.strategy, args.baseline)
    finally:
        conn.close()


def cmd_list_runs(args):
    """Show existing evaluation runs."""
    from eval.config import get_pg_connection

    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT run_id, run_name, model_name, task, split, prompt_strategy,
                       metrics, started_at, completed_at
                FROM evaluation_runs
            """
            params = []
            if args.task:
                sql += " WHERE task = %s"
                params.append(args.task)
            sql += " ORDER BY run_id"
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            print("No evaluation runs found.")
            return

        # Simple tabular output
        header = f"{'ID':>4} {'Model':<16} {'Task':<24} {'Split':<6} {'Strategy':<12} {'Status':<10} {'Primary Metric'}"
        print(header)
        print("-" * len(header))
        for row in rows:
            run_id, name, model, task, split, strategy, metrics, started, completed = row
            status = "done" if completed else "running"
            primary = ""
            if metrics:
                m = metrics if isinstance(metrics, dict) else json.loads(metrics)
                # Show first metric
                if m:
                    k, v = next(iter(m.items()))
                    primary = f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            print(f"{run_id:>4} {(model or '-'):<16} {task:<24} {split:<6} {(strategy or '-'):<12} {status:<10} {primary}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# classify-errors
# ---------------------------------------------------------------------------

def cmd_classify_errors(args):
    """Classify prediction errors for one or all completed runs."""
    import json as _json
    from eval.config import get_pg_connection
    from eval.errors import classify_errors
    from eval.tasks import TASK_LOADERS

    conn = get_pg_connection()
    try:
        if args.all:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT run_id, task, split FROM evaluation_runs "
                    "WHERE completed_at IS NOT NULL ORDER BY run_id"
                )
                runs = cur.fetchall()
            if not runs:
                log.info("No completed runs found.")
                return
            log.info("Classifying errors for %d completed runs", len(runs))
            for run_id, task, split in runs:
                _classify_one_run(conn, run_id, task, split, TASK_LOADERS, classify_errors)
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT task, split FROM evaluation_runs WHERE run_id = %s",
                    (args.run_id,),
                )
                row = cur.fetchone()
            if not row:
                log.error("Run %d not found", args.run_id)
                return
            task, split = row
            _classify_one_run(conn, args.run_id, task, split, TASK_LOADERS, classify_errors)
    finally:
        conn.close()


def _classify_one_run(conn, run_id, task, split, task_loaders, classify_errors):
    """Classify errors for a single run and UPDATE the DB."""
    import json as _json

    log.info("Classifying run_id=%d task=%s split=%s", run_id, task, split)

    # Load task inputs for ehr_text / judgments
    loader = task_loaders.get(task)
    input_map = {}
    if loader:
        try:
            inputs = loader(conn, split=split)
            input_map = {inp.gt_id: inp for inp in inputs}
        except Exception as e:
            log.warning("Could not load task inputs for run %d: %s", run_id, e)

    # Load predictions + GT
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ep.prediction_id, ep.gt_id, ep.prediction, ep.score,
                   bgt.ground_truth
            FROM evaluation_predictions ep
            JOIN benchmark_ground_truth bgt ON ep.gt_id = bgt.gt_id
            WHERE ep.run_id = %s
            ORDER BY ep.gt_id
        """, (run_id,))
        rows = cur.fetchall()

    if not rows:
        log.warning("No predictions for run_id=%d", run_id)
        return

    updates = []
    n_with_errors = 0

    for prediction_id, gt_id, pred_json, score, gt_json in rows:
        pred = pred_json if isinstance(pred_json, dict) else _json.loads(pred_json)
        gt = gt_json if isinstance(gt_json, dict) else _json.loads(gt_json)

        # Gather task-specific kwargs
        kwargs = {}
        inp_obj = input_map.get(gt_id)
        ehr_text = ""
        if inp_obj and hasattr(inp_obj, "ehr_text"):
            ehr_text = inp_obj.ehr_text
        if task == "evidence_retrieval" and inp_obj and hasattr(inp_obj, "judgments"):
            kwargs["judgments"] = inp_obj.judgments

        error_cats = classify_errors(task, gt, pred, ehr_text=ehr_text,
                                     score=score, **kwargs)
        updates.append((prediction_id, error_cats))
        if error_cats:
            n_with_errors += 1

    # Batch UPDATE
    with conn.cursor() as cur:
        for prediction_id, error_cats in updates:
            cur.execute(
                "UPDATE evaluation_predictions SET error_categories = %s "
                "WHERE prediction_id = %s",
                (_json.dumps(error_cats), prediction_id),
            )
    conn.commit()

    n_correct = len(rows) - n_with_errors
    log.info("  run %d: %d total, %d with errors, %d correct",
             run_id, len(rows), n_with_errors, n_correct)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="eval.cli",
        description="Evaluation pipeline for benchmarking LLMs on clinical NLP tasks",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # split
    p = sub.add_parser("split", help="Assign dev/test splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-patients", type=int, default=200)
    p.set_defaults(func=cmd_split)

    # verify-split
    p = sub.add_parser("verify-split", help="Verify split integrity")
    p.set_defaults(func=cmd_verify_split)

    # run
    p = sub.add_parser("run", help="Execute one model on one task")
    p.add_argument("--task", required=True, choices=EVAL_TASKS)
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    p.add_argument("--strategy", default="zero_shot", choices=PROMPT_STRATEGIES)
    p.add_argument("--granularity", choices=["question", "patient", "encounter", "specialty"],
                   help="encounter=current-visit, specialty=specialty-conditioned summarization variant")
    p.add_argument("--pilot", type=int, default=None)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--resume", type=int, default=None,
                   help="resume an existing run_id, skipping rows already completed (interruption-safe)")
    p.add_argument("--tier", choices=["small", "medium", "large"], default=None,
                   help="specialty-conditioned only: select rows by the frozen patient-aligned "
                        "cohort in specialty_dataset_manifest.json (overrides --split)")
    p.set_defaults(func=cmd_run)

    # score
    p = sub.add_parser("score", help="Recompute metrics for an existing run")
    p.add_argument("--run-id", type=int, required=True)
    p.set_defaults(func=cmd_score)

    # report
    p = sub.add_parser("report", help="Generate comparison tables")
    p.add_argument("--task", choices=EVAL_TASKS)
    p.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    p.add_argument("--strategy", choices=PROMPT_STRATEGIES, default=None,
                   help="Filter by prompt strategy (default: show all)")
    p.add_argument("--format", default="terminal", choices=["terminal", "csv", "json"])
    p.add_argument("--output", type=str, default=None)
    p.set_defaults(func=cmd_report)

    # matrix
    p = sub.add_parser("matrix", help="Run all 10 models on one task")
    p.add_argument("--task", required=True, choices=EVAL_TASKS)
    p.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), default=None,
                   help="Run only this model (default: all)")
    p.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    p.add_argument("--strategy", default="zero_shot", choices=PROMPT_STRATEGIES)
    p.add_argument("--granularity", choices=["question", "patient", "encounter", "specialty"],
                   help="encounter=current-visit, specialty=specialty-conditioned summarization variant")
    p.add_argument("--pilot", type=int, default=None)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--tier", choices=["small", "medium", "large"], default=None,
                   help="specialty-conditioned only: select rows by the frozen patient-aligned cohort")
    p.set_defaults(func=cmd_matrix)

    # validate-gt
    p = sub.add_parser("validate-gt", help="Validate ground truth rows")
    p.add_argument("--task", choices=EVAL_TASKS, default=None)
    p.set_defaults(func=cmd_validate_gt)

    # classify-errors
    p = sub.add_parser("classify-errors", help="Classify prediction errors")
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--all", action="store_true", help="Classify all completed runs")
    p.set_defaults(func=cmd_classify_errors)

    # pilot-report
    p = sub.add_parser("pilot-report", help="Pilot report with delta-vs-baseline")
    p.add_argument("--task", choices=EVAL_TASKS, default=None)
    p.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    p.add_argument("--strategy", required=True, choices=PROMPT_STRATEGIES)
    p.add_argument("--baseline", default="zero_shot", choices=PROMPT_STRATEGIES)
    p.set_defaults(func=cmd_pilot_report)

    # list-runs
    p = sub.add_parser("list-runs", help="Show existing evaluation runs")
    p.add_argument("--task", choices=EVAL_TASKS)
    p.set_defaults(func=cmd_list_runs)

    # --- Phase F: Agent evaluation ---
    AGENT_MODELS_LIST = ["gpt-5.3", "gemini-3.1", "glm-5-agent", "opus-4.6", "kimi-2.5-thinking-improved", "kimi-2.5-improved"]
    AGENT_TASKS_LIST = ["patient_diagnosis", "context_summarization", "evidence_retrieval", "imaging_indication"]

    # agent-run
    p = sub.add_parser("agent-run", help="Run agent evaluation (Phase F)")
    p.add_argument("--task", required=True, choices=AGENT_TASKS_LIST)
    p.add_argument("--model", required=True, choices=AGENT_MODELS_LIST)
    p.add_argument("--arm", required=True, choices=["structured", "bash"])
    p.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    p.add_argument("--pilot", type=int, default=None, help="Limit to N patients")
    p.add_argument("--max-patients", type=int, default=None,
                   help="Truncate patient list to N (default: 100, Kimi: 50)")
    p.add_argument("--api-base", default="http://localhost:8000")
    p.add_argument("--container-id", default=None, help="Docker container ID for bash arm")
    p.add_argument("--resume-run", type=int, default=None, help="Resume an incomplete run by run_id")
    p.set_defaults(func=cmd_agent_run)

    # agent-report
    p = sub.add_parser("agent-report", help="Phase F comparison report")
    p.add_argument("--task", choices=AGENT_TASKS_LIST, default=None)
    p.add_argument("--model", choices=AGENT_MODELS_LIST, default=None)
    p.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    p.add_argument("--format", default="terminal", choices=["terminal", "csv"])
    p.add_argument("--output", type=str, default=None)
    p.set_defaults(func=cmd_agent_report)

    # agent-select-patients
    p = sub.add_parser("agent-select-patients", help="Select patients for Phase F")
    p.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--no-ceiling-exclude", action="store_true")
    p.set_defaults(func=cmd_agent_select_patients)

    args = parser.parse_args()
    args.func(args)


def cmd_agent_run(args):
    """Run agent evaluation for one (model, arm, task)."""
    from eval.agents.runner import get_auth_token, run_agent_evaluation, start_bash_container

    container_id = args.container_id
    if args.arm == "bash" and not container_id:
        log.info("Starting bash_sandbox container...")
        container_id = start_bash_container()

    auth_token = get_auth_token(args.api_base)

    run_id = run_agent_evaluation(
        model_name=args.model,
        arm=args.arm,
        task=args.task,
        split=args.split,
        pilot=args.pilot,
        max_patients=args.max_patients,
        api_base=args.api_base,
        container_id=container_id,
        auth_token=auth_token,
        resume_run_id=args.resume_run,
    )
    print(f"Agent run completed: run_id={run_id}")


def cmd_agent_report(args):
    """Generate Phase F comparison report."""
    from eval.agents.report import export_agent_csv, generate_agent_report

    if args.format == "csv":
        output = args.output or "results/phase_f_report.csv"
        export_agent_csv(output, split=args.split)
        print(f"Exported to {output}")
    else:
        report = generate_agent_report(split=args.split, task=args.task, model=args.model)
        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)


def cmd_agent_select_patients(args):
    """Select and display patients for Phase F."""
    from eval.agents.runner import select_patients
    from eval.config import get_pg_connection

    conn = get_pg_connection()
    try:
        patients = select_patients(
            conn,
            split=args.split,
            n=args.n,
            exclude_ceiling=not args.no_ceiling_exclude,
        )
        print(f"Selected {len(patients)} patients:")
        for p in patients[:20]:
            tasks = list(p.get("tasks", {}).keys())
            print(f"  patient_id={p['patient_id']}  encounters={p['num_encounters']}  tasks={tasks}")
        if len(patients) > 20:
            print(f"  ... and {len(patients) - 20} more")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
