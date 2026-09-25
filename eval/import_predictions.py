"""Import JSONL predictions from remote execution into PostgreSQL.

Usage:
    python -m eval.import_predictions \
        --input results/deepseek-v3.2_patient_diagnosis_cot.jsonl \
        --task patient_diagnosis \
        --model deepseek-v3.2 \
        --split val \
        --strategy zero_shot

    # Then score:
    python -m eval.cli score --run-id <N>

Reads JSONL prediction files produced by remote_runner.py, creates an
evaluation_runs record, and inserts predictions into evaluation_predictions.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from eval.config import EVAL_TASKS, MODEL_REGISTRY, TASK_MAX_TOKENS, get_pg_connection

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_jsonl(path: str) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def create_run(conn, task: str, model_name: str, split: str,
               prompt_strategy: str) -> int:
    """Create evaluation_runs record. Mirrors runner.py _create_run()."""
    run_name = f"{model_name}_{task}_{split}_{prompt_strategy}"

    config = MODEL_REGISTRY.get(model_name)
    run_config = {}
    if config:
        run_config = {
            "model_id": config.model_id,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "extra": config.extra,
        }
    if task in TASK_MAX_TOKENS:
        run_config["task_max_tokens"] = TASK_MAX_TOKENS[task]
    run_config["imported_from"] = "remote_runner"

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO evaluation_runs
                (run_name, method_name, method_type, task, split,
                 config, prompt_strategy, model_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING run_id
        """, (
            run_name,
            model_name,
            "llm_direct",
            task,
            split,
            json.dumps(run_config),
            prompt_strategy,
            model_name,
        ))
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def import_predictions(conn, run_id: int, records: list[dict],
                       prompt_strategy: str) -> int:
    """Insert prediction records into evaluation_predictions. Returns count inserted."""
    n = 0
    with conn.cursor() as cur:
        for rec in records:
            cur.execute("""
                INSERT INTO evaluation_predictions
                    (run_id, gt_id, prediction, latency_ms, token_count,
                     input_tokens, output_tokens, prompt_strategy, raw_output)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, gt_id) DO NOTHING
            """, (
                run_id,
                rec["gt_id"],
                json.dumps(rec["prediction"]),
                rec.get("latency_ms"),
                (rec.get("input_tokens", 0) or 0) + (rec.get("output_tokens", 0) or 0),
                rec.get("input_tokens"),
                rec.get("output_tokens"),
                prompt_strategy,
                rec.get("raw_output", "")[:10000],
            ))
            n += 1
    conn.commit()
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Import JSONL predictions into PostgreSQL",
    )
    parser.add_argument("--input", required=True, help="JSONL prediction file from remote_runner.py")
    parser.add_argument("--task", required=True, choices=EVAL_TASKS)
    parser.add_argument("--model", required=True, help="Model name (e.g., deepseek-v3.2)")
    parser.add_argument("--split", required=True, choices=["public", "train", "heldout"])
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot", "structured"])
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    records = read_jsonl(args.input)
    log.info("Read %d predictions from %s", len(records), args.input)

    if not records:
        log.warning("No records to import")
        sys.exit(0)

    conn = get_pg_connection()
    try:
        run_id = create_run(conn, args.task, args.model, args.split, args.strategy)
        log.info("Created evaluation run: run_id=%d (%s/%s/%s/%s)",
                 run_id, args.task, args.model, args.split, args.strategy)

        n = import_predictions(conn, run_id, records, args.strategy)
        log.info("Imported %d predictions into run_id=%d", n, run_id)

        # Mark run as completed
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE evaluation_runs SET completed_at = %s WHERE run_id = %s",
                (datetime.now(timezone.utc).isoformat(), run_id),
            )
        conn.commit()

        log.info("Done. Score with: python -m eval.cli score --run-id %d", run_id)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
