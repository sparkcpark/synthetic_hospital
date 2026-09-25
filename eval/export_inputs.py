"""Export evaluation inputs from PostgreSQL to JSONL files for remote execution.

Usage:
    python -m eval.export_inputs --split val [--pilot 75] [--task patient_diagnosis]

Calls existing load_inputs() per task and serializes TaskInput dataclasses to JSONL.
Output directory: exports/{split}_{task}.jsonl
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict

from eval.config import EVAL_TASKS, get_pg_connection
from eval.tasks import TASK_LOADERS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")


def export_task(conn, task: str, split: str, pilot: int | None = None) -> int:
    """Export a single task's inputs to JSONL. Returns number of items exported."""
    loader = TASK_LOADERS[task]
    inputs = loader(conn, split=split, pilot=pilot)
    log.info("Loaded %d inputs for task=%s split=%s", len(inputs), task, split)

    if not inputs:
        log.warning("No inputs found for task=%s split=%s", task, split)
        return 0

    os.makedirs(EXPORT_DIR, exist_ok=True)
    out_path = os.path.join(EXPORT_DIR, f"{split}_{task}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for inp in inputs:
            record = asdict(inp)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info("Exported %d items to %s (%.1f MB)", len(inputs), out_path, size_mb)
    return len(inputs)


def main():
    parser = argparse.ArgumentParser(description="Export eval inputs to JSONL")
    parser.add_argument("--split", default="public", choices=["public", "train", "heldout"])
    parser.add_argument("--task", default=None, help="Single task (default: all tasks)")
    parser.add_argument("--pilot", type=int, default=None, help="Limit items per task")
    args = parser.parse_args()

    tasks = [args.task] if args.task else EVAL_TASKS

    conn = get_pg_connection()
    try:
        total = 0
        for task in tasks:
            n = export_task(conn, task, args.split, args.pilot)
            total += n
        log.info("Export complete: %d total items across %d tasks", total, len(tasks))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
