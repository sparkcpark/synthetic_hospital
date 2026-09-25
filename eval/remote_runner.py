"""Lightweight eval runner for remote execution — reads JSONL, calls models, writes JSONL.

No database required. Designed to run on CMU Babel or any remote host with
Python + httpx + OpenRouter access.

Usage:
    python -m eval.remote_runner \
        --task patient_diagnosis \
        --model deepseek-v3.2 \
        --input exports/public_patient_diagnosis.jsonl \
        --output results/deepseek-v3.2_patient_diagnosis_cot.jsonl \
        --strategy zero_shot \
        [--workers 5]

Reuses eval.tasks.TASK_FORMATTERS/TASK_PARSERS and eval.adapters for model calls.
Resume support: on restart, skips gt_ids already present in the output file.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from eval.adapters import create_adapter
from eval.config import MODEL_REGISTRY, EVAL_TASKS
from eval.tasks import TASK_FORMATTERS, TASK_PARSERS

# Task-specific dataclasses
from eval.tasks.diagnosis import DiagnosisInput
from eval.tasks.patient_diagnosis import PatientDiagnosisInput
from eval.tasks.summarization import SummarizationInput
from eval.tasks.retrieval import RetrievalInput, Passage
from eval.tasks.imaging import ImagingInput

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

# Thread-safe JSONL writer
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _reconstruct_input(task: str, item: dict):
    """Reconstruct the appropriate TaskInput dataclass from a JSONL dict."""
    if task == "diagnosis_accuracy":
        return DiagnosisInput(**item)
    elif task == "patient_diagnosis":
        return PatientDiagnosisInput(**item)
    elif task == "context_summarization":
        return SummarizationInput(**item)
    elif task == "evidence_retrieval":
        # Reconstruct Passage objects from dicts
        corpus = [Passage(**p) for p in item.pop("corpus", [])]
        return RetrievalInput(corpus=corpus, **item)
    elif task == "imaging_indication":
        return ImagingInput(**item)
    else:
        raise ValueError(f"Unknown task: {task}")


def load_jsonl_inputs(task: str, input_file: str, pilot: int | None = None) -> list:
    """Load TaskInput instances from a JSONL file."""
    inputs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            inputs.append(_reconstruct_input(task, item))
            if pilot and len(inputs) >= pilot:
                break
    return inputs


def load_completed_gt_ids(output_file: str) -> set[int]:
    """Read existing output JSONL and return set of completed gt_ids."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(record["gt_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_jsonl(output_file: str, record: dict):
    """Thread-safe append of a single JSON record to output file."""
    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_remote(
    task: str,
    model_name: str,
    input_file: str,
    output_file: str,
    strategy: str = "zero_shot",
    workers: int = 5,
    pilot: int | None = None,
):
    """Run model evaluation on JSONL inputs, write predictions to JSONL output."""
    config = MODEL_REGISTRY[model_name]
    adapter = create_adapter(config)
    formatter = TASK_FORMATTERS[task]
    parser = TASK_PARSERS[task]

    # Load inputs
    inputs = load_jsonl_inputs(task, input_file, pilot=pilot)
    log.info("Loaded %d inputs from %s", len(inputs), input_file)

    if not inputs:
        log.warning("No inputs found in %s", input_file)
        return

    # Resume: skip completed
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    completed = load_completed_gt_ids(output_file)
    remaining = [inp for inp in inputs if inp.gt_id not in completed]
    log.info("Resume: %d completed, %d remaining", len(completed), len(remaining))

    if not remaining:
        log.info("All items already completed. Nothing to do.")
        return

    # Track progress
    n_success = 0
    n_errors = 0
    t_start = time.monotonic()

    def _call_model(inp):
        """Thread-safe: format prompt + call LLM + parse output."""
        system_prompt, user_prompt = formatter(inp, strategy)
        response = adapter.call_with_retry(system_prompt, user_prompt)
        parsed = parser(response.text)
        return inp, response, parsed

    def _process_result(inp, response, parsed):
        """Write prediction to JSONL output."""
        record = {
            "gt_id": inp.gt_id,
            "prediction": parsed,
            "raw_output": response.text[:10000],
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "model_name": model_name,
            "prompt_strategy": strategy,
            "task": task,
        }
        append_jsonl(output_file, record)

    # Run with parallelism
    if workers > 1 and len(remaining) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_call_model, inp): inp for inp in remaining}
            for future in as_completed(futures):
                inp = futures[future]
                try:
                    inp_out, response, parsed = future.result()
                    _process_result(inp_out, response, parsed)
                    n_success += 1
                except Exception as e:
                    n_errors += 1
                    log.error("Failed gt_id=%d: %s", inp.gt_id, e)

                done = n_success + n_errors
                if done % 10 == 0 or done == len(remaining):
                    elapsed = time.monotonic() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    log.info(
                        "Progress: %d/%d (%.1f/s) — %d ok, %d err",
                        done, len(remaining), rate, n_success, n_errors,
                    )
    else:
        for inp in remaining:
            try:
                inp_out, response, parsed = _call_model(inp)
                _process_result(inp_out, response, parsed)
                n_success += 1
            except Exception as e:
                n_errors += 1
                log.error("Failed gt_id=%d: %s", inp.gt_id, e)

            done = n_success + n_errors
            if done % 10 == 0 or done == len(remaining):
                elapsed = time.monotonic() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                log.info(
                    "Progress: %d/%d (%.1f/s) — %d ok, %d err",
                    done, len(remaining), rate, n_success, n_errors,
                )

    elapsed = time.monotonic() - t_start
    log.info(
        "Done: %d success, %d errors in %.1fs (%.1f items/s)",
        n_success, n_errors, elapsed, (n_success + n_errors) / elapsed if elapsed > 0 else 0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remote eval runner — JSONL in, JSONL out, no DB required",
    )
    parser.add_argument("--task", required=True, choices=EVAL_TASKS)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--input", required=True, help="Input JSONL file (from export_inputs.py)")
    parser.add_argument("--output", required=True, help="Output JSONL file for predictions")
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot", "structured"])
    parser.add_argument("--workers", type=int, default=5, help="Parallel LLM call threads")
    parser.add_argument("--pilot", type=int, default=None, help="Limit to N items")
    args = parser.parse_args()

    log.info("Starting: task=%s model=%s strategy=%s workers=%d",
             args.task, args.model, args.strategy, args.workers)

    run_remote(
        task=args.task,
        model_name=args.model,
        input_file=args.input,
        output_file=args.output,
        strategy=args.strategy,
        workers=args.workers,
        pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
