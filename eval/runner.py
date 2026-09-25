"""Evaluation orchestrator: load GT, call models, store predictions, compute metrics.

Thread safety: LLM calls run in ThreadPoolExecutor threads; ALL DB writes happen in
the main thread (matching the etl/stages/s10_ground_truth.py pattern).
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from eval.adapters import ModelResponse, create_adapter
from eval.config import MODEL_REGISTRY, get_pg_connection
from eval.scoring import compute_all_metrics
from eval.tasks import TASK_FORMATTERS, TASK_LOADERS, TASK_PARSERS

log = logging.getLogger(__name__)

# Commit DB writes every N predictions
BATCH_COMMIT_SIZE = 10


# ============================================================================
# Main LLM evaluation
# ============================================================================

def run_evaluation(
    task: str,
    model_name: str,
    split: str = "public",
    prompt_strategy: str = "zero_shot",
    granularity: str | None = None,
    pilot: int | None = None,
    workers: int = 5,
    resume_run_id: int | None = None,
    tier: str | None = None,
    only_gt_ids: set[int] | None = None,
) -> int:
    """Run a full evaluation: load inputs, call model, store predictions, score.

    If `resume_run_id` is given, reuse that run and skip already-completed gt_ids
    (interruption-safe: re-run with --resume <run_id> after a sleep/crash).
    `tier` (specialty-conditioned summarization only) selects rows by the frozen
    patient-aligned cohort (small|medium|large) instead of the split column.
    `only_gt_ids` restricts the run to a specific item set -- used to top up
    baseline coverage on the exact items an agent arm was evaluated on, without
    mutating the published baseline run.
    Returns the run_id.
    """
    config = MODEL_REGISTRY[model_name]
    adapter = create_adapter(config)
    conn = get_pg_connection()

    try:
        # 1. Create (or resume) the evaluation run
        if resume_run_id is not None:
            run_id = resume_run_id
            log.info("Resuming run_id=%d: %s / %s / %s / %s", run_id, task, model_name, split, prompt_strategy)
        else:
            run_id = _create_run(conn, task, model_name, split, prompt_strategy, config,
                                 tier=tier, granularity=granularity)
            log.info("Created run_id=%d: %s / %s / %s / %s", run_id, task, model_name, split, prompt_strategy)

        # 2. Load task inputs
        loader = TASK_LOADERS[task]
        loader_kwargs = {"split": split, "granularity": granularity, "pilot": pilot}
        if tier is not None:  # only the summarization loader accepts tier=
            loader_kwargs["tier"] = tier
        inputs = loader(conn, **loader_kwargs)
        log.info("Loaded %d inputs for task=%s", len(inputs), task)

        if not inputs:
            log.warning("No inputs found — completing run with empty metrics")
            _complete_run(conn, run_id, {})
            return run_id

        # 3. Resume: find already-completed gt_ids
        completed_gt_ids = _get_completed_gt_ids(conn, run_id)
        remaining = [inp for inp in inputs if inp.gt_id not in completed_gt_ids]
        if only_gt_ids is not None:
            remaining = [inp for inp in remaining if inp.gt_id in only_gt_ids]
        log.info("Resuming: %d already done, %d remaining", len(completed_gt_ids), len(remaining))

        # 4. Format prompts and call model in parallel
        formatter = TASK_FORMATTERS[task]
        parser = TASK_PARSERS[task]

        def _call_model(inp):
            """Thread-safe: format prompt + call LLM. No DB writes here."""
            # TODO: Few-shot example selection logic pending implementation.
            # format_prompt() currently falls back to zero_shot when strategy='few_shot'.
            system_prompt, user_prompt = formatter(inp, prompt_strategy)
            response = adapter.call_with_retry(system_prompt, user_prompt)
            parsed = parser(response.text)
            return inp, response, parsed

        results = []
        errors = 0

        if workers > 1 and len(remaining) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_call_model, inp): inp for inp in remaining}
                for future in as_completed(futures):
                    inp = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        errors += 1
                        log.error("Failed gt_id=%d: %s", inp.gt_id, e)

                    # Batch commit from main thread
                    if len(results) >= BATCH_COMMIT_SIZE:
                        _store_predictions(conn, run_id, results, prompt_strategy)
                        results = []
        else:
            for inp in remaining:
                try:
                    result = _call_model(inp)
                    results.append(result)
                except Exception as e:
                    errors += 1
                    log.error("Failed gt_id=%d: %s", inp.gt_id, e)

                if len(results) >= BATCH_COMMIT_SIZE:
                    _store_predictions(conn, run_id, results, prompt_strategy)
                    results = []

        # Flush remaining
        if results:
            _store_predictions(conn, run_id, results, prompt_strategy)

        log.info("Predictions stored: %d success, %d errors", len(inputs) - len(remaining) + len(completed_gt_ids) + (len(remaining) - errors), errors)

        # 5. Compute metrics
        metrics = _score_run(conn, run_id, task, inputs)

        # 6. Complete run
        _complete_run(conn, run_id, metrics)
        log.info("Run %d completed. Metrics: %s", run_id, json.dumps(metrics, indent=2))
        return run_id

    finally:
        conn.close()


# ============================================================================
# Retrieval baselines (deterministic, no LLM)
# ============================================================================

def run_retrieval_baseline(
    method: str,
    split: str = "public",
    pilot: int | None = None,
) -> int:
    """Run a non-LLM retrieval baseline (bm25, sapbert, hybrid).

    Returns the run_id.
    """
    from eval.tasks.retrieval import bm25_retrieve, hybrid_retrieve, sapbert_retrieve

    method_type_map = {
        "bm25": "sparse_retrieval",
        "sapbert": "dense_retrieval",
        "hybrid": "hybrid_retrieval",
    }
    retrieve_fn_map = {
        "bm25": bm25_retrieve,
        "sapbert": sapbert_retrieve,
        "hybrid": hybrid_retrieve,
    }

    retrieve_fn = retrieve_fn_map[method]
    conn = get_pg_connection()

    try:
        # 1. Create run
        run_id = _create_baseline_run(conn, method, split, method_type_map[method])
        log.info("Created baseline run_id=%d: %s / %s", run_id, method, split)

        # 2. Load retrieval inputs
        from eval.tasks.retrieval import load_inputs
        inputs = load_inputs(conn, split=split, pilot=pilot)
        log.info("Loaded %d retrieval inputs", len(inputs))

        if not inputs:
            _complete_run(conn, run_id, {})
            return run_id

        # 3. Resume
        completed_gt_ids = _get_completed_gt_ids(conn, run_id)
        remaining = [inp for inp in inputs if inp.gt_id not in completed_gt_ids]
        log.info("Resuming: %d done, %d remaining", len(completed_gt_ids), len(remaining))

        # 4. Run retrieval (deterministic, single-threaded)
        results_batch = []
        for inp in remaining:
            t0 = time.monotonic()
            ranked = retrieve_fn(inp.query, inp.corpus, k=20)
            latency_ms = int((time.monotonic() - t0) * 1000)

            prediction = {"rankings": ranked}
            results_batch.append((inp, prediction, latency_ms))

            if len(results_batch) >= BATCH_COMMIT_SIZE:
                _store_baseline_predictions(conn, run_id, results_batch)
                results_batch = []

        if results_batch:
            _store_baseline_predictions(conn, run_id, results_batch)

        # 5. Score
        metrics = _score_run(conn, run_id, "evidence_retrieval", inputs)

        # 6. Complete
        _complete_run(conn, run_id, metrics)
        log.info("Baseline run %d completed. Metrics: %s", run_id, json.dumps(metrics, indent=2))
        return run_id

    finally:
        conn.close()


# ============================================================================
# Rescore existing run
# ============================================================================

def rescore_run(run_id: int) -> dict:
    """Recompute metrics for an existing run and update the DB."""
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT task, split, config FROM evaluation_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Run {run_id} not found")
            task, split, run_config = row
        run_config = (run_config if isinstance(run_config, dict)
                      else json.loads(run_config)) if run_config else {}

        # Load inputs for context (ehr_texts, judgments, etc.) — variant/tier-aware
        # for runs that recorded granularity/dataset_tier in their config.
        loader = TASK_LOADERS[task]
        loader_kwargs = {"split": split}
        if run_config.get("granularity"):
            loader_kwargs["granularity"] = run_config["granularity"]
        if run_config.get("dataset_tier"):
            loader_kwargs["tier"] = run_config["dataset_tier"]
        inputs = loader(conn, **loader_kwargs)

        metrics = _score_run(conn, run_id, task, inputs)

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE evaluation_runs SET metrics = %s WHERE run_id = %s",
                (json.dumps(metrics), run_id),
            )
        conn.commit()
        log.info("Rescored run %d: %s", run_id, json.dumps(metrics, indent=2))
        return metrics
    finally:
        conn.close()


# ============================================================================
# Internal helpers
# ============================================================================

def _create_run(conn, task: str, model_name: str, split: str,
                prompt_strategy: str, config, tier: str | None = None,
                granularity: str | None = None) -> int:
    """INSERT into evaluation_runs and return the run_id."""
    run_name = f"{model_name}_{task}_{tier or split}_{prompt_strategy}"

    run_config = {
        "model_id": config.model_id,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "extra": config.extra,
    }
    if tier is not None:
        run_config["dataset_tier"] = tier  # frozen cohort from specialty_dataset_manifest.json
    if granularity is not None:
        run_config["granularity"] = granularity  # so rescore_run reloads the right variant
    # Store task-specific max_tokens for reproducibility
    from eval.config import TASK_MAX_TOKENS
    if task in TASK_MAX_TOKENS:
        run_config["task_max_tokens"] = TASK_MAX_TOKENS[task]

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


def _create_baseline_run(conn, method: str, split: str, method_type: str) -> int:
    """INSERT a retrieval baseline run."""
    run_name = f"{method}_evidence_retrieval_{split}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO evaluation_runs
                (run_name, method_name, method_type, task, split, model_name)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING run_id
        """, (
            run_name,
            method,
            method_type,
            "evidence_retrieval",
            split,
            method,
        ))
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def _get_completed_gt_ids(conn, run_id: int) -> set[int]:
    """Get gt_ids that already have predictions for this run (for resume)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT gt_id FROM evaluation_predictions WHERE run_id = %s",
            (run_id,),
        )
        return {row[0] for row in cur.fetchall()}


def _store_predictions(conn, run_id: int, results: list[tuple], prompt_strategy: str):
    """Store LLM prediction results. Called from main thread only."""
    with conn.cursor() as cur:
        for inp, response, parsed in results:
            cur.execute("""
                INSERT INTO evaluation_predictions
                    (run_id, gt_id, prediction, latency_ms, token_count,
                     input_tokens, output_tokens, prompt_strategy, raw_output)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, gt_id) DO NOTHING
            """, (
                run_id,
                inp.gt_id,
                json.dumps(parsed),
                response.latency_ms,
                response.input_tokens + response.output_tokens,
                response.input_tokens,
                response.output_tokens,
                prompt_strategy,
                response.text[:10000],  # Truncate raw output for storage
            ))
    conn.commit()
    log.info("Stored %d predictions for run_id=%d", len(results), run_id)


def _store_baseline_predictions(conn, run_id: int, results: list[tuple]):
    """Store retrieval baseline predictions. Called from main thread only."""
    with conn.cursor() as cur:
        for inp, prediction, latency_ms in results:
            cur.execute("""
                INSERT INTO evaluation_predictions
                    (run_id, gt_id, prediction, latency_ms)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id, gt_id) DO NOTHING
            """, (
                run_id,
                inp.gt_id,
                json.dumps(prediction),
                latency_ms,
            ))
    conn.commit()
    log.info("Stored %d baseline predictions for run_id=%d", len(results), run_id)


def _score_run(conn, run_id: int, task: str, inputs: list) -> dict:
    """Load predictions from DB and compute all metrics for a run."""
    # Build gt_id → input lookup
    input_map = {inp.gt_id: inp for inp in inputs}

    # Load all predictions for this run
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ep.gt_id, ep.prediction, ep.latency_ms, ep.input_tokens, ep.output_tokens,
                   bgt.ground_truth
            FROM evaluation_predictions ep
            JOIN benchmark_ground_truth bgt ON ep.gt_id = bgt.gt_id
            WHERE ep.run_id = %s AND bgt.is_diagnostic
            ORDER BY ep.gt_id
        """, (run_id,))
        rows = cur.fetchall()

    if not rows:
        log.warning("No predictions found for run_id=%d", run_id)
        return {}

    predictions = []
    ground_truths = []
    latencies_ms = []
    input_tokens = []
    output_tokens = []
    ehr_texts = []

    for gt_id, pred_json, lat_ms, inp_tok, out_tok, gt_json in rows:
        pred = pred_json if isinstance(pred_json, dict) else json.loads(pred_json)
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)

        predictions.append(pred)
        ground_truths.append(gt)
        latencies_ms.append(lat_ms or 0)
        input_tokens.append(inp_tok or 0)
        output_tokens.append(out_tok or 0)

        # For summarization hallucination rate, we need the EHR text
        inp_obj = input_map.get(gt_id)
        if inp_obj and hasattr(inp_obj, "ehr_text"):
            ehr_texts.append(inp_obj.ehr_text)

    # For retrieval, attach judgments to ground_truths
    if task == "evidence_retrieval":
        for i, gt_id_row in enumerate(r[0] for r in rows):
            inp_obj = input_map.get(gt_id_row)
            if inp_obj:
                ground_truths[i]["_judgments"] = inp_obj.judgments

    # Compute task-specific metrics
    kwargs = {}
    if task == "context_summarization" and ehr_texts:
        kwargs["ehr_texts"] = ehr_texts
    if task == "evidence_retrieval":
        kwargs["latencies_ms"] = latencies_ms
    if task == "imaging_indication":
        from eval.imaging_concepts import ConceptExtractor
        kwargs["concept_extractor"] = ConceptExtractor.from_db(conn)

    metrics = compute_all_metrics(task, predictions, ground_truths, **kwargs)

    # Add aggregate stats
    metrics["n_predictions"] = len(predictions)
    metrics["mean_latency_ms"] = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    metrics["total_input_tokens"] = sum(input_tokens)
    metrics["total_output_tokens"] = sum(output_tokens)

    return metrics


def _complete_run(conn, run_id: int, metrics: dict):
    """Mark a run as completed with metrics."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE evaluation_runs
            SET metrics = %s, completed_at = %s
            WHERE run_id = %s
        """, (
            json.dumps(metrics),
            datetime.now(timezone.utc).isoformat(),
            run_id,
        ))
    conn.commit()
