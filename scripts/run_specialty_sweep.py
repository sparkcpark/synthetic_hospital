"""Sequential OpenRouter sweep for the specialty-conditioned small set.

Runs each remaining model on `val` (983 rows) one at a time — sequential, not concurrent,
to stay under OpenRouter account rate limits. Resume-aware at the MODEL level: a model that
already has a complete run (>= EXPECTED specialty predictions for this split) is skipped, so
re-running the driver after an interruption picks up where it left off. A model that errors
is logged and skipped (the sweep continues).

Prereq: OPENROUTER_API_KEY in the environment. Run once under `caffeinate -is`:
  caffeinate -is .venv/bin/python scripts/run_specialty_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from eval.config import get_pg_connection
from eval.runner import run_evaluation

MODELS = ["gpt-5.3", "opus-4.6", "gemini-3.1", "deepseek-v3.2",
          "qwen-3", "mistral-3", "llama-4", "gemma-3"]
SPLIT = "val"
EXPECTED = 983
WORKERS = 20


def complete_count(conn, model):
    """Max specialty-conditioned predictions across any run for (model, split)."""
    row = conn.execute("""
        SELECT COALESCE(MAX(c), 0) FROM (
            SELECT COUNT(*) c
            FROM evaluation_runs er
            JOIN evaluation_predictions ep ON ep.run_id = er.run_id
            JOIN benchmark_ground_truth gt ON gt.gt_id = ep.gt_id
            WHERE er.model_name = %s AND er.split = %s
              AND gt.ground_truth->>'variant' = 'specialty_conditioned'
            GROUP BY er.run_id
        ) t""", (model, SPLIT)).fetchone()
    return row[0]


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set — export it before running the sweep.")
    conn = get_pg_connection()
    print(f"OpenRouter specialty sweep: {len(MODELS)} models on split={SPLIT} ({EXPECTED} rows), {WORKERS} workers\n")
    for i, m in enumerate(MODELS, 1):
        n = complete_count(conn, m)
        if n >= EXPECTED:
            print(f"[{i}/{len(MODELS)}] SKIP {m}: already complete ({n} preds)")
            continue
        print(f"[{i}/{len(MODELS)}] RUN  {m} ...")
        try:
            rid = run_evaluation(task="context_summarization", model_name=m, split=SPLIT,
                                 granularity="specialty", workers=WORKERS)
            print(f"[{i}/{len(MODELS)}] DONE {m}: run_id={rid}")
        except Exception as e:
            print(f"[{i}/{len(MODELS)}] FAIL {m}: {type(e).__name__}: {e}  (continuing)")
    print("\nsweep finished.")


if __name__ == "__main__":
    main()
