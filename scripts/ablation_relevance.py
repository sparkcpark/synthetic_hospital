"""Paired ablation for the typed-edge relevance machinery (THREE conditions).

The novelty evidence (NOT the headline conditioned_f1, which is ~98% primary-driven). Same
credited-relevant rows (involved rows with >=1 gold = Class-1/curated relevant finding), three
prompt conditions on the SAME model, scored PER ROW — never as independent aggregates:

  graph    = SPECIALTY_ZERO_SHOT   (specialty problems + relevant comorbidities — the method)
  neutral  = SPECIALTY_NEUTRAL     (specialty framing only; NO graph cue, NO omit)  <-- baseline
  omit     = SPECIALTY_SAME_ONLY   (specialty's own problems, explicitly NO comorbidities) <-- floor

Each condition gets its OWN clinical question (the GT question mentions comorbidities, so it
cannot be reused for neutral/omit). Per-row relevant_recall (cascade over the row's gold
relevant findings) under each; then PAIRED Wilcoxon signed-rank + paired bootstrap on:
  PRIMARY   Δ = graph − neutral   (the real evidence: does the graph add coverage beyond
                                    the model's spontaneous comorbidity inclusion?)
  SECONDARY Δ = neutral − omit    (is spontaneous inclusion real but incomplete?)
Expected ordering omit < neutral < graph. primary_recall across conditions is a manipulation
check (should be ~equal). Bootstrap resamples ROWS to preserve pairing. -> per-row CSV.

Usage: python scripts/ablation_relevance.py --model kimi-2.5-thinking --workers 16
(needs the gateway; 3*N calls — run under `caffeinate -is`).
"""
import argparse
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import wilcoxon

from eval.adapters import create_adapter
from eval.config import MODEL_REGISTRY, get_pg_connection
from eval import prompts as P, scoring
from eval.tasks.summarization import load_inputs, format_prompt, parse_output

CONDITIONS = ("graph", "neutral", "omit")


def _build_prompts(inp):
    """The three (system, user) prompts for one row, each with a condition-appropriate question."""
    label = inp.specialty.replace("_", "/")
    graph = format_prompt(inp, "zero_shot")                      # method: graph-aware GT question
    neutral = (P.SPECIALTY_NEUTRAL.system, P.SPECIALTY_NEUTRAL.user.format(
        clinical_question=f"Summarize this patient's chart from a {label} perspective.",
        ehr_text=inp.ehr_text, few_shot_examples="", structured_hints=""))
    omit = (P.SPECIALTY_SAME_ONLY.system, P.SPECIALTY_SAME_ONLY.user.format(
        clinical_question=f"Summarize only the active {label} problems.",
        ehr_text=inp.ehr_text, few_shot_examples="", structured_hints=""))
    return {"graph": graph, "neutral": neutral, "omit": omit}


def _recall(summary, findings):
    r, _ = scoring._list_recall_cascade([summary], [findings])
    return r


def paired_bootstrap(diffs, n=10000, seed=0):
    """95% CI on the mean per-row difference by resampling ROWS (preserves pairing)."""
    rng = random.Random(seed)
    N = len(diffs)
    means = sorted(sum(diffs[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired_report(name, a_vals, b_vals, seed):
    diffs = np.array(a_vals) - np.array(b_vals)
    try:
        _, p = wilcoxon(diffs)
    except ValueError:
        p = float("nan")                       # degenerate (all diffs zero)
    lo, hi = paired_bootstrap(list(diffs), seed=seed)
    print(f"  {name:16s} Δ={diffs.mean():+.3f}  Wilcoxon p={p:.2e}  "
          f"boot95%=[{lo:+.3f},{hi:+.3f}]  Δ>0:{int((diffs > 0).sum())}/{len(diffs)} "
          f"(Δ<0:{int((diffs < 0).sum())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kimi-2.5-thinking")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    conn = get_pg_connection()
    rows = (load_inputs(conn, split="val", granularity="specialty")
            + load_inputs(conn, split="test", granularity="specialty"))
    rows = [r for r in rows
            if r.ground_truth.get("involvement") == "involved"
            and r.ground_truth["tiers"]["relevant"]]
    print(f"credited-relevant rows (full set): {len(rows)}")

    adapter = create_adapter(MODEL_REGISTRY[args.model])

    def gen(inp):
        out = {}
        for cond, (s, u) in _build_prompts(inp).items():
            out[cond] = parse_output(adapter.call_with_retry(s, u).text).get("summary", "")
        return inp, out

    per_row = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(gen, r): r for r in rows}
        for fut in as_completed(futs):
            try:
                inp, summ = fut.result()
            except Exception as e:
                print(f"  skip gt_id={futs[fut].gt_id}: {e}")
                continue
            rel = inp.ground_truth["tiers"]["relevant"]
            prim = inp.ground_truth["tiers"]["primary"]
            row = {"gt_id": inp.gt_id, "specialty": inp.specialty}
            for c in CONDITIONS:
                row[f"rel_{c}"] = _recall(summ[c], rel)
                row[f"prim_{c}"] = _recall(summ[c], prim)
            per_row.append(row)

    slug = args.model.replace("/", "_")
    out_csv = f"ablation_relevance_perrow_{slug}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_row[0])); w.writeheader(); w.writerows(per_row)

    def col(metric, cond):
        return [r[f"{metric}_{cond}"] for r in per_row]

    print(f"\n=== PAIRED relevance ablation (n={len(per_row)}, model={args.model}) ===")
    print("relevant_recall by condition (expect omit < neutral < graph):")
    for c in CONDITIONS:
        print(f"  {c:8s} {np.mean(col('rel', c)):.3f}")
    print("paired contrasts on per-row Δ:")
    paired_report("graph − neutral", col("rel", "graph"), col("rel", "neutral"), args.seed)   # PRIMARY
    paired_report("neutral − omit", col("rel", "neutral"), col("rel", "omit"), args.seed)      # SECONDARY
    print("manipulation check — primary_recall by condition (should be ~equal):")
    for c in CONDITIONS:
        print(f"  {c:8s} {np.mean(col('prim', c)):.3f}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
