"""Dump a random sample of N specialty-conditioned small-set rows with Kimi + GLM summaries
side by side (plus the GT tiers) for qualitative inspection. Seeded for reproducibility."""
import csv
import json
import random

import psycopg
from epic_sim.app.config import settings

KIMI_RUN, GLM_RUN, N, SEED = 291, 293, 20, 42
OUT = "sample_summaries_kimi_glm.csv"


def _asdict(v):
    return v if isinstance(v, dict) else json.loads(v)


def _summary(pred):
    p = _asdict(pred)
    return (p.get("summary") or "").strip() if isinstance(p, dict) else str(pred)


def _findings(tier):
    return "; ".join(f.get("display_name", "") for f in tier)


def main():
    c = psycopg.connect(settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://"))
    rows = c.execute(
        "SELECT gt_id, patient_id, ground_truth FROM benchmark_ground_truth "
        "WHERE task='context_summarization' AND granularity='patient' "
        "AND ground_truth->>'variant'='specialty_conditioned' AND split='val'").fetchall()
    random.seed(SEED)
    sample = sorted(random.sample(rows, N), key=lambda r: (_asdict(r[2])["involvement"], r[0]))
    ids = [r[0] for r in sample]

    def preds(run):
        return {gt: _summary(p) for gt, p in c.execute(
            "SELECT gt_id, prediction FROM evaluation_predictions WHERE run_id=%s AND gt_id = ANY(%s)",
            (run, ids))}

    kimi, glm = preds(KIMI_RUN), preds(GLM_RUN)
    n_inv = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gt_id", "patient_id", "specialty", "involvement",
                    "primary_findings", "relevant_findings", "excluded_sample(8)",
                    "kimi_summary", "glm_summary"])
        for gt_id, pid, gtj in sample:
            g = _asdict(gtj)
            t = g["tiers"]
            n_inv += g["involvement"] == "involved"
            w.writerow([gt_id, pid, g["specialty"], g["involvement"],
                        _findings(t["primary"]), _findings(t["relevant"]),
                        _findings(t["excluded_sample"][:8]),
                        kimi.get(gt_id, "<no prediction>"), glm.get(gt_id, "<no prediction>")])
    print(f"wrote {OUT}: {N} rows ({n_inv} involved / {N - n_inv} absent), "
          f"kimi={len(kimi)} glm={len(glm)} predictions matched")


if __name__ == "__main__":
    main()
