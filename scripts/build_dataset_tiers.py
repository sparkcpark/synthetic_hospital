"""Assign patient-aligned small/medium/large tiers to the specialty-conditioned GT and
write a reproducibility manifest.

Per the paper convention (spec §, lines 3163/3586): tiers are PATIENT-based and nested.
  small  = the 200 existing val ("dev") patients          -> split='val'
  large  = all 1,268 patients (full release)              -> split='val'+'test'
  medium = small + a deterministic subset of test patients (the benchmark's forthcoming
           medium cohort; recorded here so the specialty task uses the SAME patients).
This replaces the earlier over-sampled Phase G split (no train/val/test needed — nothing
is trained; reproducibility is by frozen patient lists + the deterministic Phase C config).

Writes: specialty_dataset_manifest.json (+ _SUMMARY.md). Run on the copy, then migrate.
"""
import csv
import json
import sqlite3
from collections import Counter, defaultdict

DB = "data/benchmark_v1.2_copy.db"
MEDIUM_TEST_PATIENTS = 300   # test patients added to the 200 small -> 500-patient medium cohort

# Phase C config that produced the tiers (frozen for reproducibility).
PHASE_C_CONFIG = {
    "specialty_map": "etl/ontology/specialty_map.py (chapter ranges + CODE_OVERRIDES, clinician-reviewed audit)",
    "relatedness_graph": "etl/stages/s06d_diagnosis_relations.py (typed-edge rule)",
    "relevant_tier (CREDITED — recall target)": "Class-1 definitional OR source='curated' edges (definitional OR adjudicated = credit)",
    "neutral_tier": "all non-curated Class-2 associative + all stored Class-3 residual. No tau gate: "
                    "the builder's SHARED_FINDING_STORE_MIN=0.20 is the residual admission boundary, so no "
                    "per-run residual calibration affects the headline.",
    "excluded_tier (LEAK target)": "no relatedness link of any class",
    "leak_protection": "leakage penalizes the EXCLUDED tier only; relevant AND neutral are leak-protected. "
                       "So TIERED-relevant (leak-protected = Class-1/2/3 + curated) STRICTLY CONTAINS "
                       "CREDITED-relevant (recall target = Class-1 + curated) — the scorer credits more "
                       "conservatively than phaseB_typed_edge_relevance_rule.md tiers. leakage_rate_strict "
                       "(excluded ∪ neutral) reports the boundary sensitivity.",
    "verbosity": "headline governs FOCUS (off-target exclusion, via leakage) NOT CONCISENESS — the neutral "
                 "tier is a free no-credit/no-penalty band, so within-linked padding is unpenalized; "
                 "mean_summary_words is reported as a length descriptor, not a headline term.",
    "importance": "critical = finding is pathognomonic/highly_suggestive for its owning dx (diagnosis_findings.relationship)",
    "finding_attribution": "diagnosis_findings (present-restricted) with question-correct fallback; demographics excluded",
    "involvement": "involved = >=1 present diagnosis home to the specialty; else absent (abstention control)",
}


def main():
    con = sqlite3.connect(DB)
    # small = the 200 dev patients = the existing val split (defined in POSTGRES, not the copy)
    import psycopg
    from epic_sim.app.config import settings
    pg = psycopg.connect(settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://"))
    small = sorted(p for (p,) in pg.execute(
        "SELECT DISTINCT patient_id FROM benchmark_ground_truth WHERE task='context_summarization' "
        "AND granularity='patient' AND split='val' "
        "AND COALESCE(ground_truth->>'variant','unconditioned')='unconditioned'"))
    pg.close()
    all_pts = sorted(p for (p,) in con.execute("SELECT DISTINCT patient_id FROM longitudinal_patients"))
    test_pts = [p for p in all_pts if p not in set(small)]
    medium = sorted(set(small) | set(test_pts[:MEDIUM_TEST_PATIENTS]))

    # re-assign split on specialty rows: val for the small (dev) patients, test for the rest
    small_set = set(small)
    rows = con.execute(
        "SELECT gt_id, patient_id, ground_truth FROM benchmark_ground_truth WHERE task='context_summarization' "
        "AND granularity='patient' AND json_extract(ground_truth,'$.variant')='specialty_conditioned'").fetchall()
    tier_rowcount = defaultdict(lambda: defaultdict(int))
    keys = []
    for gt_id, pid, gtj in rows:
        g = json.loads(gtj)
        split = "val" if pid in small_set else "test"
        con.execute("UPDATE benchmark_ground_truth SET split=? WHERE gt_id=?", (split, gt_id))
        keys.append((pid, g["specialty"], g["involvement"], g.get("eval_stratum", "")))
        # per-tier involvement counts (nested)
        for tier, members in (("small", small_set), ("medium", set(medium)), ("large", set(all_pts))):
            if pid in members:
                tier_rowcount[tier][g["involvement"]] += 1
    con.commit()

    manifest = {
        "dataset": "specialty_conditioned_summarization",
        "row_key": "(patient_id, specialty, involvement) — stable across regeneration (gt_id is not)",
        "tiers": {
            "small": {"n_patients": len(small), "patient_ids": small,
                      "rows": dict(tier_rowcount["small"]), "db_split": "val"},
            "medium": {"n_patients": len(medium), "patient_ids": medium,
                       "rows": dict(tier_rowcount["medium"]),
                       "note": "small + first %d test patients by id; align to the benchmark medium cohort when frozen" % MEDIUM_TEST_PATIENTS},
            "large": {"n_patients": len(all_pts), "patient_ids": "ALL (1268)",
                      "rows": dict(tier_rowcount["large"]), "db_split": "val+test"},
        },
        "phase_c_config": PHASE_C_CONFIG,
        "reproduce": "re-run etl.stages.s10f_specialty on the same diagnoses/findings/graph + this config; "
                     "tiers are deterministic, so the (patient,specialty,involvement) rows reproduce exactly.",
    }
    json.dump(manifest, open("specialty_dataset_manifest.json", "w"), indent=2)
    # row-key table (the frozen membership, for byte-stable reproduction)
    with open("specialty_dataset_rowkeys.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["patient_id", "specialty", "involvement", "eval_stratum", "tier"])
        for pid, spec, inv, strat in sorted(keys):
            tier = "small" if pid in small_set else ("medium" if pid in set(medium) else "large")
            w.writerow([pid, spec, inv, strat, tier])

    print("tiers (patient-aligned, nested):")
    for t in ("small", "medium", "large"):
        info = manifest["tiers"][t]
        print(f"  {t:7s} {info['n_patients']:>5} patients | rows {info['rows']} = {sum(info['rows'].values())}")
    print("wrote specialty_dataset_manifest.json, specialty_dataset_rowkeys.csv")


if __name__ == "__main__":
    main()
