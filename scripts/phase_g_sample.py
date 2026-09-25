"""Phase G: stratified eval sample for specialty-conditioned summarization.

Selects (patient, specialty) GT rows into the eval set and assigns a split,
over-sampling the rare cross-specialty (relevant>0) cases so that metric keeps
its power. Strata:
  - involved_relevant : involved row with >=1 relevant finding  -> take ALL
  - involved_primary  : involved row with no relevant finding   -> sample N_PRIMARY
  - absent            : abstention negative control             -> sample N_ABSENT

Selection is deterministic (sorted by patient_id, specialty). Split is assigned
by patient (all of a patient's rows share a split) to prevent patient leakage:
~25% val / ~75% test via patient_id % 4. The chosen stratum is written into each
row's ground_truth as `eval_stratum` so analysis can weight/report per stratum
(the over-sample biases naive aggregates). Non-selected rows get split=NULL.
"""
import argparse
import json
import sqlite3
from collections import defaultdict

N_PRIMARY = 400
N_ABSENT = 300


def run(db, n_primary=N_PRIMARY, n_absent=N_ABSENT):
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT gt_id, patient_id, ground_truth FROM benchmark_ground_truth "
        "WHERE task='context_summarization' AND granularity='patient' "
        "AND json_extract(ground_truth,'$.variant')='specialty_conditioned'").fetchall()

    by_stratum = defaultdict(list)
    for gt_id, pid, gtj in rows:
        g = json.loads(gtj)
        if g["involvement"] == "involved":
            stratum = "involved_relevant" if g["tiers"]["relevant"] else "involved_primary"
        else:
            stratum = "absent"
        by_stratum[stratum].append((gt_id, pid, g))
    for s in by_stratum:
        by_stratum[s].sort(key=lambda r: (r[1], r[2].get("specialty", "")))

    selected = (by_stratum["involved_relevant"]
                + by_stratum["involved_primary"][:n_primary]
                + by_stratum["absent"][:n_absent])

    # reset prior assignment, then assign split + stratum to the selected rows
    con.execute("UPDATE benchmark_ground_truth SET split=NULL WHERE task='context_summarization' "
                "AND granularity='patient' AND json_extract(ground_truth,'$.variant')='specialty_conditioned'")
    counts = defaultdict(lambda: defaultdict(int))
    for gt_id, pid, g in selected:
        split = "val" if pid % 4 == 0 else "test"
        g["eval_stratum"] = stratum_of(g)
        con.execute("UPDATE benchmark_ground_truth SET split=?, ground_truth=? WHERE gt_id=?",
                    (split, json.dumps(g, ensure_ascii=False), gt_id))
        counts[g["eval_stratum"]][split] += 1
    con.commit()

    print(f"population: " + ", ".join(f"{s}={len(v)}" for s, v in by_stratum.items()))
    print(f"sampled {len(selected)} rows (involved_relevant ALL={len(by_stratum['involved_relevant'])}, "
          f"involved_primary={min(n_primary, len(by_stratum['involved_primary']))}, "
          f"absent={min(n_absent, len(by_stratum['absent']))}):")
    for s in ("involved_relevant", "involved_primary", "absent"):
        v = counts[s]
        print(f"  {s:18s} val={v['val']:>4} test={v['test']:>4} total={v['val']+v['test']:>4}")
    tot = {sp: sum(counts[s][sp] for s in counts) for sp in ("val", "test")}
    print(f"  {'TOTAL':18s} val={tot['val']:>4} test={tot['test']:>4}")


def stratum_of(g):
    if g["involvement"] == "involved":
        return "involved_relevant" if g["tiers"]["relevant"] else "involved_primary"
    return "absent"


def main():
    ap = argparse.ArgumentParser(description="Phase G: specialty-conditioned eval sample")
    ap.add_argument("--db", default="data/benchmark_v1.2_copy.db")
    ap.add_argument("--n-primary", type=int, default=N_PRIMARY)
    ap.add_argument("--n-absent", type=int, default=N_ABSENT)
    args = ap.parse_args()
    run(args.db, args.n_primary, args.n_absent)


if __name__ == "__main__":
    main()
