"""DECIDE-group (C3/sparse) overlap analysis.

For each of the 8 intentionally-uncurated comorbidity rows, compute the actual
shared-finding (Class-3) weighted-Jaccard overlap between its A and B diagnoses,
using the builder's exact weighting (_DF_REL_WEIGHT) and non-specific-finding gate
(SHARED_FINDING_MAX_DF). This tells us, per pair, the tau_residual at/below which it
would be caught as residual overlap — versus pairs that share ~0 findings and can
only be documented as accepted gaps.
"""
import re
import sqlite3
from collections import defaultdict

from scripts.comorbidity_coverage import _resolve
from etl.stages.s06d_diagnosis_relations import _DF_REL_WEIGHT, SHARED_FINDING_MAX_DF

DB = "data/benchmark_v1.2_copy.db"
MD = "comorbidity_coverage_list.md"
DECIDE = {26, 49, 71, 79, 96, 101, 103, 117}


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dx_rows = con.execute("SELECT diagnosis_id, REPLACE(UPPER(icd10_code),'.',''), "
                          "LOWER(display_name), icd10_code FROM diagnoses").fetchall()
    icd_cache, kw_cache = {}, {}

    def by_icd(prefix):
        p = prefix.replace(".", "").upper()
        if p not in icd_cache:
            icd_cache[p] = {d for d, code, _, _ in dx_rows if code and code.startswith(p)}
        return icd_cache[p]

    def by_kw(kw):
        if kw not in kw_cache:
            kw_cache[kw] = {d for d, _, name, _ in dx_rows if kw in name}
        return kw_cache[kw]

    name_of = {d: (raw or "?", name) for d, _, name, raw in dx_rows}

    # diagnosis -> {finding: weight}; finding -> #diagnoses (for the non-specific gate)
    dx_find = defaultdict(dict)
    finding_n = defaultdict(int)
    finding_seen = defaultdict(set)
    for dx_id, fid, rel in con.execute(
            "SELECT diagnosis_id, finding_id, relationship FROM diagnosis_findings"):
        w = _DF_REL_WEIGHT.get(rel, 0.3)
        if w > dx_find[dx_id].get(fid, 0.0):
            dx_find[dx_id][fid] = w
        finding_seen[fid].add(dx_id)
    finding_n = {f: len(s) for f, s in finding_seen.items()}
    dx_total = {d: sum(fs.values()) for d, fs in dx_find.items()}

    def overlap(a, b):
        shared = 0.0
        n = 0
        for fid in set(dx_find.get(a, {})) & set(dx_find.get(b, {})):
            if finding_n.get(fid, 0) > SHARED_FINDING_MAX_DF:   # non-specific -> gated out
                continue
            shared += min(dx_find[a][fid], dx_find[b][fid])
            n += 1
        denom = dx_total.get(a, 0) + dx_total.get(b, 0) - shared
        return (shared / denom if denom > 0 else 0.0), n

    # pull the DECIDE rows' pair text
    pairs = {}
    for line in open(MD):
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if m and int(m.group(1)) in DECIDE:
            cells = [c.strip() for c in line.split("|")]
            pairs[int(cells[1])] = cells[2]

    print(f"{'row':>4} {'maxJac':>7} {'#shf':>4}  pair  ->  best (a | b)")
    results = []
    for row in sorted(DECIDE):
        pair = pairs[row]
        segs = [s for s in re.split(r"↔|->", pair) if s.strip()]
        A = _resolve(segs[0], by_icd, by_kw)
        B = _resolve(segs[-1], by_icd, by_kw)
        best, best_pair, best_n = 0.0, None, 0
        for a in A:
            for b in B:
                if a == b:
                    continue
                jac, n = overlap(a, b)
                if jac > best:
                    best, best_pair, best_n = jac, (a, b), n
        results.append((row, best, best_n))
        bp = ""
        if best_pair:
            a, b = best_pair
            bp = f"{name_of[a][0]} {name_of[a][1][:20]} | {name_of[b][0]} {name_of[b][1][:20]}"
        print(f"{row:>4} {best:>7.3f} {best_n:>4}  {pair[:46]:46s}  {bp}")

    print("\n=== verdict (store floor / tau_residual currently 0.20 / 0.50) ===")
    for thr in (0.20, 0.10, 0.05):
        caught = [r for r, b, _ in results if b >= thr]
        print(f"  tau_residual <= {thr:.2f}  would catch rows: {sorted(caught)}")
    negligible = [r for r, b, _ in results if b < 0.05]
    print(f"\n  share ~0 findings (<0.05) -> DOCUMENT as accepted gaps: {sorted(negligible)}")


if __name__ == "__main__":
    main()
