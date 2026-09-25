"""Tier-driver attribution: per (specialty, diagnosis, tier) — does a diagnosis drive
a specialty's PRIMARY tier (it's home to S) or its RELEVANT tier (graph-linked to an
S-home present dx), how many patients, and how many distinct findings it contributes.
Plus n_specialties (multi-home) and home_provenance (chapter-default vs curated-override).

Reflects the CURRENT specialty map + the typed diagnosis_relations graph. Lets you read
off, e.g., how many stroke patients convert from Cardiology-primary to Cardiology-relevant.
"""
import csv
import json
from collections import defaultdict

import sqlite3

from etl.ontology.specialty_map import specialties_for_icd10, home_provenance

DB = "data/benchmark_v1.2_copy.db"
TAU_RESIDUAL = 0.50


def _parse_qids(raw):
    try:
        v = json.loads(raw) if raw else []
        return [int(q) for q in v] if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dx_info = {d: (icd, nm) for d, icd, nm in
               con.execute("SELECT diagnosis_id, icd10_code, display_name FROM diagnoses")}
    dx_specs = {d: tuple(specialties_for_icd10(icd)) for d, (icd, _) in dx_info.items()}
    q_correct = defaultdict(list)
    for qid, dx in con.execute("SELECT question_id, diagnosis_id FROM question_diagnoses WHERE role='correct'"):
        q_correct[qid].append(dx)
    q_findings = defaultdict(list)
    for qid, fid, name, ftype in con.execute(
            "SELECT qf.question_id, qf.finding_id, cf.display_name, cf.finding_type "
            "FROM question_findings qf JOIN clinical_findings cf ON qf.finding_id=cf.finding_id "
            "WHERE qf.relevance='key'"):
        q_findings[qid].append((fid, name, ftype))
    df_dx = defaultdict(set)
    for fid, dx in con.execute("SELECT finding_id, diagnosis_id FROM diagnosis_findings"):
        df_dx[fid].add(dx)
    typed_adj, residual_adj = defaultdict(set), defaultdict(set)
    for a, b, rc, ov in con.execute("SELECT dx_a, dx_b, relation_class, overlap_score FROM diagnosis_relations"):
        if rc in ("1_definitional", "2_associative"):
            typed_adj[a].add(b); typed_adj[b].add(a)
        elif rc == "3_residual" and ov is not None and ov >= TAU_RESIDUAL:
            residual_adj[a].add(b); residual_adj[b].add(a)
    enc_qids = defaultdict(list)
    for pid, sq in con.execute("SELECT patient_id, source_question_ids FROM longitudinal_encounters "
                               "WHERE source_question_ids IS NOT NULL"):
        enc_qids[pid].extend(_parse_qids(sq))

    # (specialty, dx, tier) -> (patients set, findings set)
    agg = defaultdict(lambda: (set(), set()))
    for pid, qids in enc_qids.items():
        qids = set(qids)
        present = set()
        for qid in qids:
            present.update(q_correct.get(qid, []))
        if not present:
            continue
        finding_owners = defaultdict(set)
        for qid in qids:
            q_owners = set(q_correct.get(qid, [])) & present
            for fid, name, ftype in q_findings.get(qid, []):
                if ftype == "demographic":
                    continue
                finding_owners[name] |= (df_dx.get(fid, set()) & present) or q_owners
        home_by_spec = defaultdict(set)
        for o in present:
            for s in dx_specs.get(o, ()):
                home_by_spec[s].add(o)
        related = {o: (typed_adj.get(o, set()) | residual_adj.get(o, set())) & present for o in present}

        for S, s_home in home_by_spec.items():
            for name, owners in finding_owners.items():
                prim = owners & s_home
                if prim:
                    for o in prim:
                        agg[(S, o, "primary")][0].add(pid); agg[(S, o, "primary")][1].add(name)
                    continue
                rel = {o for o in owners if related[o] & s_home}
                for o in rel:
                    agg[(S, o, "relevant")][0].add(pid); agg[(S, o, "relevant")][1].add(name)

    rows = []
    for (S, dx, tier), (pats, finds) in agg.items():
        icd, nm = dx_info.get(dx, ("", ""))
        rows.append({"specialty": S, "tier": tier, "icd10": icd, "diagnosis": nm,
                     "n_patients": len(pats), "n_findings_contributed": len(finds),
                     "n_specialties": len(dx_specs.get(dx, ())),
                     "home_provenance": home_provenance(icd), "diagnosis_id": dx})
    rows.sort(key=lambda r: (r["specialty"], r["tier"], -r["n_patients"]))

    with open("involved_tier_drivers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["specialty", "tier", "icd10", "diagnosis",
                           "n_patients", "n_findings_contributed", "n_specialties",
                           "home_provenance", "diagnosis_id"])
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    t = Counter((r["specialty"], r["tier"]) for r in rows)
    print(f"wrote involved_tier_drivers.csv ({len(rows)} (specialty,dx,tier) rows)")
    print(f"primary driver-rows: {sum(1 for r in rows if r['tier']=='primary')} | "
          f"relevant driver-rows: {sum(1 for r in rows if r['tier']=='relevant')} | "
          f"curated-override drivers: {sum(1 for r in rows if r['home_provenance']=='curated-override')}")
    # quantify the stroke conversion: cerebrovascular (I6x) as Neurology-primary vs Cardiology-relevant
    strokep = sum(r["n_patients"] for r in rows if r["specialty"] == "Neurology" and r["tier"] == "primary"
                  and r["icd10"][:2] == "I6")
    stroker = sum(r["n_patients"] for r in rows if r["specialty"] == "Cardiology" and r["tier"] == "relevant"
                  and r["icd10"][:2] == "I6")
    print(f"stroke (I6x): Neurology-PRIMARY pt-inst={strokep} | Cardiology-RELEVANT pt-inst={stroker} "
          f"(the primary->relevant conversion the split buys)")


if __name__ == "__main__":
    main()
