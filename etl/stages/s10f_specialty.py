"""Phase C: tiered (finding, specialty) labeling (typed-edge rule, 2026-06-07).

Uses the locked specialty map (Phase A) + the TYPED diagnosis_relations graph
(Phase B, phaseB_typed_edge_relevance_rule.md). Tier rule for a finding F (owned
by present diagnoses) and specialty S:

  primary  : an owning diagnosis is home to S (Phase A map).
  relevant : an owning diagnosis dx_A is connected to a PRESENT diagnosis dx_B
             that is home to S, by a Class 1/2 typed edge, OR a Class 3 edge
             whose overlap >= tau_residual.
  excluded : neither.

Because relevance is anchored to a PRESENT home diagnosis, involvement is binary:
a specialty is INVOLVED (>=1 present home diagnosis) or ABSENT (none -> the
abstention negative-control case). Class 1/2 are admitted unconditionally;
tau_residual applies ONLY to the untyped Class 3 residual. Deterministic; no LLM.
"""

import argparse
import json
import sqlite3
from collections import defaultdict

from etl.ontology.specialty_map import SPECIALTIES, specialties_for_icd10

TAU_RESIDUAL = 0.50        # Class-3-only threshold; conservative (prefer under-inclusion)
MAX_TIER = 20
EXCLUDED_SAMPLE = 15
MAX_ABSENT = 2
ALL_SPECIALTIES = set(SPECIALTIES)


def _parse_qids(raw):
    try:
        v = json.loads(raw) if raw else []
        return [int(q) for q in v] if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _build_specialty_tiers(present_dx, finding_owners, dx_specs, gold_adj, neutral_adj,
                           all_specialties=ALL_SPECIALTIES):
    """Pure tiering with a NEUTRAL buffer to de-circularize the leak boundary.
    gold_adj = Class-1 definitional + clinician-curated edges (credited as 'relevant',
    and leak-protected); neutral_adj = Class-2/3 associative/residual edges (NEITHER
    credited nor a leak). Tier for a finding owned by present dx, specialty S:
      primary  : an owner is home to S
      relevant : an owner is gold-linked to a present S-home dx
      neutral  : an owner is neutral-linked to a present S-home dx (not relevant)
      excluded : no link to S at all (a leak if the summary mentions it)."""
    home_dx_by_spec = defaultdict(set)
    for o in present_dx:
        for s in dx_specs.get(o, ()):
            home_dx_by_spec[s].add(o)
    involved = set(home_dx_by_spec) & all_specialties

    gold_rel = {o: gold_adj.get(o, set()) & present_dx for o in present_dx}
    neut_rel = {o: neutral_adj.get(o, set()) & present_dx for o in present_dx}

    all_names = list(finding_owners.keys())
    out = []
    for S in sorted(involved):
        s_home = home_dx_by_spec[S]
        primary, relevant, neutral, excluded = [], [], [], []
        for name, owners in finding_owners.items():
            if any(o in s_home for o in owners):
                primary.append(name)
            elif any(gold_rel[o] & s_home for o in owners):
                relevant.append(name)
            elif any(neut_rel[o] & s_home for o in owners):
                neutral.append(name)
            else:
                excluded.append(name)
        out.append({"specialty": S, "involvement": "involved",
                    "primary": primary[:MAX_TIER], "relevant": relevant[:MAX_TIER],
                    "neutral": neutral[:MAX_TIER], "excluded_sample": excluded[:EXCLUDED_SAMPLE]})
    for S in sorted(all_specialties - involved)[:MAX_ABSENT]:
        out.append({"specialty": S, "involvement": "absent", "primary": [], "relevant": [],
                    "neutral": [], "excluded_sample": all_names[:EXCLUDED_SAMPLE]})
    return out


def run_specialty(conn, tau_residual=TAU_RESIDUAL, pilot=None):
    dx_specs = {dx: tuple(specialties_for_icd10(icd))
                for dx, icd in conn.execute("SELECT diagnosis_id, icd10_code FROM diagnoses")}
    q_correct = defaultdict(list)
    for qid, dx in conn.execute(
            "SELECT question_id, diagnosis_id FROM question_diagnoses WHERE role='correct'"):
        q_correct[qid].append(dx)
    q_findings = defaultdict(list)
    for qid, fid, name, ftype in conn.execute("""
            SELECT qf.question_id, qf.finding_id, cf.display_name, cf.finding_type
            FROM question_findings qf JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
            WHERE qf.relevance='key'"""):
        q_findings[qid].append((fid, name, ftype))
    df_dx = defaultdict(set)
    df_rel = {}   # (finding_id, diagnosis_id) -> relationship (importance signal)
    for fid, dx, rel in conn.execute(
            "SELECT finding_id, diagnosis_id, relationship FROM diagnosis_findings"):
        df_dx[fid].add(dx)
        df_rel[(fid, dx)] = rel

    # GOLD = Class-1 definitional + clinician-curated -> credited 'relevant' (leak-protected).
    # NEUTRAL = ALL non-curated Class-2 associative + ALL stored Class-3 residual -> neutral
    # buffer (neither credited nor a leak). NB: tau_residual no longer gates tiering — the
    # builder's shared-finding store floor (SHARED_FINDING_STORE_MIN=0.20) is the residual
    # admission boundary, so no per-run residual calibration affects the headline.
    gold_adj = defaultdict(set)
    neutral_adj = defaultdict(set)
    for a, b, rclass, source, overlap in conn.execute(
            "SELECT dx_a, dx_b, relation_class, source, overlap_score FROM diagnosis_relations"):
        if rclass == "1_definitional" or source == "curated":
            gold_adj[a].add(b); gold_adj[b].add(a)
        elif rclass in ("2_associative", "3_residual"):
            neutral_adj[a].add(b); neutral_adj[b].add(a)

    patients = [p[0] for p in conn.execute(
        "SELECT patient_id FROM longitudinal_patients ORDER BY patient_id")]
    if pilot:
        patients = patients[:pilot]
    enc_qids = defaultdict(list)
    for pid, sq in conn.execute(
            "SELECT patient_id, source_question_ids FROM longitudinal_encounters "
            "WHERE source_question_ids IS NOT NULL"):
        enc_qids[pid].extend(_parse_qids(sq))

    conn.execute(
        "DELETE FROM benchmark_ground_truth WHERE task='context_summarization' "
        "AND granularity='patient' AND json_extract(ground_truth,'$.variant')='specialty_conditioned'")

    inserted = 0
    inv_counts = defaultdict(int)
    for pid in patients:
        qids = set(enc_qids.get(pid, []))
        present_dx = set()
        for qid in qids:
            present_dx.update(q_correct.get(qid, []))
        if not present_dx:
            continue
        finding_owners = defaultdict(set)
        finding_imp = {}   # finding name -> 'critical' (pathognomonic/highly_suggestive) | 'optional'
        for qid in qids:
            q_owners = set(q_correct.get(qid, [])) & present_dx
            for fid, name, ftype in q_findings.get(qid, []):
                if ftype == "demographic":
                    continue
                df_owners = df_dx.get(fid, set()) & present_dx
                finding_owners[name] |= df_owners or q_owners
                imp = finding_imp.get(name, "optional")
                for d in df_owners:
                    if df_rel.get((fid, d)) in ("pathognomonic", "highly_suggestive"):
                        imp = "critical"
                finding_imp[name] = imp

        for t in _build_specialty_tiers(present_dx, finding_owners, dx_specs, gold_adj, neutral_adj):
            inv_counts[t["involvement"]] += 1
            label = t["specialty"].replace("_", "/")
            gt = {
                "variant": "specialty_conditioned",
                "specialty": t["specialty"],
                "involvement": t["involvement"],
                "clinical_question": (
                    f"Summarize this patient's chart from a {label} perspective: the active "
                    f"{label} problems and the comorbidities, labs, and medications relevant "
                    f"to {label} care."),
                "reference_summary": "",
                "tiers": {
                    "primary": [{"display_name": n, "importance": finding_imp.get(n, "optional")}
                                for n in t["primary"]],
                    "relevant": [{"display_name": n, "importance": finding_imp.get(n, "optional")}
                                 for n in t["relevant"]],
                    "neutral": [{"display_name": n} for n in t["neutral"]],
                    "excluded_sample": [{"display_name": n} for n in t["excluded_sample"]],
                },
                "tau_residual": tau_residual,
            }
            difficulty = ("hard" if t["involvement"] == "involved" and len(t["primary"]) >= 5
                          else "medium" if t["involvement"] == "involved" else "easy")
            conn.execute(
                "INSERT INTO benchmark_ground_truth "
                "(task, granularity, patient_id, ground_truth, difficulty) VALUES (?,?,?,?,?)",
                ("context_summarization", "patient", pid, json.dumps(gt, ensure_ascii=False), difficulty))
            inserted += 1
    conn.commit()
    return {"rows": inserted, "patients": len(patients), "by_involvement": dict(inv_counts)}


def main():
    ap = argparse.ArgumentParser(description="Phase C: specialty-conditioned tiered labeling")
    ap.add_argument("--db", default="data/benchmark_v1.2_copy.db")
    ap.add_argument("--tau-residual", type=float, default=TAU_RESIDUAL)
    ap.add_argument("--pilot", type=int, default=None)
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    res = run_specialty(conn, tau_residual=args.tau_residual, pilot=args.pilot)
    print(f"rows: {res['rows']} across {res['patients']} patients (tau_residual={args.tau_residual})")
    print(f"by involvement: {res['by_involvement']}")


if __name__ == "__main__":
    main()
