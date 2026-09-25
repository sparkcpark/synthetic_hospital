"""Apply the curated ICD-10 corrections from miscoding_audit.csv to the diagnoses
table. Target = the SNOMED crosswalk's primary code (first non-placeholder).
Writes a before/after report (which is also the revert map). Copy DB by default.
"""
import argparse
import csv
import sqlite3

from etl.ontology.specialty_map import specialties_for_icd10


def _target(crosswalk: str) -> str:
    codes = [c.strip().replace("?", "") for c in crosswalk.split(";") if c.strip()]
    clean = [c for c in codes if "?" not in c]  # already stripped, kept for clarity
    return (clean or codes)[0] if codes else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="miscoding_audit.csv")
    ap.add_argument("--db", default="data/benchmark_v1.2_copy.db")
    ap.add_argument("--report", default="icd_corrections_applied.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.audit)) if r["flag_type"] == "icd_miscode"]
    con = sqlite3.connect(args.db)
    report = []
    conflicts = []
    to_none = changed_spec = 0
    for r in rows:
        dx = int(r["diagnosis_id"])
        new = _target(r["crosswalk_icd10"])
        if not new:
            continue
        cur = con.execute("SELECT icd10_code FROM diagnoses WHERE diagnosis_id=?", (dx,)).fetchone()
        if not cur:
            continue
        old = cur[0]
        old_spec = " | ".join(specialties_for_icd10(old)) or "(none)"
        new_spec = " | ".join(specialties_for_icd10(new)) or "(none)"
        try:
            if not args.dry_run:
                con.execute("UPDATE diagnoses SET icd10_code=? WHERE diagnosis_id=?", (new, dx))
        except sqlite3.IntegrityError:
            # New (icd10, snomed) collides with another diagnosis row (duplicate).
            conflicts.append({"diagnosis_id": dx, "display_name": r["display_name"],
                              "old_icd10": old, "new_icd10": new,
                              "reason": "UNIQUE(icd10,snomed) collision (duplicate diagnosis)"})
            continue
        if new_spec != old_spec:
            changed_spec += 1
        if new_spec == "(none)":
            to_none += 1
        report.append({"diagnosis_id": dx, "display_name": r["display_name"],
                       "old_icd10": old, "new_icd10": new,
                       "old_specialty": old_spec, "new_specialty": new_spec})
    if not args.dry_run:
        con.commit()
    if conflicts:
        with open("icd_corrections_conflicts.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(conflicts[0].keys()))
            w.writeheader()
            w.writerows(conflicts)

    with open(args.report, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["diagnosis_id", "display_name", "old_icd10",
                                           "new_icd10", "old_specialty", "new_specialty"])
        w.writeheader()
        w.writerows(report)

    print(f"{'DRY-RUN: would apply' if args.dry_run else 'Applied'} {len(report)} corrections to {args.db}")
    print(f"  specialty changed: {changed_spec} | now -> (none/symptom/status): {to_none}")
    print(f"  conflicts skipped (duplicate-creating): {len(conflicts)}")
    print(f"  report (also the revert map): {args.report}")


if __name__ == "__main__":
    main()
