"""Write icd_corrections_conflicts_NOTE.md documenting the 88 blocked duplicates."""
import csv
import sqlite3

con = sqlite3.connect("file:data/benchmark_v1.2_copy.db?mode=ro", uri=True)
conf = list(csv.DictReader(open("icd_corrections_conflicts.csv")))

intro = [
    "# ICD-10 correction conflicts - 88 duplicate diagnoses (kept as-is)",
    "",
    "Of 352 curated ICD-10 corrections (miscoding_audit.csv), **264 were applied** to",
    "benchmark_v1.2_copy.db. The remaining **88 were BLOCKED** by the",
    "UNIQUE(icd10_code, snomed_id) constraint: applying them would collapse the diagnosis",
    "onto the same concept+code as another existing diagnosis row -- i.e. these 88 are",
    "**duplicate diagnoses**.",
    "",
    "**Decision (2026-06-07):** left unchanged (kept their original ICD-10 codes) and flagged",
    "here rather than merged. The full machine-readable list is icd_corrections_conflicts.csv.",
    "",
    "| blocked dx | display | old | intended | twin dx | twin display |",
    "|---|---|---|---|---|---|",
]
for r in conf:
    dx = int(r["diagnosis_id"]); target = r["new_icd10"]
    sn = con.execute("SELECT snomed_id FROM diagnoses WHERE diagnosis_id=?", (dx,)).fetchone()
    twin = None
    if sn:
        twin = con.execute(
            "SELECT diagnosis_id, display_name FROM diagnoses "
            "WHERE icd10_code=? AND snomed_id=? AND diagnosis_id!=?",
            (target, sn[0], dx)).fetchone()
    td, tn = (twin[0], twin[1]) if twin else ("(unresolved)", "")
    intro.append(f"| {dx} | {r['display_name'][:36]} | {r['old_icd10']} | {target} | {td} | {tn[:36]} |")

open("icd_corrections_conflicts_NOTE.md", "w").write("\n".join(intro) + "\n")
print(f"wrote icd_corrections_conflicts_NOTE.md ({len(conf)} blocked duplicates)")
