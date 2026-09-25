"""Miscoding audit: flag diagnoses whose assigned ICD-10 code disagrees, at the
chapter level, with the SNOMED CT -> ICD-10-CM crosswalk for the same concept.

Each diagnosis carries an ICD-10 code AND a SNOMED concept. The SNOMED->ICD map
(the ontology's own crosswalk) says which ICD code(s) a concept should map to.
When the assigned code is in a *different ICD chapter* than every crosswalk code,
the assigned code is almost certainly wrong (e.g. "Sheehan syndrome" coded K00.7
[dental] while its SNOMED concept maps to E23.0 [pituitary]).

Output: miscoding_audit.csv + a summary. Deterministic; no LLM.
"""

import argparse
import csv
import sqlite3
import sys
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

from etl.ontology.snomed import EXTENDED_MAP_FILE  # noqa: E402
from etl.ontology.specialty_map import specialty_for_icd10  # noqa: E402
from eval import semantic_match  # noqa: E402

# Below this display-name vs SNOMED-term similarity, the assigned SNOMED concept
# itself is probably wrong (e.g. "Disorder of sex development" grounded to a
# dental concept). Conservative so custom phrasings aren't false-flagged.
NAME_SIM_MIN = 0.15


def _norm(code: str) -> str:
    return (code or "").replace(".", "").replace(" ", "").strip().upper()


def _chapter(code: str) -> str:
    n = _norm(code)
    return n[:1] if n else ""


def load_snomed_to_icd(path) -> dict[str, set[str]]:
    """snomed concept id -> set of ICD-10-CM codes (active crosswalk rows)."""
    m: dict[str, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("active") != "1":
                continue
            icd = (row.get("mapTarget") or "").strip()
            sid = (row.get("referencedComponentId") or "").strip()
            if icd and sid:
                m[sid].add(icd)
    return m


_FIELDS = ["diagnosis_id", "display_name", "flag_type", "assigned_icd10",
           "assigned_specialty", "snomed_id", "snomed_term", "name_similarity",
           "crosswalk_icd10", "suggested_specialty", "specialty_changes"]


def audit(db: str, out_csv: str) -> dict:
    s2i = load_snomed_to_icd(EXTENDED_MAP_FILE)
    from etl.ontology.snomed import SNOMEDDictionary  # heavy; load once here
    snomed_dict = SNOMEDDictionary()

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT diagnosis_id, icd10_code, snomed_id, display_name FROM diagnoses "
        "WHERE icd10_code IS NOT NULL AND snomed_id IS NOT NULL"
    ).fetchall()

    flags = []
    n_checked = n_unverifiable = 0
    pattern = Counter()
    for dx_id, icd, snomed, name in rows:
        term = snomed_dict.get_preferred_term(str(snomed)) or ""
        assigned_spec = specialty_for_icd10(icd) or "(none)"

        # Pass 1: ICD wrong — assigned chapter disagrees with the SNOMED crosswalk.
        expected = s2i.get(str(snomed))
        if expected:
            n_checked += 1
            exp_chapters = {_chapter(e) for e in expected}
            if _chapter(icd) not in exp_chapters:
                sugg_spec = next((specialty_for_icd10(e) for e in sorted(expected)
                                  if specialty_for_icd10(e)), None) or "(none)"
                pattern[f"{_chapter(icd)}->{'/'.join(sorted(exp_chapters))}"] += 1
                flags.append({
                    "diagnosis_id": dx_id, "display_name": name,
                    "flag_type": "icd_miscode", "assigned_icd10": icd,
                    "assigned_specialty": assigned_spec, "snomed_id": snomed,
                    "snomed_term": term, "name_similarity": "",
                    "crosswalk_icd10": "; ".join(sorted(expected)[:4]),
                    "suggested_specialty": sugg_spec,
                    "specialty_changes": "yes" if assigned_spec != sugg_spec else "no",
                })
        else:
            n_unverifiable += 1

        # Pass 2: SNOMED wrong — the display name doesn't match the assigned
        # concept's term (so its crosswalk/code is suspect too).
        if term:
            sim = semantic_match.similarity(name, term)
            if sim < NAME_SIM_MIN:
                flags.append({
                    "diagnosis_id": dx_id, "display_name": name,
                    "flag_type": "snomed_miscode", "assigned_icd10": icd,
                    "assigned_specialty": assigned_spec, "snomed_id": snomed,
                    "snomed_term": term, "name_similarity": round(sim, 3),
                    "crosswalk_icd10": "(re-ground SNOMED)",
                    "suggested_specialty": "", "specialty_changes": "",
                })

    flags.sort(key=lambda r: (r["flag_type"], r["specialty_changes"] != "yes",
                              r["assigned_icd10"]))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(flags)

    n_icd = sum(1 for f in flags if f["flag_type"] == "icd_miscode")
    n_snomed = sum(1 for f in flags if f["flag_type"] == "snomed_miscode")
    return {
        "checked": n_checked,
        "unverifiable_no_crosswalk": n_unverifiable,
        "icd_miscode_flags": n_icd,
        "icd_specialty_changes": sum(1 for f in flags
                                     if f["flag_type"] == "icd_miscode" and f["specialty_changes"] == "yes"),
        "snomed_miscode_flags": n_snomed,
        "top_patterns": pattern.most_common(10),
    }


def main():
    ap = argparse.ArgumentParser(description="Audit ICD-10 miscoding via SNOMED crosswalk")
    ap.add_argument("--db", default="data/benchmark_v1.2.db")
    ap.add_argument("--out", default="miscoding_audit.csv")
    args = ap.parse_args()
    res = audit(args.db, args.out)
    print(f"checked (have crosswalk): {res['checked']} | "
          f"unverifiable: {res['unverifiable_no_crosswalk']}")
    print(f"ICD-miscode flags (wrong ICD vs SNOMED crosswalk): {res['icd_miscode_flags']} "
          f"(specialty changes: {res['icd_specialty_changes']})")
    print(f"SNOMED-miscode flags (name != assigned concept): {res['snomed_miscode_flags']}")
    print("top ICD assigned->expected chapter patterns:")
    for pat, n in res["top_patterns"]:
        print(f"  {pat:18s} {n}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
