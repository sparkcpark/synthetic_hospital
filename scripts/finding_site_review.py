"""Grading CSV for the Phase B Finding-site (Class 2) granularity cutoff (open item 2).

Lists every SNOMED `Finding site` shared by >=2 of our diagnoses (the sites that
create Class-2 shared-finding-site edges), with hierarchy context so a clinician
can set the "organ granularity or finer" cutoff:
  - finding_site (SNOMED preferred term) + id
  - parent: immediate is-a parent concept name(s)
  - depth: number of is-a ancestors (system-level concepts are shallow; organ/
    tissue are deeper)
  - n_diagnoses: how many of our diagnoses share this site (broad sites = many)
  - currently_class2: yes if it passes the current count guard (<=30 dx)
  - example_diagnoses
  - GRADE: blank -> mark `organ` (keep as Class 2) or `system` (demote to Class 3)
Sorted broadest-first (highest n_diagnoses) so the system->organ boundary is obvious.
"""
import csv
import sqlite3

from etl.ontology.snomed import SNOMEDDictionary
from etl.stages.s06d_diagnosis_relations import _load_snomed_graph, _load_ancestors

DB = "data/benchmark_v1.2_copy.db"


def main():
    snomed = SNOMEDDictionary()
    _out_typed, isa_parents, finding_sites = _load_snomed_graph()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dx = con.execute("SELECT diagnosis_id, snomed_id, display_name FROM diagnoses "
                     "WHERE snomed_id IS NOT NULL").fetchall()

    site_dx = {}
    for dx_id, snomed_id, name in dx:
        for site in finding_sites.get(str(snomed_id), ()):
            site_dx.setdefault(site, []).append(name)
    shared = {s: names for s, names in site_dx.items() if len(names) >= 2}

    anc = _load_ancestors(set(shared))

    rows = []
    for site, names in shared.items():
        parents = [snomed.get_preferred_term(p) or p for p in sorted(isa_parents.get(site, ()))]
        rows.append({
            "finding_site_id": site,
            "finding_site": snomed.get_preferred_term(site) or "(unknown)",
            "parent": " ; ".join(parents[:2]),
            "depth": len(anc.get(site, ())),
            "n_diagnoses": len(names),
            "currently_class2": "yes" if len(names) <= 30 else "no(>30)",
            "example_diagnoses": " ; ".join(sorted(set(names))[:3]),
            "GRADE_organ_or_system": "",
        })
    # Sort shallowest (most general / system-level) first so the system->organ
    # depth boundary is gradeable top-down. depth = # is-a ancestors.
    rows.sort(key=lambda r: (r["depth"], -r["n_diagnoses"]))

    with open("finding_site_review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote finding_site_review.csv: {len(rows)} shared finding-sites "
          f"({sum(1 for r in rows if r['currently_class2'].startswith('yes'))} currently Class 2)")


if __name__ == "__main__":
    main()
