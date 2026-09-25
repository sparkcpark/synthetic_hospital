"""Phase A: ICD-10 chapter -> clinical specialty map (spec_v1.33 §17.4).

The deterministic spine for specialty-conditioned summarization. Each diagnosis
is assigned one or more *home* specialties from its ICD-10-CM code (the PRIMARY
tier). Relevance to OTHER specialties is layered on separately via the
relatedness graph (Phase B); a diagnosis can also be PRIMARY to several
specialties (co-managed), so the map is a relation, not a partition.

Mapping uses the 3-character ICD-10 prefix and lexical range comparison.

Reflects the second-clinician review (2026-06-07): multi-home co-management for
infection-prone organ chapters (G/H/I/J/L/M/P), sub-range homes for congenital
(Q), injury (S/T) and status (Z) chapters, and four added specialties.
"""

from __future__ import annotations

# Home specialties (() = cross-cutting / not a clinical specialty).
SPECIALTIES = (
    "Cardiology", "Pulmonology", "Gastroenterology", "Nephrology", "Urology",
    "Endocrinology", "Neurology", "Psychiatry", "Dermatology", "Rheumatology",
    "Hematology_Oncology", "Ophthalmology", "Otolaryngology",
    "Obstetrics_Gynecology", "Neonatology", "Infectious_Disease",
    # Added in the second-clinician review:
    "Emergency_Medicine", "Orthopedics", "General_Surgery", "Medical_Genetics",
)

# Ordered (low_prefix, high_prefix, home_specialties_tuple) by 3-char ICD-10 code.
# An empty tuple means no home specialty (cross-cutting / not a problem).
CHAPTER_RANGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A00", "B99", ("Infectious_Disease",)),
    ("C00", "D89", ("Hematology_Oncology",)),
    ("E00", "E89", ("Endocrinology",)),
    # F split (review): F01-F09 organic/dementia -> Neurology; rest -> Psychiatry.
    ("F01", "F09", ("Neurology", "Psychiatry")),
    ("F10", "F99", ("Psychiatry",)),
    # G: only inflammatory/infectious CNS (G00-G09: meningitis/encephalitis) co-homes to ID
    ("G00", "G09", ("Neurology", "Infectious_Disease")),
    ("G10", "G99", ("Neurology",)),
    # Eye: keratitis (H16) is infectious/inflammatory -> +ID. Other infectious-eye codes
    # (orbital cellulitis H05.0, endophthalmitis H44.0, infectious conjunctivitis H10.0/2/3)
    # are finer than 3-char and handled by CODE_OVERRIDES below.
    ("H00", "H15", ("Ophthalmology",)),
    ("H16", "H16", ("Ophthalmology", "Infectious_Disease")),
    ("H17", "H59", ("Ophthalmology",)),
    # ENT: otitis externa/media (H60-H67) are infectious -> +ID; rest ENT only
    ("H60", "H67", ("Otolaryngology", "Infectious_Disease")),
    ("H68", "H95", ("Otolaryngology",)),
    # I: only rheumatic fever / chronic rheumatic heart disease (I00-I09) co-homes to
    # Rheumatology; cerebrovascular (I60-I69) -> Neurology ONLY (cardio-source links such
    # as AFib->stroke are carried by the typed-edge graph, not by co-homing); rest Cardiology.
    ("I00", "I09", ("Cardiology", "Rheumatology")),
    ("I10", "I59", ("Cardiology",)),
    ("I5A", "I5A", ("Cardiology",)),   # non-ischemic myocardial injury (alphanumeric, sorts after I59)
    ("I60", "I69", ("Neurology",)),
    ("I70", "I80", ("Cardiology",)),
    ("I81", "I81", ("Gastroenterology",)),   # portal vein thrombosis (portal-HTN/cirrhosis, not cardiac)
    ("I82", "I84", ("Cardiology",)),
    ("I85", "I85", ("Gastroenterology",)),   # esophageal varices (portal-HTN/cirrhosis)
    ("I86", "I99", ("Cardiology",)),
    # J: acute respiratory infections + influenza/pneumonia (J00-J22) co-home to ID
    ("J00", "J22", ("Pulmonology", "Infectious_Disease")),
    ("J30", "J99", ("Pulmonology",)),
    # K oral cavity (K00-K14) -> Otolaryngology (closest specialty; no Dentistry/OMFS in set).
    # NB: ICD-10-CM has NO K07/K10 — dentofacial anomalies & jaw diseases live at M26/M27.
    ("K00", "K14", ("Otolaryngology",)),     # teeth / gingiva / salivary / stomatitis / lip-oral / tongue
    ("K15", "K34", ("Gastroenterology",)),   # esophagus / stomach / duodenum
    ("K35", "K46", ("Gastroenterology", "General_Surgery")),  # appendicitis (K35-38) + hernias (K40-46)
    ("K47", "K95", ("Gastroenterology",)),   # intestine / peritoneum / liver / biliary / pancreas
    # L: skin/subcutaneous infections (L00-L08) co-home to ID; rest Dermatology only
    ("L00", "L08", ("Dermatology", "Infectious_Disease")),
    ("L10", "L99", ("Dermatology",)),
    # M: split inflammatory/connective-tissue (Rheumatology) from structural/mechanical
    # (Orthopedics); ID only for infective arthropathy/osteomyelitis. Was whole-M->Rheum,
    # which mis-routed diastasis recti (M62), OA (M15-19), spine (M40-54), fractures' sequelae.
    ("M00", "M03", ("Rheumatology", "Infectious_Disease")),   # infective/reactive arthropathy
    ("M04", "M14", ("Rheumatology",)),                        # inflammatory arthropathies
    ("M1A", "M1A", ("Rheumatology",)),                        # chronic gout (alphanumeric code)
    ("M15", "M19", ("Orthopedics",)),                        # osteoarthritis (degenerative -> Ortho)
    ("M20", "M25", ("Orthopedics",)),                         # other joint / acquired deformity
    ("M26", "M27", ("Otolaryngology",)),                      # dentofacial anomalies / jaw diseases (OMFS)
    ("M28", "M29", ("Orthopedics",)),                         # (reserved; no current ICD-10-CM codes)
    ("M30", "M36", ("Rheumatology",)),                        # systemic connective tissue
    ("M40", "M44", ("Orthopedics",)),                         # kyphosis/lordosis/other mechanical spine
    ("M45", "M46", ("Rheumatology",)),                        # ankylosing spondylitis / inflammatory spondylopathy
    ("M47", "M54", ("Orthopedics",)),                         # spondylosis / disc / dorsalgia (mechanical)
    ("M60", "M79", ("Orthopedics",)),                         # muscle / soft tissue (M62.0 diastasis,
    #                                  M62.82 rhabdo, M79.7 fibromyalgia carved out via CODE_OVERRIDES)
    ("M80", "M80", ("Endocrinology", "Orthopedics")),         # osteoporosis WITH pathological fracture
    ("M81", "M83", ("Endocrinology",)),                       # osteoporosis w/o fracture; osteomalacia (metabolic)
    ("M84", "M84", ("Orthopedics",)),                         # disorder of continuity of bone (fracture/nonunion)
    ("M85", "M85", ("Endocrinology", "Orthopedics")),         # other bone density/structure
    ("M86", "M86", ("Orthopedics", "Infectious_Disease")),    # osteomyelitis (infective) -> +ID
    ("M87", "M87", ("Orthopedics",)),                         # osteonecrosis (ischemic, not infective)
    ("M88", "M88", ("Endocrinology", "Orthopedics")),         # Paget disease of bone (metabolic)
    ("M89", "M90", ("Orthopedics",)),                         # other osteopathies / chondropathies
    ("M91", "M99", ("Orthopedics",)),                         # chondropathies / other MSK
    ("N00", "N19", ("Nephrology",)),
    ("N20", "N23", ("Nephrology", "Urology")),   # urinary calculi / renal colic (procedurally Urology-led)
    ("N24", "N29", ("Nephrology",)),
    ("N30", "N53", ("Urology",)),
    ("N60", "N98", ("Obstetrics_Gynecology",)),
    ("N99", "N99", ("Urology",)),
    ("O00", "O9A", ("Obstetrics_Gynecology",)),
    ("P00", "P96", ("Neonatology", "Obstetrics_Gynecology")),
    # Q congenital — organ specialist by sub-range + Medical_Genetics
    ("Q00", "Q07", ("Neurology", "Medical_Genetics")),
    ("Q10", "Q18", ("Ophthalmology", "Otolaryngology", "Medical_Genetics")),
    ("Q20", "Q28", ("Cardiology", "Medical_Genetics")),
    ("Q30", "Q34", ("Pulmonology", "Medical_Genetics")),
    ("Q35", "Q37", ("Otolaryngology", "Medical_Genetics")),
    ("Q38", "Q45", ("Gastroenterology", "Medical_Genetics")),
    ("Q50", "Q56", ("Urology", "Obstetrics_Gynecology", "Medical_Genetics")),
    ("Q60", "Q64", ("Nephrology", "Urology", "Medical_Genetics")),
    ("Q65", "Q79", ("Orthopedics", "Medical_Genetics")),
    ("Q80", "Q99", ("Medical_Genetics",)),
    # R symptoms/signs — findings, NOT problems -> no home (see REVIEW_NOTES).
    ("R00", "R99", ()),
    # S/T injury/poisoning -> trauma team (multi-home; relation refines per dx).
    ("S00", "T88", ("Emergency_Medicine", "Orthopedics", "General_Surgery")),
    ("U00", "U99", ("Infectious_Disease",)),          # COVID/SARS
    ("V00", "Y99", ()),                                # external causes -> none
    # Z status/context -> mostly none, a few sub-ranges have a home.
    ("Z00", "Z29", ()),
    ("Z30", "Z39", ("Obstetrics_Gynecology",)),        # reproductive / pregnancy
    ("Z3A", "Z3A", ("Obstetrics_Gynecology",)),        # weeks of gestation (alphanumeric, sorts after Z39)
    ("Z40", "Z50", ()),
    ("Z51", "Z51", ("Hematology_Oncology",)),          # chemo / radiotherapy encounter
    ("Z52", "Z84", ()),
    ("Z85", "Z86", ("Hematology_Oncology",)),          # personal history of malignancy
    ("Z87", "Z99", ()),
)

# Sub-3-character carve-outs the prefix-only CHAPTER_RANGES cannot express. Keyed by
# normalized (dot-stripped) code prefix; the LONGEST matching prefix wins and overrides
# the chapter range. Reviewed in the specialty-map audit (2026-06-08).
CODE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "M620":  ("General_Surgery",),                    # diastasis recti (M62.0) -> surgery
    "M6282": (),                                       # rhabdomyolysis (M62.82) -> no home (metabolic; via AKI edge)
    "M797":  ("Rheumatology",),                        # fibromyalgia (M79.7)
    "H050":  ("Ophthalmology", "Infectious_Disease"),  # orbital cellulitis (H05.0)
    "H440":  ("Ophthalmology", "Infectious_Disease"),  # endophthalmitis (H44.0)
    "H100":  ("Ophthalmology", "Infectious_Disease"),  # mucopurulent conjunctivitis (H10.0)
    "H102":  ("Ophthalmology", "Infectious_Disease"),  # other acute conjunctivitis (H10.2)
    "H103":  ("Ophthalmology", "Infectious_Disease"),  # acute conjunctivitis, unspecified (H10.3)
}


def _override_for(norm: str) -> str | None:
    """Longest matching CODE_OVERRIDES key for a normalized (dot-stripped) code, or None."""
    best = None
    for k in CODE_OVERRIDES:
        if norm.startswith(k) and (best is None or len(k) > len(best)):
            best = k
    return best


REVIEW_NOTES = {
    "Multi-home (co-primary)": (
        "G/H/I/J/L/M/P and the Q sub-ranges assign SEVERAL home specialties "
        "(e.g. M septic arthritis -> Rheumatology + Infectious_Disease). A "
        "diagnosis is primary to all of them; the relatedness graph adds further "
        "relevance links."
    ),
    "F (FLAGGED)": (
        "Reviewer marked F->Neurology, but the F samples shown were only F01 "
        "dementia (the first codes). F10-F99 are core psychiatric disorders "
        "(depression, schizophrenia, substance use). Applied as: F01-09 organic "
        "-> Neurology+Psychiatry, F10-99 -> Psychiatry. PLEASE CONFIRM."
    ),
    "R (FLAGGED)": (
        "Reviewer marked R->Cardiology, but the R samples shown were only R00-R02 "
        "(cardiac rhythm — the first codes). R spans ALL systems (R10 abd pain, "
        "R51 headache, ...). Kept R as no-home (symptoms are findings, surfaced "
        "via links). PLEASE CONFIRM or specify a sub-range split."
    ),
    "S/T (1017)": "Emergency_Medicine + Orthopedics + General_Surgery (coarse multi-home).",
    "Z": "Most Z -> none; Z30-39 -> OB/GYN; Z51 + Z85-86 -> Hematology_Oncology.",
}


def _prefix(code: str) -> str:
    return code.replace(".", "").replace(" ", "").strip().upper()[:3]


def specialties_for_icd10(code: str | None) -> tuple[str, ...]:
    """All home specialties for an ICD-10-CM code (() if cross-cutting/non-specialty).
    A CODE_OVERRIDES sub-category carve-out takes precedence over the chapter range."""
    if not code:
        return ()
    norm = code.replace(".", "").replace(" ", "").strip().upper()
    ov = _override_for(norm)
    if ov is not None:
        return CODE_OVERRIDES[ov]
    p = norm[:3]
    if len(p) < 3:
        return ()
    for lo, hi, specs in CHAPTER_RANGES:
        if lo <= p <= hi:
            return specs
    return ()


def specialty_for_icd10(code: str | None) -> str | None:
    """The single primary (first) home specialty, or None."""
    specs = specialties_for_icd10(code)
    return specs[0] if specs else None


def home_provenance(code: str | None) -> str:
    """How a code's home was assigned: 'curated-override' (CODE_OVERRIDES),
    'chapter-default' (CHAPTER_RANGES), or 'unmapped'."""
    if not code:
        return "unmapped"
    norm = code.replace(".", "").replace(" ", "").strip().upper()
    if _override_for(norm) is not None:
        return "curated-override"
    p = norm[:3]
    if len(p) >= 3 and any(lo <= p <= hi for lo, hi, _ in CHAPTER_RANGES):
        return "chapter-default"
    return "unmapped"


def specialty_distribution(conn) -> dict[str | None, int]:
    """Count diagnoses per primary home specialty (for review)."""
    counts: dict[str | None, int] = {}
    for (code,) in conn.execute("SELECT icd10_code FROM diagnoses"):
        s = specialty_for_icd10(code)
        counts[s] = counts.get(s, 0) + 1
    return counts


def _matched_range_index(code: str) -> int | None:
    p = _prefix(code)
    if len(p) < 3:
        return None
    for idx, (lo, hi, _specs) in enumerate(CHAPTER_RANGES):
        if lo <= p <= hi:
            return idx
    return None


def _fmt(specs: tuple[str, ...]) -> str:
    return " | ".join(specs) if specs else "(none)"


def write_review(conn, md_path: str, csv_path: str) -> None:
    """Write a clinician-review packet: a readable .md and a markable .csv."""
    import csv as _csv

    rows = conn.execute(
        "SELECT diagnosis_id, icd10_code, display_name FROM diagnoses "
        "WHERE icd10_code IS NOT NULL ORDER BY icd10_code"
    ).fetchall()
    by_range: dict[int | None, list] = {}
    by_specialty: dict[str, list] = {}
    for dx_id, code, name in rows:
        by_range.setdefault(_matched_range_index(code), []).append((code, name))
        for spec in (specialties_for_icd10(code) or ("(none)",)):
            by_specialty.setdefault(spec, []).append((code, name))

    def samples(items, n=6):
        return "; ".join(f"{nm} [{c}]" for c, nm in items[:n])

    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["icd10_range", "current_specialty", "n_diagnoses",
                    "sample_diagnoses", "corrected_specialty", "reviewer_note"])
        for idx, (lo, hi, specs) in enumerate(CHAPTER_RANGES):
            items = by_range.get(idx, [])
            w.writerow([f"{lo}-{hi}", _fmt(specs), len(items), samples(items), "", ""])

    lines = ["# Specialty Map — Review (rev. 2026-06-07)\n",
             "Each diagnosis gets one or more *home* specialties (PRIMARY tier). "
             "Multi-home = co-managed. `(none)` = cross-cutting (surfaces via links).\n",
             "## Chapter → specialty rules\n",
             "| ICD-10 range | Specialty(ies) | # dx | sample diagnoses |",
             "|---|---|---|---|"]
    for idx, (lo, hi, specs) in enumerate(CHAPTER_RANGES):
        items = by_range.get(idx, [])
        lines.append(f"| {lo}–{hi} | {_fmt(specs)} | {len(items)} | {samples(items, 4)} |")
    lines.append("\n## Notes\n")
    for k, v in REVIEW_NOTES.items():
        lines.append(f"- **{k}**: {v}")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    import argparse
    import sqlite3
    ap = argparse.ArgumentParser(description="Phase A: ICD-10 -> specialty map review")
    ap.add_argument("--db", default="data/benchmark_v1.2.db")
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    if args.review:
        write_review(conn, "specialty_map_review.md", "specialty_map_review.csv")
        print("Wrote specialty_map_review.md and specialty_map_review.csv")
        return
    dist = specialty_distribution(conn)
    total = sum(dist.values())
    none_n = dist.get(None, 0)
    print(f"diagnoses: {total} | with home specialty: {total - none_n} "
          f"({100*(total-none_n)/total:.0f}%) | none: {none_n}")
    for spec, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {str(spec):24s} {n}")


if __name__ == "__main__":
    main()
