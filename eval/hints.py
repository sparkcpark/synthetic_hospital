"""Ontology hint helpers for structured (ontology-grounded) prompt strategy.

Builds formatted hint strings from preloaded DB data. Each helper returns ""
if inputs are empty/None, so the {structured_hints} placeholder gracefully
disappears from the template.
"""

from __future__ import annotations

# Sync with etl/stages/s08_patients.py:ORGAN_SYSTEM_CANONICAL when adding new source decks.
ORGAN_SYSTEM_NORMALIZER: dict[str, str] = {
    # Cardiovascular
    "Cardiovascular System": "cardiovascular",
    "cardiovascular system": "cardiovascular",
    # Nervous System
    "Nervous System": "nervous_system",
    "neurology": "nervous_system",
    # Psychiatry
    "PsychiatricBehavioral & Substance Abuse": "psychiatry",
    "psychiatry": "psychiatry",
    # GI
    "Gastrointestinal & Nutrition": "gi",
    "gastroenterology": "gi",
    "gastrointestinal system": "gi",
    # Infectious Disease
    "Infectious Diseases": "infectious",
    "infectious diseases": "infectious",
    "infectious disease": "infectious",
    # Pulmonary
    "Pulmonary & Critical Care": "pulmonary",
    "pulmonary and critical care": "pulmonary",
    "respiratory system": "pulmonary",
    # Obstetrics
    "Pregnancy Childbirth & Puerperium": "obstetrics",
    # Female Reproductive
    "Female Reproductive System & Breast": "reproductive_female",
    "reproductive female": "reproductive_female",
    # Male Reproductive
    "Male Reproductive System": "reproductive_male",
    "reproductive male": "reproductive_male",
    # Reproductive (general)
    "reproductive system": "reproductive_general",
    # MSK / Rheumatology
    "RheumatologyOrthopedics & Sports": "msk",
    "musculoskeletal": "msk",
    "rheumatology": "msk",
    # Renal
    "Renal Urinary Systems & Electrolytes": "renal",
    "renal urinary systems and electrolytes": "renal",
    # Heme/Onc
    "Hematology & Oncology": "heme_onc",
    "hematology and oncology": "heme_onc",
    "hematology": "heme_onc",
    "hematologic": "heme_onc",
    # Endocrine
    "Endocrine Diabetes & Metabolism": "endocrine",
    "endocrine diabetes and metabolism": "endocrine",
    # Dermatology
    "Dermatology": "derm",
    "dermatology": "derm",
    # ENT
    "Ear Nose & Throat ENT": "ent",
    "ear nose and throat ent": "ent",
    # Ophthalmology
    "Ophthalmology": "ophthalmology",
    "ophthalmology": "ophthalmology",
    # Toxicology
    "Poisoning & Environmental Exposure": "toxicology",
    "poisoning and environmental exposure": "toxicology",
    # Immunology
    "Allergy & Immunology": "immunology",
    "immunology": "immunology",
    "immune": "immunology",
    # General Principles (non-clinical — no ICD-10 chapter)
    "pharmacology general principles": "pharmacology",
    "pharmacology": "pharmacology",
    "biochemistry general principles": "biochemistry",
    "genetics general principles": "genetics",
    "microbiology general principles": "microbiology",
    "microbiology": "microbiology",
    "pathology general principles": "pathology",
    # Biostats / Social Sciences (non-clinical)
    "biostatistics and epidemiology": "biostatistics",
    "Biostatistics & Epidemiology": "biostatistics",
    "social sciences ethics legal professional": "social_sciences",
    "Social Sciences EthicsLegalProfessional": "social_sciences",
    # Multi-system / General (no ICD-10 chapter)
    "Miscellaneous Multisystem": "multisystem",
    "General Principles": "general",
}

# Only clinical organ systems have chapter mappings.
# Non-clinical keys (pharmacology, biochemistry, genetics, microbiology,
# pathology, biostatistics, social_sciences, general) are absent — lookup
# returns None and the helper functions omit the chapter hint line.
# multisystem omitted — maps to heterogeneous disease chapters (lupus M32,
# sarcoidosis D86, vasculitis M30-M31). R00-R99 (symptom codes) would be
# actively misleading. No chapter hint is better than a wrong one.
CANONICAL_ICD10_CHAPTERS: dict[str, str] = {
    "cardiovascular":       "I00-I99 (Circulatory)",
    "pulmonary":            "J00-J99 (Respiratory)",
    "gi":                   "K00-K95 (Digestive)",
    "renal":                "N00-N99 (Genitourinary)",
    "nervous_system":       "G00-G99 (Nervous system)",
    "endocrine":            "E00-E89 (Endocrine/metabolic)",
    "msk":                  "M00-M99 (Musculoskeletal)",
    "heme_onc":             "C00-D49, D50-D89 (Neoplasms/Blood/immune)",
    "hematologic":          "D50-D89 (Blood/immune disorders)",
    "immunology":           "D50-D89 (Blood/immune disorders)",
    "reproductive_female":  "N00-N99 (Genitourinary)",
    "reproductive_male":    "N00-N99 (Genitourinary)",
    "reproductive_general": "N00-N99 (Genitourinary)",
    "obstetrics":           "O00-O9A (Pregnancy/childbirth)",
    "derm":                 "L00-L99 (Skin)",
    "psychiatry":           "F01-F99 (Mental/behavioral)",
    "infectious":           "A00-B99 (Infectious/parasitic)",
    "ophthalmology":        "H00-H59 (Eye)",
    "ent":                  "H60-H95, J00-J39 (Ear/Upper respiratory)",
    "toxicology":           "T36-T65 (Poisoning)",
}


def _normalize_organ_system(raw: str) -> str:
    """Normalize a raw organ system string to its canonical key."""
    return ORGAN_SYSTEM_NORMALIZER.get(raw, raw.lower().replace(" ", "_"))


def _get_icd10_chapters(organ_systems: list[str]) -> list[str]:
    """Deduplicate organ systems and return unique ICD-10 chapter strings."""
    seen_chapters: dict[str, None] = {}  # ordered dedup
    for raw in organ_systems:
        canonical = _normalize_organ_system(raw)
        chapter = CANONICAL_ICD10_CHAPTERS.get(canonical)
        if chapter and chapter not in seen_chapters:
            seen_chapters[chapter] = None
    return list(seen_chapters)


def format_diagnosis_hints(
    organ_systems: list[str] | None,
    key_findings: list[dict] | None,
) -> str:
    """Build ontology hint block for diagnosis structured strategy."""
    parts: list[str] = []

    chapters = _get_icd10_chapters(organ_systems or [])
    if chapters:
        parts.append(
            "The following ICD-10-CM code ranges cover the most likely diagnostic "
            "categories for this presentation. Use these as anchors when forming "
            "your differential — do not limit yourself to only these codes, but "
            "prefer specific codes within these ranges where clinically appropriate.\n\n"
            f"Relevant categories: {', '.join(chapters)}"
        )

    if key_findings:
        lines = []
        for f in key_findings:
            snomed = f.get("snomed_id", "")
            ftype = f.get("type", "")
            name = f.get("name", "")
            entry = f"- {name}"
            if snomed:
                entry += f" (SNOMED {snomed}"
                if ftype:
                    entry += f", {ftype}"
                entry += ")"
            elif ftype:
                entry += f" ({ftype})"
            lines.append(entry)
        parts.append(
            "Relevant SNOMED findings already identified in this record:\n"
            + "\n".join(lines)
        )

    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


def format_summarization_hints(key_findings: list[dict] | None) -> str:
    """Build ontology hint block for summarization structured strategy."""
    if not key_findings:
        return ""
    lines = []
    for f in key_findings:
        name = f.get("name", "")
        ftype = f.get("type", "")
        entry = f"- {name}"
        if ftype:
            entry += f" ({ftype})"
        lines.append(entry)
    return (
        "The following findings have been identified as clinically key for this "
        "patient's active diagnoses. Your summary must address each of these "
        "findings either directly or in clinical context — do not omit them:\n"
        + "\n".join(lines)
        + "\n\n"
    )


def format_retrieval_hints(
    target_dx: list[dict] | None,
    pathognomonic: list[dict] | None,
) -> str:
    """Build ontology hint block for retrieval structured strategy."""
    parts: list[str] = []

    if target_dx:
        lines = []
        for dx in target_dx:
            name = dx.get("name", "")
            icd10 = dx.get("icd10_code", "")
            if icd10:
                lines.append(f"- {name} ({icd10})")
            else:
                lines.append(f"- {name}")
        parts.append(
            "The target diagnoses for this grading task are:\n" + "\n".join(lines)
        )

    if pathognomonic:
        lines = []
        for f in pathognomonic:
            name = f.get("name", "")
            snomed = f.get("snomed_id", "")
            if snomed:
                lines.append(f"- {name} (SNOMED {snomed})")
            else:
                lines.append(f"- {name}")
        parts.append(
            "Key pathognomonic and highly suggestive findings for these diagnoses "
            "include:\n" + "\n".join(lines) + "\n\n"
            "Passages containing any of these findings should receive Grade 3 or "
            "Grade 2 as appropriate."
        )

    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


def format_imaging_hints(
    findings: list[dict] | None,
    differentials: list[dict] | None,
) -> str:
    """Build ontology hint block for imaging structured strategy."""
    parts: list[str] = []

    if findings:
        lines = []
        for f in findings:
            name = f.get("name", "")
            loinc = f.get("loinc_code", "")
            ftype = f.get("type", "")
            entry = f"- {name}"
            if loinc:
                entry += f" (LOINC {loinc})"
            elif ftype:
                entry += f" ({ftype})"
            lines.append(entry)
        parts.append(
            "The following clinical findings are documented in this patient's "
            "record and are relevant to the imaging order:\n" + "\n".join(lines)
        )

    if differentials:
        lines = []
        for dx in differentials:
            name = dx.get("name", "")
            icd10 = dx.get("icd10_code", "")
            role = dx.get("role", "")
            if icd10:
                lines.append(f"- {name} ({icd10})")
            else:
                lines.append(f"- {name}")
        parts.append(
            "Reference ICD-10-CM codes for likely differentials given this "
            "presentation:\n" + "\n".join(lines)
        )

    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"
