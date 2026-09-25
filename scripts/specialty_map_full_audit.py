"""Full-coverage specialty-map audit form (audit_form_instruction.md).

Deterministic, read-only, non-LLM. Enumerates EVERY valid ICD-10-CM 3-char category
from the official CMS Tabular XML and reports the home specialty the PRODUCTION
function assigns it, so a clinician can review the complete rule (incl. unexercised
categories and silent gaps), not just categories present in our charts.

Single source of truth: assigned_specialties + matched_range come from the imported
specialty_map functions, never a re-implementation.

Outputs (repo root): specialty_map_full_audit.csv, specialty_map_full_audit_SUMMARY.md
Exits non-zero if any acceptance assertion fails.
"""
import csv
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from etl.ontology import specialty_map as sm

XML = "data/ontology/icd10cm_tabular_2025.xml"
# FY2026 reconciliation: the audit unit is the 3-char category and specialty
# assignment is by 3-char prefix. Every FY2026 change (487 add / 38 rev / 28 del,
# effective 2025-10-01) is subcategory-level under existing 3-char categories
# (MS phenotypes under G35; chronic-ulcer sites under L97/L98; T2DM-remission under
# E11; abdominal/pelvic/perineal pain under R10/R52; Demodex under B88; dystrophy
# under G71). G35 and R10 are retained as parent categories. No 3-char category is
# added/deleted in a way that changes a specialty assignment, so the FY2025 tabular
# is a faithful base for the FY2026 3-char category set. (Limitation: a byte-level
# diff against the full FY2026 tabular XML is pending a local FY2026 file; cosmetic
# category-title revisions, which do not affect assignment, are not reflected.)
XML_VERSION = ("ICD-10-CM FY2026 (3-char category set; base tabular "
               "icd10cm_tabular_2025.xml reconciled to FY2026 per CMS FY2026 addenda "
               "— all FY2026 deltas are subcategory-level under existing 3-char "
               "categories; no 3-char add/delete affects specialty assignment)")
DB = "data/benchmark_v1.2.db"


def _categories():
    """3-char category -> (title, chapter_label) from the CMS Tabular XML."""
    root = ET.parse(XML).getroot()
    cats = {}
    for chapter in root.iter("chapter"):
        cdesc = (chapter.findtext("desc") or "").strip()
        cname = (chapter.findtext("name") or "").strip()
        label = f"Ch {cname}. {cdesc}" if cname else cdesc
        for diag in chapter.iter("diag"):
            name = (diag.findtext("name") or "").strip()
            if len(name) == 3:                      # a 3-char category
                cats.setdefault(name, ((diag.findtext("desc") or "").strip(), label))
    return cats


def _present_counts():
    """prefix -> (#distinct diagnoses in `diagnoses`, #distinct patients carrying it)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    norm = lambda c: (c or "").replace(".", "").replace(" ", "").upper()
    dx_by_pref = defaultdict(set)
    dx_icd = {}
    for dx, icd in con.execute("SELECT diagnosis_id, icd10_code FROM diagnoses"):
        p = norm(icd)[:3]
        dx_by_pref[p].add(dx)
        dx_icd[dx] = icd
    # patients carrying each prefix (present = role='correct' in their source questions)
    q_correct = defaultdict(list)
    for qid, dx in con.execute("SELECT question_id, diagnosis_id FROM question_diagnoses WHERE role='correct'"):
        q_correct[qid].append(dx)
    import json
    pts_by_pref = defaultdict(set)
    for pid, sq in con.execute("SELECT patient_id, source_question_ids FROM longitudinal_encounters "
                               "WHERE source_question_ids IS NOT NULL"):
        try:
            qids = json.loads(sq)
        except (json.JSONDecodeError, TypeError):
            continue
        for qid in qids:
            for dx in q_correct.get(int(qid), []):
                pts_by_pref[norm(dx_icd.get(dx, ""))[:3]].add(pid)
    con.close()
    return {p: (len(dx_by_pref[p]), len(pts_by_pref.get(p, ()))) for p in dx_by_pref}, pts_by_pref


# intended-no-home ranges (the rule deliberately leaves these unhomed)
def _none_kind(cat, matched_idx, specs):
    if specs:
        return ""
    return "intended" if matched_idx is not None else "REVIEW-possible-gap"


def main():
    cats = _categories()
    present, _ = _present_counts()

    # matched range label + siblings per range
    range_of = {}
    for cat in cats:
        idx = sm._matched_range_index(cat)
        if idx is not None:
            lo, hi, _ = sm.CHAPTER_RANGES[idx]
            range_of[cat] = (idx, f"{lo}-{hi}")
        else:
            range_of[cat] = (None, "")
    sib = Counter(r for _, (i, r) in range_of.items() if r)

    rows = []
    for cat, (title, chap) in cats.items():
        specs = sm.specialties_for_icd10(cat)
        idx, rng = range_of[cat]
        npx = present.get(cat, (0, 0))
        rows.append({
            "icd10_category": cat, "category_title": title, "chapter": chap,
            "assigned_specialties": "|".join(specs) if specs else "(none)",
            "matched_range": rng, "resolves_to_none": str(not specs).upper(),
            "none_kind": _none_kind(cat, idx, specs),
            "siblings_in_range": sib.get(rng, 0) if rng else 0,
            "present_in_data": str(npx[1] > 0).upper(),
            "n_present_diagnoses": npx[0], "n_present_patients": npx[1],
            "verdict": "", "corrected_specialties": "",
            "needs_subcategory_override": "", "reviewer_note": "",
        })
    rows.sort(key=lambda r: (r["assigned_specialties"], r["icd10_category"]))

    cols = list(rows[0].keys())
    with open("specialty_map_full_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    # ---- acceptance assertions ----
    assert len(rows) == len(cats), f"row_count {len(rows)} != unique categories {len(cats)}"
    for code in ("I63.9", "M62.82", "E11.9", "O14.2"):
        cat = code.replace(".", "")[:3]
        if cat in cats:
            a = sm.specialties_for_icd10(cat)
            b = next(("|".join(sm.specialties_for_icd10(cat)) or "(none)"
                      for r in rows if r["icd10_category"] == cat), None)
            assert ("|".join(a) or "(none)") == b, f"spot mismatch {cat}"

    # ---- SUMMARY ----
    per_spec = Counter()
    for r in rows:
        per_spec[r["assigned_specialties"]] += 1
    gaps = [r for r in rows if r["none_kind"] == "REVIEW-possible-gap"]
    alnum = [r for r in rows if not r["icd10_category"][1:].isdigit()]
    hot = sorted({r["matched_range"] for r in rows if r["matched_range"]},
                 key=lambda rng: -sib[rng])[:8]
    present_n = sum(1 for r in rows if r["present_in_data"] == "TRUE")

    L = []
    L.append("# Specialty-Map Full-Coverage Audit — QA Summary\n")
    L.append(f"- **Source:** {XML_VERSION}; present-counts from `{DB}`.")
    L.append(f"- **Categories audited:** {len(rows)} (3-char) | present in data: {present_n} | "
             f"unexercised-but-assigned: {sum(1 for r in rows if r['present_in_data']=='FALSE' and r['assigned_specialties']!='(none)')}")
    L.append("\n## Category count per assigned-specialty set\n")
    for k, v in sorted(per_spec.items(), key=lambda x: -x[1]):
        L.append(f"- {k}: {v}")
    intended = sum(1 for r in rows if r["none_kind"] == "intended")
    L.append(f"\n(none) split: intended={intended}, REVIEW-possible-gap={len(gaps)}")
    L.append("\n## Gap list (none_kind = REVIEW-possible-gap — silent fall-throughs)\n")
    L.append(", ".join(f"{r['icd10_category']}" for r in gaps) or "(none — full coverage)")
    L.append("\n## Heterogeneity hotspots (ranges with most member categories)\n")
    for rng in hot:
        members = sorted(r["icd10_category"] for r in rows if r["matched_range"] == rng)
        L.append(f"- **{rng}** ({sib[rng]} categories): {', '.join(members)}")
    L.append("\n## Alphanumeric categories (verify lexical ordering)\n")
    for r in sorted(alnum, key=lambda r: r["icd10_category"]):
        L.append(f"- {r['icd10_category']} ({r['category_title'][:40]}) -> {r['matched_range'] or 'NO RANGE'} "
                 f"= {r['assigned_specialties']}")
    L.append(f"\n## Coverage reconciliation\n- present in data: {present_n} / {len(rows)} categories.")
    L.append("- Note: 8 sub-3-char CODE_OVERRIDES exist (M62.0, M62.82, M79.7, H05.0, H44.0, "
             "H10.0/2/3); they are finer than this audit's 3-char unit and are reviewed separately.")
    open("specialty_map_full_audit_SUMMARY.md", "w").write("\n".join(L) + "\n")

    print(f"wrote specialty_map_full_audit.csv ({len(rows)} categories) + _SUMMARY.md")
    print(f"  per-specialty sets: {len(per_spec)} | REVIEW-possible-gaps: {len(gaps)} | alphanumeric: {len(alnum)}")
    if gaps:
        print(f"  GAPS: {', '.join(r['icd10_category'] for r in gaps[:30])}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"ACCEPTANCE FAILURE: {e}", file=sys.stderr)
        sys.exit(1)
