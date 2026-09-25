#!/usr/bin/env python3
"""Export the 1,268 synthetic longitudinal patient profiles for public release.

Produces, in an output directory:
  - patient_profiles.db      Stripped SQLite (patients + encounters, provenance removed)
  - patient_profiles.jsonl   One JSON object per patient (profile + nested encounters)
  - leakage_audit.md         Verbatim-overlap report of generated note_text vs. source tables

The export deliberately EXCLUDES: raw_cards, board_questions, source_decks, ehr_sections,
encounter_ehr_sections, and the source_question_ids column — i.e. anything carrying
source-extracted text or a pointer back to it.

Usage:
    python scripts/export_patient_profiles.py \
        --db data/benchmark_v1.2.db --out release_export [--ngram 12 --threshold 0.02]
"""
import argparse
import json
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

# Columns kept for the public encounter record (source_question_ids dropped).
ENCOUNTER_COLS = [
    "encounter_id", "patient_id", "encounter_date", "encounter_type",
    "chief_complaint", "attending_name", "department", "encounter_order",
    "note_text", "generation_method",
]
PATIENT_COLS = [
    # pcp_name intentionally dropped: it is a non-informative placeholder
    # ("Dr. Smith-<patient_id>"). The meaningful provider field is
    # encounters.attending_name.
    "patient_id", "profile", "age", "sex", "race_ethnicity", "insurance",
    "num_encounters", "primary_diagnoses", "comorbidities",
    "generation_seed",
]

_word = re.compile(r"[a-z0-9]+")


def _tokens(text: str):
    return _word.findall((text or "").lower())


def _ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


# Overlap is split by the provenance of the SOURCE deck, which is what determines
# copyright relevance:
#   board_exam decks  -> AI-generated question vignettes (the notes are built from these,
#                        so high overlap is expected-by-construction and benign).
#   fact decks        -> genuine third-party medical-fact decks. Their *facts* are not
#                        copyrightable, but verbatim reuse of a card's explanatory prose
#                        would be. This overlap must be near-zero for a clean release.
SOURCE_GROUPS = {
    "board_exam decks (AI-generated vignettes — benign, overlap expected by construction)":
        "board_exam",
    "fact decks (third-party medical-fact sources — COPYRIGHT SIGNAL, must be ~0)":
        "fact",
}


def build_group_ngrams(conn, deck_type, n):
    """n-grams from raw_cards belonging to decks of the given deck_type.

    deck_type == 'board_exam' selects the AI-generated question decks;
    anything else is treated as the fact-deck group (deck_type != 'board_exam').
    """
    if deck_type == "board_exam":
        deck_ids = [r[0] for r in conn.execute(
            "SELECT deck_id FROM source_decks WHERE deck_type = 'board_exam'")]
    else:
        deck_ids = [r[0] for r in conn.execute(
            "SELECT deck_id FROM source_decks WHERE deck_type != 'board_exam'")]
    if not deck_ids:
        return set()
    qmarks = ",".join("?" * len(deck_ids))
    src = set()
    for (txt,) in conn.execute(
        f"SELECT field_data_text FROM raw_cards "
        f"WHERE deck_id IN ({qmarks}) AND field_data_text IS NOT NULL", deck_ids
    ):
        src |= _ngrams(_tokens(txt), n)
    return src


def audit_leakage(conn, n, threshold, out_dir):
    rows = conn.execute(
        "SELECT encounter_id, patient_id, note_text FROM longitudinal_encounters "
        "WHERE note_text IS NOT NULL"
    ).fetchall()
    note_grams = [(e, p, _ngrams(_tokens(t), n)) for e, p, t in rows]
    note_grams = [(e, p, g) for e, p, g in note_grams if g]

    lines = [
        "# Leakage Audit — generated note_text vs. source tables",
        "",
        f"- n-gram size: **{n} tokens**  |  flag threshold: **{threshold:.0%}**  |  "
        f"encounters audited: **{len(note_grams):,}**",
        "",
        "Overlap is split by the provenance of the source deck. The copyright-relevant number",
        "is the **fact-deck** group (genuine third-party sources) — it must be near zero. The",
        "**board_exam** group is AI-generated vignette content that the notes are assembled from,",
        "so high overlap there is expected by construction and is not a copyright concern.",
        "",
    ]

    results = {}
    for label, deck_type in SOURCE_GROUPS.items():
        print(f"[audit] indexing: {label}")
        src = build_group_ngrams(conn, deck_type, n)
        overlaps, flagged = [], []
        for enc_id, pat_id, grams in note_grams:
            hit = len(grams & src) / len(grams)
            overlaps.append(hit)
            if hit >= threshold:
                flagged.append((enc_id, pat_id, hit, len(grams)))
        mean = sum(overlaps) / len(overlaps) if overlaps else 0.0
        mx = max(overlaps) if overlaps else 0.0
        flagged.sort(key=lambda r: -r[2])
        results[label] = (mean, mx, flagged, len(src))
        print(f"[audit]   mean={mean:.4%} max={mx:.4%} flagged={len(flagged)}")

        lines += [
            f"## {label}",
            "",
            f"- distinct source {n}-grams: **{len(src):,}**",
            f"- mean overlap: **{mean:.4%}**  |  max overlap: **{mx:.4%}**  |  "
            f"flagged: **{len(flagged)}**",
            "",
        ]
        if flagged:
            lines += ["| encounter_id | patient_id | overlap | n-grams |", "|---|---|---|---|"]
            lines += [f"| {e} | {p} | {h:.2%} | {ng} |" for e, p, h, ng in flagged[:25]]
            if len(flagged) > 25:
                lines.append(f"\n_…and {len(flagged) - 25} more._")
        else:
            lines.append("No encounters exceeded the threshold.")
        lines.append("")

    (out_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n")
    return results


def export(conn, out_dir):
    # --- stripped SQLite ---
    # Build on local tmpfs first: networked/overlay mounts reject SQLite journal writes.
    tmp_db = Path(tempfile.gettempdir()) / "patient_profiles_build.db"
    if tmp_db.exists():
        tmp_db.unlink()
    dst = sqlite3.connect(tmp_db)
    dst.execute("DROP TABLE IF EXISTS patients")
    dst.execute("DROP TABLE IF EXISTS encounters")
    dst.execute(f"CREATE TABLE patients ({', '.join(PATIENT_COLS)})")
    dst.execute(f"CREATE TABLE encounters ({', '.join(ENCOUNTER_COLS)})")

    pats = conn.execute(f"SELECT {', '.join(PATIENT_COLS)} FROM longitudinal_patients").fetchall()
    dst.executemany(
        f"INSERT INTO patients VALUES ({', '.join('?' * len(PATIENT_COLS))})", pats)
    encs = conn.execute(
        f"SELECT {', '.join(ENCOUNTER_COLS)} FROM longitudinal_encounters ORDER BY patient_id, encounter_order"
    ).fetchall()
    dst.executemany(
        f"INSERT INTO encounters VALUES ({', '.join('?' * len(ENCOUNTER_COLS))})", encs)
    dst.commit()
    dst.close()
    shutil.copyfile(tmp_db, out_dir / "patient_profiles.db")
    tmp_db.unlink()

    # --- JSONL (one record per patient) ---
    enc_by_pat = {}
    for row in encs:
        d = dict(zip(ENCOUNTER_COLS, row))
        enc_by_pat.setdefault(d["patient_id"], []).append(d)
    with open(out_dir / "patient_profiles.jsonl", "w") as f:
        for row in pats:
            p = dict(zip(PATIENT_COLS, row))
            p["encounters"] = enc_by_pat.get(p["patient_id"], [])
            f.write(json.dumps(p) + "\n")

    print(f"[export] {len(pats)} patients, {len(encs)} encounters -> {out_dir}")
    return len(pats), len(encs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/benchmark_v1.2.db")
    ap.add_argument("--out", default="release_export")
    ap.add_argument("--ngram", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.02)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)

    n_pat, n_enc = export(conn, out_dir)
    audit_leakage(conn, args.ngram, args.threshold, out_dir)
    print("done.")


if __name__ == "__main__":
    main()
