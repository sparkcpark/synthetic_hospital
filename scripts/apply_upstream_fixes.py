#!/usr/bin/env python3
"""Apply upstream text fixes 1-4 from synthetic_data_upstream_fixes.md to the cohort.

  1. Medication list contradicts itself   -- denial clause above a populated
     Home Medications block. Resolved by precedence: the patient profile is the
     longitudinal truth, the source vignette describes one moment, so the denial
     is dropped and the list stands.
  2. PMH asserts no history above a populated problem list -- clean_pmh_tail,
     ported from scripts/trial2_harmonize.py.
  3. Sections begin mid-sentence -- repair_synthetic_fragments, ported likewise.
  4. Leading whitespace -- falls out of the strip/collapse applied to every row.

These edit `section_text`, which is the eval corpus: the published metrics were
computed on the pre-fix text. Run scripts/measure_fix_delta.py before releasing
a cohort fixed this way alongside numbers computed on the old one.

Usage:
    python scripts/apply_upstream_fixes.py --dry-run          # counts + samples
    python scripts/apply_upstream_fixes.py --limit 200        # small live batch
    python scripts/apply_upstream_fixes.py --apply            # whole cohort
    python scripts/apply_upstream_fixes.py --apply --sqlite release_export/....db
"""
from __future__ import annotations

import argparse
import re
import sys

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Patterns (ported verbatim from scripts/trial2_harmonize.py so the cohort fix
# and the validated trial-2 repair cannot drift apart)
# ---------------------------------------------------------------------------

PMH_VERBS = (r"\b(?:is|are|was|were|has|have|had|does|do|did|reports?|denies|"
             r"includes?|included|notes?|notable|significant|presents?|underwent|"
             r"received|takes?|took|remains?|reveals?|shows?|carries|complains?|"
             r"endorses?|describes?|states?|admits?|suffers?|developed|"
             r"diagnosed|treated|managed|hospitalized|follows?)\b")

PMH_DENIALS = (r"\bno (?:past )?(?:medical )?history\b|"
               r"\bno (?:known )?(?:chronic )?(?:medical )?(?:conditions|comorbidities|problems)\b|"
               r"\bdenies any (?:past )?(?:medical )?history\b|"
               r"\bunremarkable (?:past )?medical history\b|"
               r"\b(?:past )?medical history is unremarkable\b|"
               r"\bno significant (?:past )?medical history\b")

ORPHAN_VERBS = (r"is|are|was|were|has|have|had|does|do|did|presents?|reports?|"
                r"denies|complains?|underwent|received|takes?|took|returns?|"
                r"comes?|came|arrives?|states?|notes?|admits?|endorses?|"
                r"describes?|reveals?|shows?|demonstrates?")

ORPHAN_LEADINS = (r"over|during|after|before|for|in|on|at|with|without|since|"
                  r"about|approximately|following|prior")

# Clauses denying medications. Kept narrow: only an explicit assertion of *no*
# medications contradicts a populated list. "takes no aspirin" or "no new
# medications" are compatible with a home-medication list and must survive.
MED_DENIALS = (r"(?:and |but )?(?:s?he |the patient |patient )?"
               r"(?:currently )?(?:is )?(?:takes?|taking|on|receiving|uses?)? ?"
               r"no (?:home |regular |daily |scheduled |current |prescription |chronic )?"
               r"medications?\b[^.]*\.?|"
               r"(?:s?he |the patient )?(?:denies|reports) (?:any |taking )?"
               r"(?:current |home )?medications?\b[^.]*\.?|"
               r"\bnot (?:currently )?(?:taking|on) any (?:medications?|meds)\b[^.]*\.?|"
               r"\bno known medications?\b[^.]*\.?")


# Leading coordinating conjunction left behind when the stem was cut away.
# "and took prenatal vitamins" must not become "She takes and took ...", and
# "and her abdomen is soft" must not become "And her abdomen is soft" -- both
# are the doubled-verb / dangling-conjunction cases the QA sweep requires to be
# zero. The trial-2 sample of 5 patients contained none; the cohort does.
LEADING_CONJ = re.compile(r"^(?:and|but|or|so|yet|then|also)\s+", re.I)


def _strip_conj(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = LEADING_CONJ.sub("", s).lstrip()
    return s


def repair_fragment(text: str, sex: str, section: str) -> str:
    """Restore a subject lost to the board-question stem, or drop a dead fragment."""
    if not text:
        return text
    t = text.strip()
    if not t or not t[0].islower():
        return t

    pronoun = "He" if sex == "M" else "She"

    if section == "medications" and "Home Medications:" in t:
        head, _, tail = t.partition("Home Medications:")
        head = _strip_conj(head.strip())
        if head and head[0].islower():
            if re.match(rf"^(?:{ORPHAN_VERBS})\b", head, re.I):
                head = f"{pronoun} {head}"
            else:
                head = f"{pronoun} takes {head}"
            if not head.endswith((".", "!", "?")):
                head += "."
        return f"{head}\n\nHome Medications:{tail}".strip()

    t = _strip_conj(t)
    if not t:
        return ""
    if not t[0].islower():
        return t

    sentences = re.split(r"(?<=[.!?])\s+", t)
    if len(sentences) > 1 and re.match(rf"^(?:{ORPHAN_LEADINS})\b", sentences[0], re.I) \
            and len(sentences[0]) < 70:
        rest = " ".join(sentences[1:]).strip()
        if rest:
            # The remainder can itself open on a conjunction or a subjectless
            # verb ("over the past two months. and has been noncompliant ...");
            # capitalizing it directly would leave "And has been ...". Re-run the
            # repair on it -- `rest` is strictly shorter, so this terminates.
            return repair_fragment(rest, sex, section)

    if re.match(rf"^(?:{ORPHAN_VERBS})\b", t, re.I):
        return f"{pronoun} {t}"

    return t[0].upper() + t[1:]


# Clause boundary inside a compound sentence: ", and" / ", but", or a bare
# "and"/"but" that introduces a new subject. Requiring the subject keeps
# "pregnancy and vaginal delivery" intact -- that "and" joins nouns, not clauses.
CLAUSE_SPLIT = re.compile(
    r",\s+(?:and|but)\s+|\s+(?:and|but)\s+(?=(?:s?he|the patient|her|his)\b)", re.I)


def _drop_denial_clause(sentence: str) -> str:
    """Remove only the absence-of-history clause, preserving the rest.

    A denial and real content routinely share one sentence -- "Her past medical
    history is unremarkable, and she had an uncomplicated pregnancy and vaginal
    delivery." Dropping the whole sentence, as the trial-2 helper does, deletes
    the delivery history: 431 sentences across the cohort lose content that way.
    Splitting at clause boundaries keeps the additive half the fix document
    explicitly says to retain.
    """
    clauses = [c.strip(" ,;") for c in CLAUSE_SPLIT.split(sentence) if c and c.strip(" ,;")]
    kept = [c for c in clauses if not re.search(PMH_DENIALS, c, re.I)]
    if not kept:
        return ""
    out = ", and ".join(kept) if len(kept) > 1 else kept[0]
    out = out.strip(" ,;")
    if not out:
        return ""
    # A surviving clause may start with a lowercase pronoun once its lead clause
    # is gone ("she had an uncomplicated ..." -> "She had ...").
    out = out[0].upper() + out[1:]
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


def clean_pmh_tail(pmh: str) -> str:
    """Drop vignette remnants and absence-of-history claims trailing the list."""
    if not pmh or "Active Problem List:" not in pmh:
        return pmh
    lines = pmh.split("\n")
    bullets = [i for i, l in enumerate(lines) if l.strip().startswith("-")]
    if not bullets:
        return pmh
    head = lines[:bullets[-1] + 1]
    tail = "\n".join(lines[bullets[-1] + 1:]).strip()
    if not tail:
        return "\n".join(head).strip()

    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", tail):
        s = sent.strip()
        if not s:
            continue
        if s[0].islower() or not re.search(PMH_VERBS, s, re.I):
            continue          # ungrammatical remnant
        if re.search(PMH_DENIALS, s, re.I):
            # contradicts the list above -- strip the denial, keep any real content
            s = _drop_denial_clause(s)
            if not s:
                continue
        if not s.endswith((".", "!", "?")):
            s += "."
        kept.append(s)

    body = "\n".join(head).strip()
    if kept:
        body += "\n\n" + " ".join(kept)
    return body


def reconcile_medications(text: str) -> str:
    """Drop a no-medications claim sitting above a populated Home Medications block.

    Precedence follows the profile: `home_medications` is the longitudinal record,
    while the source stem describes a single vignette moment. Only the denial is
    removed -- surrounding narrative in the same section is preserved -- and only
    when a populated list is actually present, so a genuinely unmedicated patient
    keeps the statement.
    """
    if not text or "Home Medications:" not in text:
        return text
    head, sep, tail = text.partition("Home Medications:")
    # a populated list means at least one bullet after the header
    if not re.search(r"^\s*[-*]\s*\S", tail, re.M):
        return text
    if not re.search(MED_DENIALS, head, re.I):
        return text

    cleaned = re.sub(MED_DENIALS, "", head, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;])", r"\1", cleaned)
    cleaned = re.sub(r"(?:^|(?<=\.))\s*(?:and|but)\s+", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned).strip()
    if cleaned in {".", ",", ";"}:
        cleaned = ""
    if cleaned and not cleaned.endswith((".", "!", "?", ":")):
        cleaned += "."
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return (f"{cleaned}\n\n{sep}{tail}" if cleaned else f"{sep}{tail}").strip()


def normalize_whitespace(text: str) -> str:
    """Strip edges and collapse runs of blank lines (item 4)."""
    if not text:
        return text
    t = re.sub(r"[ \t]+\n", "\n", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def fix_section(text: str, section: str, sex: str) -> tuple[str, list[str]]:
    """Apply every repair to one section; return (new_text, applied_labels)."""
    applied = []
    out = text or ""

    if section == "medications":
        step = reconcile_medications(out)
        if step != out:
            applied.append("1-med")
            out = step

    if section == "pmh":
        step = clean_pmh_tail(out)
        if step != out:
            applied.append("2-pmh")
            out = step

    step = repair_fragment(out, sex, section)
    if step != out:
        applied.append("3-fragment")
        out = step

    step = normalize_whitespace(out)
    if step != out:
        applied.append("4-whitespace")
        out = step

    return out, applied


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", type=int, default=4, help="examples per fix class")
    ap.add_argument("--sqlite", help="also apply to a release SQLite file")
    args = ap.parse_args()
    live = args.apply

    from eval.config import get_pg_connection
    conn = get_pg_connection()

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT s.encounter_id, s.section_type::text, s.section_text,
                   COALESCE(p.sex, 'F')
            FROM encounter_ehr_sections s
            JOIN longitudinal_encounters e ON s.encounter_id = e.encounter_id
            JOIN longitudinal_patients p ON e.patient_id = p.patient_id
            ORDER BY s.encounter_id, s.section_type
            {f'LIMIT {args.limit}' if args.limit else ''}
        """)
        rows = cur.fetchall()

    counts: dict[str, int] = {}
    samples: dict[str, list] = {}
    updates = []
    for enc_id, stype, text, sex in rows:
        new, applied = fix_section(text, stype, sex)
        if not applied:
            continue
        for a in applied:
            counts[a] = counts.get(a, 0) + 1
            if len(samples.setdefault(a, [])) < args.samples:
                samples[a].append((stype, text, new))
        updates.append((new, enc_id, stype))

    print(f"scanned {len(rows)} sections; {len(updates)} would change\n")
    for k in sorted(counts):
        print(f"  {k:14s} {counts[k]:6d}")

    print("\n--- samples ---")
    for k in sorted(samples):
        print(f"\n### {k}")
        for stype, before, after in samples[k]:
            b = re.sub(r"\s+", " ", before)
            a = re.sub(r"\s+", " ", after)
            i = 0
            while i < min(len(b), len(a)) and b[i] == a[i]:
                i += 1
            lo = max(0, i - 40)
            print(f"  [{stype}]\n    before: ...{b[lo:i+110]}\n    after : ...{a[lo:i+110]}")

    if not live:
        print("\n(dry run — nothing written; pass --apply to commit)")
        conn.close()
        return

    with conn.cursor() as cur:
        for new, enc_id, stype in updates:
            cur.execute(
                "UPDATE encounter_ehr_sections SET section_text=%s "
                "WHERE encounter_id=%s AND section_type=%s",
                (new, enc_id, stype),
            )
    conn.commit()
    print(f"\napplied {len(updates)} updates to Postgres")
    conn.close()

    if args.sqlite:
        import sqlite3
        db = sqlite3.connect(args.sqlite)
        n = 0
        for new, enc_id, stype in updates:
            n += db.execute(
                "UPDATE encounter_ehr_sections SET section_text=? "
                "WHERE encounter_id=? AND section_type=?",
                (new, enc_id, stype),
            ).rowcount
        db.commit()
        db.close()
        print(f"applied {n} updates to {args.sqlite}")


if __name__ == "__main__":
    main()
