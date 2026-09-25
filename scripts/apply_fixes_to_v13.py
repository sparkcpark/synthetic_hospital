"""Apply the cohort text clean-up (upstream fixes 1-4 + problem-list de-duplication) to data/benchmark_v1.3.db in place,
and rebuild longitudinal_encounters.note_text from the fixed sections with the Stage 9a assembler."""
import sqlite3, re, json, sys, importlib.util, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("fx", ROOT / "scripts/apply_upstream_fixes.py"); fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
from etl.stages.s09_encounters import _assemble_note_text
DB = ROOT / "data/benchmark_v1.3.db"
def dedup_key(text):
    k = text.lower(); k = re.sub(r"\(diagnosed[^)]*\)", " ", k)
    k = re.sub(r"\b(?:chronic|acute|severe|mild|moderate|history of|status post|question of|possible|probable|s/p)\b", " ", k)
    k = re.sub(r"[^a-z0-9 ]", " ", k); return " ".join(sorted(set(k.split())))
def dedup_problem_list(pmh):
    """Remove repeated Active Problem List bullets (same condition, differing only in case, punctuation or a dated suffix); keep the first."""
    if not pmh or "Active Problem List:" not in pmh: return pmh, 0
    lines = pmh.split("\n"); seen = set(); out = []; dropped = 0; in_list = False
    for l in lines:
        s = l.strip()
        if s.startswith("Active Problem List"): in_list = True; out.append(l); continue
        if in_list and s.startswith("- "):
            k = dedup_key(s[2:])
            if k and k in seen: dropped += 1; continue
            seen.add(k)
        elif in_list and s and not s.startswith("- "): in_list = False
        out.append(l)
    return "\n".join(out), dropped
db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
sex = {r["patient_id"]: (json.loads(r["profile"]) if isinstance(r["profile"], str) else r["profile"]).get("sex", "F") for r in db.execute("select patient_id, profile from longitudinal_patients")}
enc = {r["encounter_id"]: dict(r) for r in db.execute("select encounter_id, patient_id, encounter_date, attending_name, department, encounter_type, chief_complaint from longitudinal_encounters")}
rows = db.execute("select id, encounter_id, section_type, section_text from encounter_ehr_sections order by encounter_id, section_order").fetchall()
counts = collections.Counter(); changed = 0; by_enc = collections.defaultdict(dict); updates = []
for r in rows:
    e = enc[r["encounter_id"]]; new, applied = fx.fix_section(r["section_text"], r["section_type"], sex[e["patient_id"]])
    if r["section_type"] == "pmh":
        new2, d = dedup_problem_list(new)
        if d: applied.append("5-dup-problem"); counts["dup bullets removed"] += d; new = new2
    for a in applied: counts[a] += 1
    if new != r["section_text"]: changed += 1; updates.append((new, r["id"]))
    by_enc[r["encounter_id"]][r["section_type"]] = new
db.executemany("update encounter_ehr_sections set section_text=? where id=?", updates)
n_notes = 0
for eid, e in enc.items():
    note = _assemble_note_text(e["encounter_date"], e["attending_name"], e["department"], e["encounter_type"], e["chief_complaint"], by_enc.get(eid, {}))
    old = db.execute("select note_text from longitudinal_encounters where encounter_id=?", (eid,)).fetchone()[0]
    if note != old: n_notes += 1; db.execute("update longitudinal_encounters set note_text=? where encounter_id=?", (note, eid))
db.execute("insert or replace into release_info values ('text_cleanup', ?)", (f"upstream fixes 1-4 + Active Problem List de-duplication applied to encounter_ehr_sections ({changed} of {len(rows)} sections); note_text rebuilt with the Stage 9a assembler ({n_notes} notes changed)",))
db.commit()
print(f"sections changed: {changed} of {len(rows)}; notes rebuilt with changes: {n_notes} of {len(enc)}"); [print(f"  {k:22s}{v:7d}") for k, v in sorted(counts.items())]
# QA sweep, same checks as build_release_v1_3.py
notes = [r[0] for r in db.execute("select note_text from longitudinal_encounters")]
pats = {"header then lowercase": r"\n[A-Z][A-Za-z /]+:\n\s*[a-z]", "med denial + list": r"(?i)(takes|on) no medications[\s\S]{0,120}Home Medications:\s*\n\s*-",
        "no-history + list": r"(?i)Active Problem List:[\s\S]{0,600}?no (significant )?past medical history", "blank-line runs": r"\n{3,}"}
print("QA on rebuilt notes:", {k: sum(1 for t in notes if t and re.search(p, t)) for k, p in pats.items()})
secs = [r[0] for r in db.execute("select section_text from encounter_ehr_sections")]
print("QA on sections: starts lowercase =", sum(1 for s in secs if s and s[:1].islower()), "| leading whitespace =", sum(1 for s in secs if s and s[:1].isspace()))
pm = [r[0] for r in db.execute("select section_text from encounter_ehr_sections where section_type='pmh'")]
dups = sum(dedup_problem_list(s)[1] for s in pm); print("remaining duplicate problem bullets:", dups)
db.execute("vacuum"); db.close()
