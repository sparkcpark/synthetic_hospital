"""Build the redistributable benchmark database for the public repo from data/benchmark_v1.3.db.

Kept: patients, encounters, chart sections, ground truth for the four tasks, section-level relevance judgments,
imaging orders, the ontology graph (diagnoses, findings, diagnosis-finding relations) and the question-level
annotation links (ids and roles only). Excluded: everything carrying source text (raw cards, board-question text,
fact cards and their links, source EHR sections), licensed terminology tables, model outputs and LLM call logs."""
import sqlite3, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/benchmark_v1.3.db"; OUT = ROOT / "release_export_v1.3/benchmark_v1.3.db"
KEEP = {  # table -> (columns to drop, row filter)
 "longitudinal_patients": (["pcp_name"], None), "longitudinal_encounters": ([], None), "encounter_ehr_sections": (["source_section_id"], None),
 "benchmark_ground_truth": ([], None), "relevance_judgments": ([], "passage_source = 'encounter_section'"), "imaging_orders": ([], None),
 "diagnoses": ([], None), "clinical_findings": ([], None), "diagnosis_findings": ([], None), "question_findings": ([], None), "question_diagnoses": ([], None),
 "board_questions": (["raw_card_id", "source_qid", "vignette_text", "vignette_html", "question_stem", "answer_choices", "correct_answer", "correct_explanation",
                      "distractor_explanations", "cloze_raw", "cloze_answers", "tags_normalized", "extraction_method", "extraction_model", "extraction_confidence"], None),
 "release_info": ([], None)}
if OUT.exists(): OUT.unlink()
src = sqlite3.connect(SRC); dst = sqlite3.connect(OUT); dst.execute("PRAGMA journal_mode=OFF")
for t, (drop, where) in KEEP.items():
    cols = [r[1] for r in src.execute(f'PRAGMA table_info("{t}")') if r[1] not in drop]
    ddl = src.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0]
    # rebuild DDL from the kept columns only (keep declared types; keep CHECKs from the source DDL for ground truth)
    types = {r[1]: r[2] for r in src.execute(f'PRAGMA table_info("{t}")')}; pk = [r[1] for r in src.execute(f'PRAGMA table_info("{t}")') if r[5]]
    defs = [f'"{c}" {types[c]}' + (" PRIMARY KEY" if c in pk and len(pk) == 1 else "") for c in cols]
    if t == "benchmark_ground_truth": defs += ["CHECK (task IN ('patient_diagnosis','context_summarization','evidence_retrieval','imaging_indication'))", "CHECK (split IN ('train','public','heldout'))"]
    dst.execute(f'CREATE TABLE "{t}" ({", ".join(defs)})')
    q = f'SELECT {", ".join(chr(34)+c+chr(34) for c in cols)} FROM "{t}"' + (f" WHERE {where}" if where else "")
    rows = src.execute(q).fetchall(); dst.executemany(f'INSERT INTO "{t}" VALUES ({",".join("?"*len(cols))})', rows); dst.commit()
    print(f"  {t:26s}{len(rows):>8d} rows  {len(cols)} cols" + (f"  (dropped: {', '.join(drop)})" if drop else ""))
for t, c in (("benchmark_ground_truth", "patient_id"), ("benchmark_ground_truth", "split"), ("encounter_ehr_sections", "encounter_id"), ("longitudinal_encounters", "patient_id"), ("relevance_judgments", "gt_id"), ("imaging_orders", "encounter_id"), ("question_findings", "question_id"), ("question_diagnoses", "question_id")):
    dst.execute(f'CREATE INDEX idx_{t}_{c} ON "{t}"("{c}")')
dst.execute("INSERT OR REPLACE INTO release_info VALUES ('redistributable', 'source text (raw cards, board-question text, fact cards, source EHR sections), licensed terminology tables, model outputs and LLM logs are excluded; retrieval judgments are chart sections only')")
dst.commit(); dst.execute("VACUUM"); dst.close(); print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")
