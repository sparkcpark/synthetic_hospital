"""Export the live Postgres benchmark to the v1.3 SQLite release artifact: data/benchmark_v1.3.db.

v1.3 = v1.2 corpus + (a) the diagnosis_accuracy task removed (ground truth, runs, predictions), (b) split labels
public / train / heldout (RL release; see data/rl_split.json), (c) chart text exactly as in Postgres.
Same table set as benchmark_v1.2.db. Types: jsonb/arrays -> JSON text, timestamps -> ISO text, bool -> 0/1, enums -> text.
"""
import json, sqlite3, sys, datetime, decimal
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from eval.config import get_pg_connection
OUT = ROOT / "data" / "benchmark_v1.3.db"
TABLES = ["source_decks", "raw_cards", "board_questions", "fact_cards", "diagnoses", "clinical_findings", "question_diagnoses", "question_findings",
          "diagnosis_findings", "fact_diagnosis_links", "fact_finding_links", "ehr_sections", "longitudinal_patients", "longitudinal_encounters",
          "encounter_ehr_sections", "benchmark_ground_truth", "relevance_judgments", "evaluation_runs", "evaluation_predictions", "imaging_orders",
          "processing_log", "llm_call_log"]
FILTER = {"benchmark_ground_truth": "where task::text <> 'diagnosis_accuracy'",
          "evaluation_runs": "where task::text <> 'diagnosis_accuracy'",
          "evaluation_predictions": "where run_id in (select run_id from evaluation_runs where task::text <> 'diagnosis_accuracy') and gt_id in (select gt_id from benchmark_ground_truth where task::text <> 'diagnosis_accuracy')",
          "relevance_judgments": "where gt_id in (select gt_id from benchmark_ground_truth where task::text <> 'diagnosis_accuracy')",
          "imaging_orders": "where gt_id is null or gt_id in (select gt_id from benchmark_ground_truth where task::text <> 'diagnosis_accuracy')"}
CHECKS = {"benchmark_ground_truth": ["task IN ('patient_diagnosis','context_summarization','evidence_retrieval','imaging_indication')",
                                     "split IN ('train','public','heldout')", "granularity IN ('question','patient','encounter')"]}
def sqlite_type(dt): return "INTEGER" if dt in ("integer", "bigint", "smallint", "boolean") else "REAL" if dt in ("double precision", "real", "numeric") else "TEXT"
def conv(v):
    if v is None or isinstance(v, (int, float, str)): return v
    if isinstance(v, bool): return int(v)
    if isinstance(v, decimal.Decimal): return float(v)
    if isinstance(v, (datetime.datetime, datetime.date)): return v.isoformat()
    if isinstance(v, (dict, list)): return json.dumps(v)
    return str(v)
if OUT.exists(): OUT.unlink()
pg = get_pg_connection(); cur = pg.cursor(); db = sqlite3.connect(OUT); db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
counts = {}
for t in TABLES:
    cur.execute("select column_name,data_type from information_schema.columns where table_name=%s order by ordinal_position", (t,)); cols = cur.fetchall()
    cur.execute("select a.attname from pg_index i join pg_attribute a on a.attrelid=i.indrelid and a.attnum=any(i.indkey) where i.indrelid=%s::regclass and i.indisprimary", (t,)); pk = {r[0] for r in cur.fetchall()}
    defs = [f'"{c}" {sqlite_type(d)}' + (" PRIMARY KEY" if c in pk and len(pk) == 1 else "") for c, d in cols]
    defs += [f"CHECK ({c})" for c in CHECKS.get(t, [])]
    db.execute(f'CREATE TABLE "{t}" ({", ".join(defs)})')
    cur.execute(f'select {", ".join(chr(34)+c+chr(34) for c,_ in cols)} from "{t}" {FILTER.get(t, "")}')
    n = 0; ph = ",".join("?" * len(cols))
    while True:
        rows = cur.fetchmany(5000)
        if not rows: break
        db.executemany(f'INSERT INTO "{t}" VALUES ({ph})', [tuple(conv(v) for v in r) for r in rows]); n += len(rows)
    db.commit(); counts[t] = n; print(f"  {t:28s}{n:>9d}", flush=True)
for t, c in (("benchmark_ground_truth", "patient_id"), ("benchmark_ground_truth", "task"), ("benchmark_ground_truth", "split"), ("encounter_ehr_sections", "encounter_id"),
             ("longitudinal_encounters", "patient_id"), ("relevance_judgments", "gt_id"), ("evaluation_predictions", "run_id"), ("evaluation_predictions", "gt_id"), ("imaging_orders", "encounter_id")):
    db.execute(f'CREATE INDEX IF NOT EXISTS idx_{t}_{c} ON "{t}"("{c}")')
db.execute("CREATE TABLE release_info (key TEXT PRIMARY KEY, value TEXT)")
db.executemany("INSERT INTO release_info VALUES (?,?)", [("version", "1.3"), ("exported_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
    ("tasks", "patient_diagnosis, context_summarization, evidence_retrieval, imaging_indication"),
    ("splits", "public = 200 patients (reported benchmark); heldout = 268 patients (private evaluation); train = 800 patients (RL training pool)"),
    ("changes_from_1.2", "diagnosis_accuracy task removed; split labels public/train/heldout applied (data/rl_split.json, seed 20260922)")])
db.commit(); db.execute("VACUUM"); db.close()
print(f"\nwrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")
