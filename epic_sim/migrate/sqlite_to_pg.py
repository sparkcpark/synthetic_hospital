"""Migrate data from SQLite (benchmark_v1.2.db) to PostgreSQL.

Usage:
    python -m epic_sim.migrate.sqlite_to_pg [--source PATH] [--batch-size N]
"""

import argparse
import json
import sqlite3
import time

import psycopg

from epic_sim.app.config import settings

# Tables in dependency order (FK-safe insert order)
TABLES_ORDERED = [
    "source_decks",
    "raw_cards",
    "board_questions",
    "fact_cards",
    "diagnoses",
    "clinical_findings",
    "question_diagnoses",
    "question_findings",
    "diagnosis_findings",
    "fact_diagnosis_links",
    "fact_finding_links",
    "ehr_sections",
    "longitudinal_patients",
    "longitudinal_encounters",
    "encounter_ehr_sections",
    "benchmark_ground_truth",
    "relevance_judgments",
    "evaluation_runs",
    "evaluation_predictions",
    "imaging_orders",
    "processing_log",
    "llm_call_log",
]

# Columns stored as INTEGER in SQLite (no native boolean) but BOOLEAN in PostgreSQL.
BOOL_COLUMNS = {
    "benchmark_ground_truth": {"is_diagnostic"},
}

# Columns that store JSON and need TEXT→JSONB conversion
JSONB_COLUMNS = {
    "longitudinal_patients": {"profile"},
    "benchmark_ground_truth": {"ground_truth"},
    "evaluation_runs": {"config", "metrics"},
    "evaluation_predictions": {"prediction"},
    "processing_log": {"config"},
}

# Columns with PostgreSQL ENUM types — values must match exactly
ENUM_COLUMNS = {
    "source_decks": {"deck_type"},
    "raw_cards": {"card_type", "classification_method"},
    "board_questions": {"question_format", "extraction_method"},
    "fact_cards": {"fact_format", "extraction_method"},
    "diagnoses": {"acuity"},
    "clinical_findings": {"finding_type"},
    "question_diagnoses": {"role", "source"},
    "question_findings": {"relevance", "source"},
    "diagnosis_findings": {"relationship"},
    "fact_diagnosis_links": {"relevance"},
    "fact_finding_links": {"relevance"},
    "ehr_sections": {"section_type", "extraction_method"},
    "longitudinal_encounters": {"encounter_type", "generation_method"},
    "benchmark_ground_truth": {"task", "granularity", "difficulty", "split"},
    "relevance_judgments": {"passage_source"},
    "evaluation_runs": {"method_type", "task", "split"},
    "imaging_orders": {"order_priority"},
    "processing_log": {"status"},
}

# Tables that have SERIAL/BIGSERIAL PKs — need sequence reset after insert
PK_COLUMNS = {
    "source_decks": ("deck_id", "source_decks_deck_id_seq"),
    "raw_cards": ("raw_card_id", "raw_cards_raw_card_id_seq"),
    "board_questions": ("question_id", "board_questions_question_id_seq"),
    "fact_cards": ("fact_id", "fact_cards_fact_id_seq"),
    "diagnoses": ("diagnosis_id", "diagnoses_diagnosis_id_seq"),
    "clinical_findings": ("finding_id", "clinical_findings_finding_id_seq"),
    "question_diagnoses": ("id", "question_diagnoses_id_seq"),
    "question_findings": ("id", "question_findings_id_seq"),
    "diagnosis_findings": ("id", "diagnosis_findings_id_seq"),
    "fact_diagnosis_links": ("id", "fact_diagnosis_links_id_seq"),
    "fact_finding_links": ("id", "fact_finding_links_id_seq"),
    "ehr_sections": ("section_id", "ehr_sections_section_id_seq"),
    "longitudinal_patients": ("patient_id", "longitudinal_patients_patient_id_seq"),
    "longitudinal_encounters": ("encounter_id", "longitudinal_encounters_encounter_id_seq"),
    "encounter_ehr_sections": ("id", "encounter_ehr_sections_id_seq"),
    "benchmark_ground_truth": ("gt_id", "benchmark_ground_truth_gt_id_seq"),
    "relevance_judgments": ("judgment_id", "relevance_judgments_judgment_id_seq"),
    "evaluation_runs": ("run_id", "evaluation_runs_run_id_seq"),
    "evaluation_predictions": ("prediction_id", "evaluation_predictions_prediction_id_seq"),
    "imaging_orders": ("order_id", "imaging_orders_order_id_seq"),
    "processing_log": ("log_id", "processing_log_log_id_seq"),
    "llm_call_log": ("call_id", "llm_call_log_call_id_seq"),
}


def _validate_json(value: str | None) -> str | None:
    """Validate and return JSON string, or None."""
    if value is None:
        return None
    try:
        json.loads(value)
        return value
    except (json.JSONDecodeError, TypeError):
        # Try wrapping as string if not valid JSON
        return json.dumps(value)


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    table: str,
    batch_size: int = 1000,
) -> int:
    """Migrate a single table from SQLite to PostgreSQL. Returns row count."""
    # Get column names from SQLite
    cursor = sqlite_conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]

    # Get column names from PostgreSQL (may differ due to added auth columns etc.)
    pg_cursor = pg_conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = %s AND table_schema = 'public' ORDER BY ordinal_position",
        (table,),
    )
    pg_columns = {row[0] for row in pg_cursor.fetchall()}

    # Only migrate columns that exist in both
    shared_columns = [c for c in columns if c in pg_columns]
    if not shared_columns:
        print(f"  SKIP {table}: no shared columns")
        return 0

    jsonb_cols = JSONB_COLUMNS.get(table, set())
    bool_cols = BOOL_COLUMNS.get(table, set())

    # Read all rows from SQLite
    col_list = ", ".join(shared_columns)
    rows = sqlite_conn.execute(f"SELECT {col_list} FROM {table}").fetchall()
    if not rows:
        return 0

    # Prepare INSERT
    placeholders = ", ".join([f"%s" for _ in shared_columns])
    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        converted = []
        for row in batch:
            new_row = []
            for col_name, val in zip(shared_columns, row):
                if col_name in jsonb_cols and val is not None:
                    val = _validate_json(val)
                    # psycopg needs Json wrapper for JSONB
                    val = psycopg.types.json.Json(json.loads(val))
                elif col_name in bool_cols and val is not None:
                    # SQLite stores these as 0/1 integers
                    val = bool(val)
                new_row.append(val)
            converted.append(tuple(new_row))

        with pg_conn.cursor() as cur:
            cur.executemany(insert_sql, converted)
        total += len(batch)

    pg_conn.commit()
    return total


def reset_sequence(pg_conn: psycopg.Connection, table: str) -> None:
    """Reset the PostgreSQL sequence to max(pk) + 1."""
    if table not in PK_COLUMNS:
        return
    pk_col, seq_name = PK_COLUMNS[table]
    pg_conn.execute(
        f"SELECT setval('{seq_name}', COALESCE((SELECT MAX({pk_col}) FROM {table}), 0) + 1, false)"
    )
    pg_conn.commit()


def verify_counts(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
) -> list[str]:
    """Compare row counts. Returns list of mismatches."""
    errors = []
    for table in TABLES_ORDERED:
        try:
            sqlite_count = sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            continue  # Table might not exist in SQLite
        pg_count = pg_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        status = "OK" if sqlite_count == pg_count else "MISMATCH"
        print(f"  {table:35s} SQLite={sqlite_count:>10,}  PG={pg_count:>10,}  {status}")
        if sqlite_count != pg_count:
            errors.append(f"{table}: SQLite={sqlite_count}, PG={pg_count}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite → PostgreSQL")
    parser.add_argument("--source", type=str, default=str(settings.sqlite_source))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Target: {settings.database_url_sync}")
    print()

    sqlite_conn = sqlite3.connect(args.source)
    # Strip SQLAlchemy dialect prefix for raw psycopg
    pg_dsn = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")
    pg_conn = psycopg.connect(pg_dsn)

    if args.verify_only:
        print("=== Verification ===")
        errors = verify_counts(sqlite_conn, pg_conn)
        if errors:
            print(f"\n{len(errors)} mismatches found!")
        else:
            print("\nAll counts match.")
        sqlite_conn.close()
        pg_conn.close()
        return

    # Truncate all tables first (in reverse FK order)
    print("=== Truncating existing data ===")
    pg_conn.execute(
        "TRUNCATE " + ", ".join(reversed(TABLES_ORDERED)) + " CASCADE"
    )
    pg_conn.commit()
    print("  Done\n")

    # Migrate each table
    print("=== Migrating data ===")
    t0 = time.time()
    total_rows = 0
    for table in TABLES_ORDERED:
        try:
            sqlite_conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        except sqlite3.OperationalError:
            print(f"  {table:35s} SKIP (not in SQLite)")
            continue

        t1 = time.time()
        count = migrate_table(sqlite_conn, pg_conn, table, args.batch_size)
        elapsed = time.time() - t1
        total_rows += count
        print(f"  {table:35s} {count:>10,} rows  ({elapsed:.1f}s)")

        # Reset sequence
        reset_sequence(pg_conn, table)

    elapsed_total = time.time() - t0
    print(f"\nTotal: {total_rows:,} rows in {elapsed_total:.1f}s\n")

    # Verify
    print("=== Verification ===")
    errors = verify_counts(sqlite_conn, pg_conn)
    if errors:
        print(f"\n{len(errors)} MISMATCHES!")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll counts match. Migration successful.")

    sqlite_conn.close()
    pg_conn.close()


if __name__ == "__main__":
    main()
