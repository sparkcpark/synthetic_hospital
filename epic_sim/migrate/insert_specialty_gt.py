"""Selective, non-destructive insert of specialty-conditioned summarization GT
from the sqlite copy into the live Postgres benchmark_ground_truth.

Unlike the full sqlite_to_pg reload, this touches NOTHING else: it deletes and
re-inserts only the variant='specialty_conditioned' rows (idempotent), leaving the
existing benchmark, results, and diagnoses intact. New gt_ids are auto-assigned by
the sequence (the copy shares gt_ids with PG, so we never carry them over). The
patient/EHR tables the eval reads are already in PG, and summarization scoring is
tier-based, so this is all the live task needs.

Usage:
    python -m epic_sim.migrate.insert_specialty_gt [--source PATH] [--sampled-only]
"""
import argparse
import json
import sqlite3

import psycopg

from epic_sim.app.config import settings

SELECT = (
    "SELECT task, granularity, patient_id, ground_truth, difficulty, split "
    "FROM benchmark_ground_truth "
    "WHERE task='context_summarization' AND granularity='patient' "
    "AND json_extract(ground_truth,'$.variant')='specialty_conditioned'"
)
DELETE = (
    "DELETE FROM benchmark_ground_truth WHERE task='context_summarization' "
    "AND granularity='patient' AND ground_truth->>'variant'='specialty_conditioned'"
)
INSERT = (
    "INSERT INTO benchmark_ground_truth "
    "(task, granularity, patient_id, ground_truth, difficulty, split) "
    "VALUES (%s, %s, %s, %s, %s, %s)"
)


def main():
    ap = argparse.ArgumentParser(description="Selective insert of specialty-conditioned GT into PG")
    ap.add_argument("--source", default="data/benchmark_v1.2_copy.db")
    ap.add_argument("--sampled-only", action="store_true",
                    help="insert only rows with a split assigned (the Phase G eval set)")
    args = ap.parse_args()

    q = SELECT + (" AND split IS NOT NULL" if args.sampled_only else "")
    sq = sqlite3.connect(args.source)
    rows = sq.execute(q).fetchall()
    sq.close()
    print(f"source: {args.source} -> {len(rows)} specialty-conditioned rows"
          f"{' (sampled only)' if args.sampled_only else ''}")

    dsn = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")
    pg = psycopg.connect(dsn)

    # sanity: which split enum values does PG accept? (fail fast if val/test invalid)
    valid = {r[0] for r in pg.execute(
        "SELECT unnest(enum_range(NULL::split_type))").fetchall()} if _has_enum(pg) else None

    with pg.cursor() as cur:
        # clear predictions referencing the specialty rows we're about to replace
        # (stale once the GT is regenerated); leaves all other results intact.
        cur.execute(
            "DELETE FROM evaluation_predictions WHERE gt_id IN "
            "(SELECT gt_id FROM benchmark_ground_truth WHERE task='context_summarization' "
            "AND granularity='patient' AND ground_truth->>'variant'='specialty_conditioned')")
        cleared_preds = cur.rowcount
        cur.execute(DELETE)
        deleted = cur.rowcount
        if cleared_preds:
            print(f"  cleared {cleared_preds} stale specialty predictions first")
        for task, gran, pid, gtj, diff, split in rows:
            if valid is not None and split is not None and split not in valid:
                raise SystemExit(f"split '{split}' not in PG enum {valid}")
            cur.execute(INSERT, (task, gran, pid, psycopg.types.json.Json(json.loads(gtj)),
                                 diff, split))
    pg.commit()

    cnt = pg.execute(
        "SELECT COALESCE(split::text,'(null)'), COUNT(*) FROM benchmark_ground_truth "
        "WHERE task='context_summarization' AND granularity='patient' "
        "AND ground_truth->>'variant'='specialty_conditioned' GROUP BY 1 ORDER BY 1").fetchall()
    print(f"deleted {deleted} prior specialty rows; inserted {len(rows)}")
    print("PG specialty rows by split:", cnt)
    pg.close()


def _has_enum(pg):
    try:
        return pg.execute("SELECT 1 FROM pg_type WHERE typname='split_type'").fetchone() is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
