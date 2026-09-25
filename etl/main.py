"""CLI entry point for the ETL pipeline.

(spec §9): python -m etl.main --stage 1..12
"""

import argparse
import json
import sys

from etl.config import DB_PATH
from etl.db import create_schema, get_connection
from etl.utils.logging import get_logger

log = get_logger("etl.main")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Medical APKG-to-Benchmark ETL Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=range(1, 13),
        help="Stage number to run (1-12)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DB_PATH),
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    args = parser.parse_args()

    conn = get_connection(args.db)
    create_schema(conn)

    try:
        if args.stage == 1:
            from etl.stages.s01_ingest import run

            summary = run(conn)
            print(json.dumps(summary, indent=2))
        elif args.stage == 2:
            from etl.stages.s02_classify import run

            summary = run(conn)
            print(json.dumps(summary, indent=2))
        elif args.stage == 3:
            from etl.stages.s03_extract_board import run

            summary = run(conn)
            print(json.dumps(summary, indent=2))
        elif args.stage == 4:
            from etl.stages.s04_fact_cards import run

            summary = run(conn)
            print(json.dumps(summary, indent=2))
        elif args.stage == 5:
            from etl.stages.s05_ontology import run

            summary = run(conn)
            print(json.dumps(summary, indent=2))
        elif args.stage == 6:
            from etl.stages.s06_relationships import run

            summary = run(conn)
            print(json.dumps(summary, indent=2))
        else:
            log.error("Stage %d is not yet implemented", args.stage)
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
