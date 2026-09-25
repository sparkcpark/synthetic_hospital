#!/bin/sh
# Container entrypoint for the Synthetic Hospital simulator.
#
# Usage (as the image's ENTRYPOINT):
#   entrypoint.sh serve          wait for Postgres, load data on first boot, run the API   (default)
#   entrypoint.sh load           wait for Postgres, load data, exit
#   entrypoint.sh <command...>   run any command after the database is ready
#                                (e.g. `python -m eval.cli run ...`, `pytest eval/tests`)
#
# First-boot loading, in order:
#   1. alembic upgrade head            creates or upgrades the schema (idempotent)
#   2. sqlite_to_pg                    loads $EPIC_SIM_SQLITE_SOURCE — only when the
#                                      database holds no patients, or EPIC_SIM_FORCE_RELOAD=1
#   3. load_terminology --auto         every boot; loads mounted ontology files not yet in the table
#   4. setup_bash_readonly.sql         read-only role for the bash-agent sandbox (idempotent)
#
# Environment (all have defaults set in the Dockerfile):
#   EPIC_SIM_DATABASE_URL_SYNC   postgresql+psycopg://user:pass@host:port/db  (migrator, alembic)
#   EPIC_SIM_SQLITE_SOURCE       path of the released benchmark database inside the container
#   EPIC_SIM_FORCE_RELOAD        1 = truncate and reload even if data is present
#   EPIC_SIM_LOAD_TERMINOLOGY    auto (load mounted ontology files not yet in the table, every boot)
#                                | 1 (reload all mounted files) | 0 (skip)
#   EPIC_SIM_DOWNLOAD_ICD10      auto (fetch the public-domain CMS ICD-10-CM release when no file is mounted) | 0
#   EPIC_SIM_ONTOLOGY_DIR        where ICD-10-CM / SNOMED CT / LOINC files are mounted (any release version)
#   EPIC_SIM_SETUP_SANDBOX_ROLE  1 = create the bash_readonly role (default) | 0 = skip
#   EPIC_SIM_SKIP_LOAD           1 = skip all loading steps (serve against an already-loaded database)
set -eu

cd /app

log() { printf '[entrypoint] %s\n' "$*"; }

# Plain DSN for psycopg (strip the SQLAlchemy dialect prefix).
PG_DSN=$(printf '%s' "${EPIC_SIM_DATABASE_URL_SYNC}" | sed 's#^postgresql+psycopg://#postgresql://#')
export PG_DSN

wait_for_postgres() {
    log "waiting for Postgres at ${PG_DSN%%@*}@…"
    python - <<'PY'
import os, sys, time
import psycopg
dsn = os.environ["PG_DSN"]
for attempt in range(90):
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last = exc
        time.sleep(2)
print(f"Postgres not reachable after 3 minutes: {last}", file=sys.stderr)
sys.exit(1)
PY
    log "Postgres is up"
}

patient_count() {
    python - <<'PY'
import os
import psycopg
with psycopg.connect(os.environ["PG_DSN"]) as conn:
    try:
        print(conn.execute("SELECT count(*) FROM longitudinal_patients").fetchone()[0])
    except Exception:  # table missing on a brand-new database
        conn.rollback()
        print(0)
PY
}

run_alembic() {
    log "applying schema migrations (alembic upgrade head)"
    alembic -c epic_sim/alembic/alembic.ini upgrade head
}

load_benchmark() {
    if [ ! -f "${EPIC_SIM_SQLITE_SOURCE}" ]; then
        log "ERROR: benchmark database not found at ${EPIC_SIM_SQLITE_SOURCE}"
        log "       mount it, e.g.  -v ./benchmark_v1.3.db:${EPIC_SIM_SQLITE_SOURCE}:ro"
        exit 1
    fi
    log "loading ${EPIC_SIM_SQLITE_SOURCE} into Postgres (this takes a few minutes)"
    python -m epic_sim.migrate.sqlite_to_pg --source "${EPIC_SIM_SQLITE_SOURCE}"
}

load_terminology() {
    # Runs on every boot: loads each system whose files are mounted under
    # $EPIC_SIM_ONTOLOGY_DIR and whose rows are not yet in the table, so adding
    # SNOMED CT / LOINC later and restarting is enough. ICD-10-CM is public domain
    # and is downloaded from CMS when absent (EPIC_SIM_DOWNLOAD_ICD10=0 disables).
    case "${EPIC_SIM_LOAD_TERMINOLOGY}" in
        0|no|false) log "terminology loading disabled"; return 0 ;;
        1|yes|true|all|force) extra="--force" ;;
        *) extra="" ;;
    esac
    dl=""
    case "${EPIC_SIM_DOWNLOAD_ICD10:-auto}" in
        0|no|false) ;;
        *) dl="--download-icd10" ;;
    esac
    log "terminology: checking ${EPIC_SIM_ONTOLOGY_DIR} (auto mode)"
    # shellcheck disable=SC2086
    python -m epic_sim.migrate.load_terminology --auto $extra $dl \
        | grep -E "^\[auto\]|^\[icd10\] (downloading|parsed|could not)|ERROR" | sed 's/^/[entrypoint]   /' || true
}

setup_sandbox_role() {
    [ "${EPIC_SIM_SETUP_SANDBOX_ROLE}" = "1" ] || return 0
    log "ensuring bash_readonly role for the agent sandbox"
    python - <<'PY'
import os
import psycopg
sql = open("scripts/setup_bash_readonly.sql", encoding="utf-8").read()
with psycopg.connect(os.environ["PG_DSN"], autocommit=True) as conn:
    conn.execute(sql)
PY
}

prepare_database() {
    wait_for_postgres
    if [ "${EPIC_SIM_SKIP_LOAD:-0}" = "1" ]; then
        log "EPIC_SIM_SKIP_LOAD=1: not touching the database"
        return 0
    fi
    run_alembic
    n=$(patient_count)
    if [ "$n" -gt 0 ] && [ "${EPIC_SIM_FORCE_RELOAD}" != "1" ]; then
        log "database already holds $n patients; skipping load (set EPIC_SIM_FORCE_RELOAD=1 to reload)"
    else
        load_benchmark
    fi
    load_terminology
    setup_sandbox_role
    log "database ready: $(patient_count) patients"
}

cmd="${1:-serve}"
case "$cmd" in
    serve)
        prepare_database
        shift
        log "starting EHR simulator on :8000"
        exec uvicorn epic_sim.app.main:app --host 0.0.0.0 --port 8000 "$@"
        ;;
    load)
        prepare_database
        ;;
    *)
        wait_for_postgres
        exec "$@"
        ;;
esac
