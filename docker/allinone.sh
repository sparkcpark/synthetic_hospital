#!/bin/sh
# All-in-one entrypoint: start PostgreSQL and Redis inside the container as the
# current (unprivileged) user, then hand over to the regular entrypoint.
#
#   docker run --rm -p 8000:8000 synthetic-hospital:1.3-allinone
#   docker run --rm -p 8000:8000 -v sh-state:/var/lib/synthetic-hospital synthetic-hospital:1.3-allinone
#   apptainer run --writable-tmpfs synthetic_hospital.sif          (see apptainer/README.md)
#
# State (Postgres cluster, Redis dump) lives under $SH_STATE_DIR, default
# /var/lib/synthetic-hospital. It must be writable. With Apptainer use
# --writable-tmpfs (ephemeral) or --bind /some/host/dir:/var/lib/synthetic-hospital.
set -eu

STATE="${SH_STATE_DIR:-/var/lib/synthetic-hospital}"
export PGPASSWORD="${PGPASSWORD:-dev_password}"   # never prompt, whatever the auth method
PGDATA="${PGDATA:-$STATE/pgdata}"
PGPORT="${PGPORT:-5432}"
REDIS_PORT="${SH_REDIS_PORT:-6379}"
SOCKDIR="${SH_SOCKET_DIR:-/tmp/synthetic-hospital}"
export PGDATA

log() { printf '[allinone] %s\n' "$*"; }

mkdir -p "$STATE" "$SOCKDIR" "$STATE/redis"

# --- PostgreSQL -------------------------------------------------------------
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    log "initialising PostgreSQL cluster in $PGDATA"
    # trust on the local socket (used by this script), password over TCP (used by the app)
    initdb -D "$PGDATA" --username=epic_sim --pwfile=/dev/stdin --auth-local=trust --auth-host=scram-sha-256 --encoding=UTF8 >/dev/null <<EOF
dev_password
EOF
    # listen on loopback only; unix socket in a writable dir
    {
        echo "listen_addresses = '127.0.0.1'"
        echo "port = $PGPORT"
        echo "unix_socket_directories = '$SOCKDIR'"
        echo "shared_buffers = 256MB"
        echo "max_connections = 50"
        echo "fsync = off"                      # ephemeral by default; state dir is disposable
    } >> "$PGDATA/postgresql.conf"
fi
log "starting PostgreSQL on 127.0.0.1:$PGPORT"
pg_ctl -D "$PGDATA" -l "$STATE/postgres.log" -o "-p $PGPORT -k $SOCKDIR" -w start >/dev/null
# create the database once
if ! psql -h "$SOCKDIR" -p "$PGPORT" -U epic_sim -d postgres -Atc "select 1 from pg_database where datname='epic_sim'" | grep -q 1; then
    createdb -h "$SOCKDIR" -p "$PGPORT" -U epic_sim epic_sim
fi

# --- Redis ------------------------------------------------------------------
log "starting Redis on 127.0.0.1:$REDIS_PORT"
redis-server --port "$REDIS_PORT" --bind 127.0.0.1 --dir "$STATE/redis" --save "" --appendonly no \
    --daemonize yes --logfile "$STATE/redis.log" --pidfile "$STATE/redis.pid" >/dev/null

# stop both cleanly when the main process exits
cleanup() {
    log "stopping services"
    redis-cli -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
    pg_ctl -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# point the app at the local services unless the caller overrode them
export EPIC_SIM_DATABASE_URL="${EPIC_SIM_DATABASE_URL:-postgresql+asyncpg://epic_sim:dev_password@127.0.0.1:$PGPORT/epic_sim}"
export EPIC_SIM_DATABASE_URL_SYNC="${EPIC_SIM_DATABASE_URL_SYNC:-postgresql+psycopg://epic_sim:dev_password@127.0.0.1:$PGPORT/epic_sim}"
export DATABASE_URL="${DATABASE_URL:-postgresql://epic_sim:dev_password@127.0.0.1:$PGPORT/epic_sim}"
export EPIC_SIM_REDIS_URL="${EPIC_SIM_REDIS_URL:-redis://127.0.0.1:$REDIS_PORT/0}"

# run the regular entrypoint in the foreground so the trap fires when it exits
entrypoint.sh "$@"
