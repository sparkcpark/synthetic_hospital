# Synthetic Hospital — EHR simulator + evaluation harness + RL environment.
#
# Build targets (docker build --target <name>):
#   app        (default) code + dependencies; the benchmark database is bind-mounted at run time
#   with-data  app + benchmark_v1.3.db baked in at /data (self-contained; what Harbor tasks use)
#   allinone   with-data + PostgreSQL + Redis inside the same image, started by docker/allinone.sh;
#              runs as an unprivileged user with no external services (Apptainer / HPC / plain docker run)
#
#   docker compose up -d                                     # app + postgres + redis (built from source)
#   docker pull ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone           # published images (scripts/publish_images.sh)
#   docker run --rm -p 8000:8000 ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone
#   docker build --target with-data -t ghcr.io/sparkcpark/synthetic-hospital:1.3-data .     # local builds under the published names
#   docker build --target allinone  -t ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone .

FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Core dependencies only: the SapBERT/torch extras in requirements.txt are
# needed solely to rebuild the semantic search index, which the release does
# not ship. Every import of those packages in epic_sim/ and eval/ is lazy.
COPY requirements-core.txt ./
RUN pip install -r requirements-core.txt

COPY epic_sim ./epic_sim
COPY eval ./eval
COPY etl ./etl
COPY scripts ./scripts
COPY ui ./ui
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/results /app/exports /app/data/ontology /data

# Defaults for a container next to the compose `postgres` and `redis` services.
# Every value can be overridden from docker-compose.yml or the environment.
ENV EPIC_SIM_DATABASE_URL=postgresql+asyncpg://epic_sim:dev_password@postgres:5432/epic_sim \
    EPIC_SIM_DATABASE_URL_SYNC=postgresql+psycopg://epic_sim:dev_password@postgres:5432/epic_sim \
    EPIC_SIM_REDIS_URL=redis://redis:6379/0 \
    DATABASE_URL=postgresql://epic_sim:dev_password@postgres:5432/epic_sim \
    EPIC_SIM_SQLITE_SOURCE=/data/benchmark_v1.3.db \
    EPIC_SIM_ONTOLOGY_DIR=/app/data/ontology \
    EPIC_SIM_LOAD_TERMINOLOGY=auto \
    EPIC_SIM_DOWNLOAD_ICD10=auto \
    EPIC_SIM_FORCE_RELOAD=0 \
    EPIC_SIM_SETUP_SANDBOX_ROLE=1 \
    EPIC_SIM_SCORER_TOKEN=dev-scorer-token-change-in-production

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=900s --retries=12 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

ENTRYPOINT ["entrypoint.sh"]
CMD ["serve"]


# ---------------------------------------------------------------------------
# with-data: the released benchmark database baked in (no bind mount needed)
# ---------------------------------------------------------------------------
FROM app AS with-data
COPY benchmark_v1.3.db /data/benchmark_v1.3.db


# ---------------------------------------------------------------------------
# allinone: PostgreSQL + Redis inside the image, unprivileged user, one process
# tree. State lives under $SH_STATE_DIR (default /var/lib/synthetic-hospital),
# which must be writable: a volume, a bind mount, or an Apptainer --bind.
# ---------------------------------------------------------------------------
FROM with-data AS allinone
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql postgresql-contrib redis-server \
    && rm -rf /var/lib/apt/lists/* \
    && PG_BIN="$(ls -d /usr/lib/postgresql/*/bin | head -1)" && ln -s "$PG_BIN" /usr/lib/postgresql/bin \
    && useradd --create-home --uid 1000 --shell /bin/bash sh \
    && mkdir -p /var/lib/synthetic-hospital /var/run/postgresql \
    && chown -R sh:sh /app /var/lib/synthetic-hospital /var/run/postgresql
COPY docker/allinone.sh /usr/local/bin/allinone.sh
RUN chmod +x /usr/local/bin/allinone.sh
USER sh
ENV EPIC_SIM_DATABASE_URL=postgresql+asyncpg://epic_sim:dev_password@127.0.0.1:5432/epic_sim \
    EPIC_SIM_DATABASE_URL_SYNC=postgresql+psycopg://epic_sim:dev_password@127.0.0.1:5432/epic_sim \
    EPIC_SIM_REDIS_URL=redis://127.0.0.1:6379/0 \
    DATABASE_URL=postgresql://epic_sim:dev_password@127.0.0.1:5432/epic_sim \
    SH_STATE_DIR=/var/lib/synthetic-hospital \
    PATH=/usr/lib/postgresql/bin:$PATH
VOLUME ["/var/lib/synthetic-hospital"]
ENTRYPOINT ["allinone.sh"]
CMD ["serve"]
