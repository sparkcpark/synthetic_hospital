# Running Synthetic Hospital with Apptainer

For clusters without Docker. The all-in-one image carries the simulator, the
benchmark database, PostgreSQL and Redis, and starts everything as the calling
user; nothing needs root and no other services are required.

## Build

The definition file bootstraps from the published all-in-one image, so no Docker is needed:

```bash
apptainer build synthetic_hospital.sif apptainer/synthetic_hospital.def
```

To build from a locally built Docker image instead (for example while developing), build it under
the published name and use the local variant of the definition:

```bash
docker build --target allinone -t ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone .
apptainer build synthetic_hospital.sif apptainer/synthetic_hospital.local.def
```

Without Docker on either machine, `docker save` the image elsewhere and bootstrap from the archive
(`Bootstrap: docker-archive`).

## Run

The container writes its database cluster under `/var/lib/synthetic-hospital`, which must be
writable and has room for the loaded benchmark (a few hundred MB). Bind a directory:

```bash
mkdir -p $HOME/sh-state
apptainer run --bind $HOME/sh-state:/var/lib/synthetic-hospital synthetic_hospital.sif
# ... API on http://localhost:8000/docs; first start loads the database (about 10 s)
```

As a background instance:

```bash
apptainer instance start --bind $HOME/sh-state:/var/lib/synthetic-hospital synthetic_hospital.sif sh
apptainer instance list
apptainer instance stop sh
```

Any command runs after the services are up, e.g. tests or an evaluation:

```bash
apptainer run --bind $HOME/sh-state:/var/lib/synthetic-hospital synthetic_hospital.sif pytest eval/tests
apptainer run --bind $HOME/sh-state:/var/lib/synthetic-hospital synthetic_hospital.sif \
    python -m eval.cli run --task patient_diagnosis --model <model> --split public --strategy cot
```

## Notes

- Ports: the API listens on 8000, PostgreSQL on 127.0.0.1:5432 and Redis on 127.0.0.1:6379 inside the
  container's network namespace, which Apptainer shares with the host by default. Override with
  `PGPORT` and `SH_REDIS_PORT` if those ports are taken on a shared node.
- `--writable-tmpfs` works only if the session directory limit in `apptainer.conf` is raised well above
  the default 64 MiB; a bind mount is simpler.
- Terminology files (SNOMED CT, LOINC) can be bound at `/app/data/ontology` and are loaded on the next
  start, exactly as with Docker (see the main README, Terminology).
- The scorer token defaults to the development value; set `EPIC_SIM_SCORER_TOKEN` with
  `--env` for anything beyond local use.
