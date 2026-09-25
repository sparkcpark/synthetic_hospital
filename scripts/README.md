# Scripts

Analysis, audit, and export utilities used to build and validate the benchmark.

## Important: these operate on the full benchmark database

With one exception (below), the scripts here query the **full benchmark database**
— the SQLite DB produced by the ETL pipeline, with tables such as `diagnoses`,
`clinical_findings`, `board_questions`, `benchmark_ground_truth`,
`longitudinal_patients`, and `diagnosis_relations`.

That database is **not shipped** in this repository (see the top-level release
notes). The released dataset, `patient_profiles.db`, is a *stripped* two-table
subset (`patients`, `encounters`) and does **not** contain the tables these
scripts need — pointing them at `patient_profiles.db` will fail with
"no such table".

To run these scripts, first rebuild the full benchmark with the ETL pipeline
(`python -m etl.main --stage 1 … 12`), which writes `benchmark_v1.3.db` (repo root)
(the default path most scripts use, overridable via `--db`). They are included
as **reproducibility / methodology artifacts**: they document how the released
data was constructed and audited.

## Requirements by backend

- **SQLite (full benchmark DB)** — most scripts. Default `benchmark_v1.3.db` (repo root).
- **Postgres (epic_sim)** — `build_dataset_tiers.py`, `matcher_diagnostic.py`,
  `sample_summaries.py`, `sample_all_summaries.py` connect to the running
  `epic_sim` database via `DATABASE_URL`.
