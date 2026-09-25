<p align="center">
  <img src="synthetic_hospital_logo.svg" alt="Synthetic Hospital emblem: a teal cross with a pulse line" width="160">
</p>
<h1 align="center">
      Synthetic Hospital:<br>
      A Medical Benchmark & EHR Simulation Platform</h1>

Converts USMLE-style medical education source content into a ground-truth benchmark database of synthetic longitudinal patient records, served through an Epic-faithful EHR simulation platform (FHIR R4 + OAuth2/RBAC), for evaluating clinical AI agents on four longitudinal-chart tasks: patient diagnosis (problem-list reconstruction), context summarization (with whole-patient, current-visit, and specialty-conditioned variants), evidence retrieval, and imaging indication. The benchmark ships with a labelled training split so it can also be used as a verifiable-reward environment.

> **Paper:** Park, Chen, Dettmers. _Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark._ Preprint, 2026. arXiv link: _to be added_.
> If you use this work, please cite it (see [CITATION.cff](CITATION.cff)).

## Architecture

```
source content                        (not distributed — see Data Setup)
      │
      ▼
  etl/  (stages s01–s10: ingest → classify → extract → fact cards →
         ontology mapping → relationships → EHR sections → patients →
         encounters → ground truth)          [SQLite: data/benchmark.db]
      │
      ▼
  epic_sim/  FastAPI EHR simulator (FHIR R4, OAuth2, RBAC, Postgres)
      │
      ▼
  eval/  task runners, scoring, semantic matching, reports
```

## Released Dataset (v1.3)

| File | Contents |
|---|---|
| `patient_profiles.db`, `patient_profiles.jsonl` | 1,268 synthetic longitudinal patients and their 5,602 clinical notes (SQLite and JSON Lines) |
| `benchmark_v1.3.db` | The benchmark database: chart sections, ground truth for the four tasks, section-level relevance judgments, imaging orders, the ontology-grounded knowledge graph, and the public / held-out / training split labels |

See **[DATA_CARD.md](DATA_CARD.md)** for schemas, provenance, and what was excluded. The data is fully synthetic and not for clinical use.

**Splits.** Patients are partitioned at the patient level: `public` (200 patients; the reported benchmark), `heldout` (268 patients; private evaluation set, reference labels are shipped so you can score locally but should not be trained on), and `train` (800 patients; released for training, including reinforcement learning with the graph-derived rewards). No patient shares source material with any other.

## Repository Map

| Path | Purpose |
|---|---|
| `etl/` | Source content → benchmark DB pipeline. `parsers/`, `stages/` (s01–s10), `ontology/` (SNOMED, ICD-10, LOINC, SapBERT embeddings), `deck_profiles/` and `pdf_profiles/` (declarative source descriptions; see `stages/EXTENDING.md`) |
| `epic_sim/` | EHR simulation platform: FastAPI app, FHIR R4 resources, OAuth2 + RBAC, Alembic migrations, SQLite→Postgres migration scripts, tests |
| `eval/` | Evaluation framework: CLI, task definitions (`tasks/`), agentic harness (`agents/`), scoring, ICD-10 validation, semantic matching, reporting |
| `scripts/` | Auditing and analysis scripts (ontology mapping QA, dataset tiers, ablations, paper tables, Harbor export, image publishing) |
| `docker/`, `Dockerfile`, `docker-compose.yml` | Container build and entrypoints (compose stack, self-contained and all-in-one images) |
| `harbor/`, `apptainer/` | Harbor task templates and the Apptainer definition |
| `benchmark_v1.3.db`, `patient_profiles.db`, `patient_profiles.jsonl` | The released data (see `DATA_CARD.md`) |

## Quickstart (Docker, one command)

The compose stack builds the simulator image, starts Postgres 16 and Redis 7, loads
`benchmark_v1.3.db` into Postgres on first boot, and serves the EHR simulator:

```bash
docker compose up -d                 # first boot loads the database (~10 s), then serves http://localhost:8000/docs
docker compose logs -f app           # watch the load; "[entrypoint] database ready: 1268 patients" means done
```

Everything else runs inside the same image, against the loaded database:

```bash
docker compose run --rm app python -m eval.cli --help
docker compose run --rm app python -m eval.cli run --task patient_diagnosis --model <model> --split public --strategy cot
docker compose run --rm app pytest eval/tests
```

Set `OPENROUTER_API_KEY` in your shell (or a `.env` file) before running models; results are written to
`./results` on the host. Host ports are configurable if the defaults are taken:
`POSTGRES_PORT=55432 REDIS_PORT=56379 APP_PORT=58000 docker compose up -d`.

The load is idempotent: restarting the stack skips it when patients are already present
(`EPIC_SIM_FORCE_RELOAD=1` reloads from the SQLite file). See `docker/entrypoint.sh` for the full list
of environment switches.

### Terminology (ICD-10-CM, SNOMED CT, LOINC)

The FHIR terminology endpoints (`$lookup`, `$expand`, `$validate-code`) and the trigram code search read
from a `terminology_codes` table that the release cannot pre-fill: SNOMED CT needs a UMLS licence and
LOINC a free registration. Everything else in the benchmark works without them.

- **ICD-10-CM** is public domain and is fetched from CMS automatically on first boot (set
  `EPIC_SIM_DOWNLOAD_ICD10=0` to disable, or drop the CMS "code descriptions in tabular order" `.txt`
  or `.zip` into the ontology directory to load offline).
- **SNOMED CT and LOINC**: download your own copies, unpack them anywhere under a directory, mount it
  at `/app/data/ontology` (uncomment the volume in `docker-compose.yml`), and restart the stack. The
  entrypoint discovers the files by pattern (`sct2_Concept_Snapshot*.txt` with its
  `sct2_Description_Snapshot-en*.txt`; `LoincTable/Loinc.csv`), so any release version and either the
  original folder layout or flat files work, and loads only the systems not yet present. Expect a few
  minutes for SNOMED CT. Nothing else needs to change; the terminology tests stop skipping once the
  systems are loaded.
- Manual install: `python -m epic_sim.migrate.load_terminology --auto --download-icd10` does the same
  from the host; `--list` shows what was found.

## Quickstart (manual)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Infrastructure (Postgres 16 + Redis 7)
docker compose up -d

# 3. EHR simulator
uvicorn epic_sim.app.main:app --reload   # http://localhost:8000/docs

# 4. Run tests
pytest epic_sim/tests
pytest eval/tests
```

## Data Setup

The benchmark database `benchmark_v1.3.db` **is distributed** (see Released Dataset) and is all you need to run the evaluations: load it into the simulator with the commands below. Rebuilding it from scratch requires source material that cannot be redistributed:

1. **Source materials**: the ETL is source-agnostic. Describe your own Anki decks or PDFs with a small YAML profile (`etl/deck_profiles/`, `etl/pdf_profiles/`; see `etl/stages/EXTENDING.md`) and place the files under `apkg/` (paths in `etl/config.py`). The material the paper used is not included.
2. **Ontologies** (place in `data/ontology/`):
   - SNOMED CT US Edition + Transitive Closure — [UMLS license required](https://www.nlm.nih.gov/healthit/snomedct/)
   - LOINC — [free registration](https://loinc.org/downloads/)
   - ICD-10-CM 2025 tabular XML — [CDC/CMS, public domain](https://www.cms.gov/medicare/coding-billing/icd-10-codes)
3. **SapBERT embeddings**: generated locally by `etl/ontology/sapbert_embedder.py` (no download needed).

Then run the ETL:

```bash
python -m etl.main --stage 1    # repeat for stages 1..12, in order
```

And load into the simulator:

```bash
alembic -c epic_sim/alembic/alembic.ini upgrade head   # creates the schema (run first on a fresh database)
python -m epic_sim.migrate.sqlite_to_pg                 # loads benchmark_v1.3.db (repo root) into Postgres
python -m epic_sim.migrate.load_terminology             # optional; needs the ontology files above
```

Or let the container do all of this: `docker compose up -d` (Quickstart above).

## Running Evaluations

Models are accessed via OpenRouter; set `OPENROUTER_API_KEY`. The registry entries named `kimi-2.5*` and
`glm-5-agent` were served through a private Anthropic-compatible gateway for the paper; they read
`SH_LLM_GATEWAY_URL` (default `http://localhost:8080`) and need your own gateway, whereas the `*-or`
entries reach the same models through OpenRouter.

```bash
python -m eval.cli run --task patient_diagnosis --model <model> --split public --strategy cot
python -m eval.cli score --run-id <id>
python -m eval.cli report --split public --format csv
python -m eval.cli matrix --task patient_diagnosis --split public   # all models
```

Tasks: `patient_diagnosis`, `context_summarization`, `evidence_retrieval`, `imaging_indication`. Splits: `public`, `heldout`, `train`. See `python -m eval.cli --help`.

Locked prompting strategies used in the paper: CoT (patient diagnosis), ontology-grounded structured (whole-patient summarization), zero-shot (retrieval, specialty-conditioned summarization), few-shot (imaging). `eval.cli score` reports the paper's primary metrics directly: patient diagnosis under the chart-neutral rule (`weighted_problem_list_f1_neutral`; sets from `eval/chart_neutral.py`), retrieval over chart sections only (`precision_5`, `ndcg_10`), summarization by must-include finding recall (`clinical_f1`), and the imaging clinical question by ontology-grounded concept F1 (`clinical_question_concept_f1`; extractor in `eval/imaging_concepts.py`, built from the loaded database with no external ontology files).

`context_summarization` has three variants selected via `--granularity`: whole-patient (default), `encounter` (current-visit, point-in-time), and `specialty` (specialty-conditioned; optionally scope to a frozen cohort with `--tier small|medium|large`). Example:

```bash
python -m eval.cli run --task context_summarization --granularity specialty --model <model> --split public
```

Agentic (tool-use) evaluation against the live EHR simulator uses `eval/agents/` with the read-only bash sandbox defined in `docker-compose.override.yml`. The `bash_readonly` role that the sandbox uses is created automatically by the container entrypoint (or by `scripts/setup_bash_readonly.sql` in a manual install).

## Scoring Endpoint (rewards)

The simulator exposes the paper's scorer over HTTP so that an external trainer can obtain a reward
for any submission without importing the evaluation code. It is guarded by a shared secret
(`EPIC_SIM_SCORER_TOKEN`, sent as `X-Scorer-Token`) that the policy under training must not hold; the
labels themselves are never returned.

```bash
TOKEN=dev-scorer-token-change-in-production        # set EPIC_SIM_SCORER_TOKEN in production
curl -s -H "X-Scorer-Token: $TOKEN" "http://localhost:8000/score/tasks"
curl -s -H "X-Scorer-Token: $TOKEN" "http://localhost:8000/score/instances?task=patient_diagnosis&split=train&limit=3"
curl -s -H "X-Scorer-Token: $TOKEN" -H "Content-Type: application/json" http://localhost:8000/score \
  -d '{"gt_id": 123, "prediction": {"active_diagnoses": [{"icd10": "E11.01", "acuity": "acute"}], "chronic_conditions": []}}'
```

The response carries `reward` in [0, 1], the metric it was taken from, and every metric the scorer
computed. Rewards are the tasks' primary metrics: severity-weighted, chart-neutral F1 for patient
diagnosis; must-include finding recall for summarization (conditioned F1 or abstention accuracy for the
specialty variant); precision at 5 over chart sections for retrieval (NDCG at 10 is also returned); and
ontology-grounded concept F1 of the inferred clinical question for imaging (the paper's metric; token-level
F1 is also returned). Submissions use the same JSON schemas the
agent tools accept (`submit_diagnosis`, `submit_summary`, `submit_rankings`, `submit_pre_read`); a malformed
or empty submission scores 0 rather than raising. `POST /score/batch` scores up to 256 items per call. The
same function is available in-process as `eval.score_one.score_submission`.

## RL Environment (reset and step)

The simulator can be driven as a reinforcement-learning environment. An episode is one benchmark
instance played as the paper's tool-use task: the policy gets the paper's agent system prompt, the
patient assignment, and the 13 EHR tools as function schemas; it acts by calling tools; the task's
submit tool ends the episode and returns the reward from the scoring endpoint above. Episode state
lives in Redis (started by `docker compose`).

```python
from eval.env_client import SyntheticHospitalEnv

env = SyntheticHospitalEnv("http://localhost:8000", scorer_token=TOKEN)
obs = env.reset(task="patient_diagnosis", split="train", seed=0)   # or reset(gt_id=...)
# obs.instructions (system prompt), obs.intro (first user message), obs.tools (function schemas)
o, reward, done, info = env.step("open_chart", {"patient_id": obs.patient_id})
o, reward, done, info = env.step(obs.submit_tool, {"active_diagnoses": [...], "chronic_conditions": []})
# done is True, reward in [0, 1], info["metrics"] holds every metric
```

The same four calls are plain HTTP (`POST /env/reset`, `POST /env/step`, `GET /env/state/{id}`,
`POST /env/close`, all with `X-Scorer-Token`), so any language works. `scripts/env_demo.py` runs a
scripted policy end to end and is the template to replace with a model. What the server enforces:

- **Hidden outcome.** Assessment and plan sections are stripped from every observation, as in the
  paper's baselines, and the problem list returned by `open_chart` / `view_problem_list` is the chart's
  documented history (the profile's chronic conditions) rather than the simulator's graph-derived
  encounter diagnoses, which are the patient-diagnosis reference itself. Imaging-indication episodes
  cannot observe encounters after the imaging order.
- **Budget.** Default 40 actions (`budget=` on reset). Once spent, only the submit tool is accepted,
  mirroring the paper's forced final turn; unknown tools and bad arguments come back as error
  observations and still cost a step.
- **Reward.** Computed once at submission with the paper's primary metrics; a malformed or missing
  submission scores 0. Labels are never returned, and the policy holds no credential: the harness
  keeps the scorer token.
- **Reproducibility.** `reset(task=, split=, seed=)` samples deterministically; `env.instances()` lists
  every instance id so you can build your own curriculum over the 7,619 training instances.

## Container images, Harbor tasks, and Apptainer

The Dockerfile has three targets:

| Target | Build | Contents |
|---|---|---|
| `app` (default, used by compose) | `docker compose build` | code and dependencies; the database is bind-mounted |
| `with-data` | `docker pull ghcr.io/sparkcpark/synthetic-hospital:1.3-data` | plus `benchmark_v1.3.db` baked in; self-contained, used by Harbor tasks |
| `allinone` | `docker pull ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone` | plus PostgreSQL and Redis inside the image, run as an unprivileged user with no other services |

The two data-bearing images are published to GitHub Container Registry; `scripts/publish_images.sh`
builds and pushes them (maintainers). To build them locally instead, use the same tags:
`docker build --target with-data -t ghcr.io/sparkcpark/synthetic-hospital:1.3-data .`

**Harbor.** `scripts/harbor_export.py` turns benchmark instances into
[Harbor](https://harborframework.com) tasks, one directory per instance, with the paper's agent
prompt as `instruction.md`, a compose environment (simulator with the episode pre-started, Postgres and
Redis on a network the agent cannot reach), an `sh-agent` command line for the agent, a verifier that
writes the episode reward to `/logs/verifier/reward.json`, and an oracle solution that scores 1.0. See
`harbor/README.md`:

```bash
python scripts/harbor_export.py --out harbor_tasks --task patient_diagnosis --split train --limit 50
uvx harbor run -p harbor_tasks -a oracle                  # self-check, then swap in any Harbor agent
```

**Apptainer.** For clusters without Docker, `apptainer/synthetic_hospital.def` converts the all-in-one
image into a `.sif` that starts everything as the calling user with a bound state directory; see
`apptainer/README.md`. The plain `docker run --rm -p 8000:8000 ghcr.io/sparkcpark/synthetic-hospital:1.3-allinone` form
works anywhere Docker does and needs no compose file.

## Reproducing Paper Results

1. Load `benchmark_v1.3.db` into the simulator (Data Setup above). The ETL that built it is deterministic for non-LLM stages; LLM-assisted stages record model + prompt provenance.
2. Run the eval matrix for each task/model pair reported in the paper.
3. Ablations: `scripts/ablation_relevance.py`, `scripts/run_specialty_sweep.py`.

## Known issues and caveats

- **The simulator's problem-list tools reveal the patient-diagnosis labels.** `view_problem_list` and the
  `active_problems` field of `open_chart` (and the FHIR Condition resource) return the graph-derived
  correct diagnoses of the patient's source questions with ICD-10 codes, which is the patient-diagnosis
  reference itself. The `/env` endpoints and the Harbor tasks replace them with the chart's documented
  history and hide assessment/plan sections server-side. The `eval/agents` harness used for the paper's
  agentic runs does **not**: it hides assessment/plan client-side but leaves the problem list, so
  agentic patient-diagnosis scores obtained through that harness are inflated. Use the environment
  endpoints for any new agent evaluation.
- Later encounter notes carry an "Active Problem List" of earlier encounters' diagnoses by construction
  (that is the chart); only the current encounter's diagnosis and the coded list are withheld.
- The imaging-indication reference question is LLM-authored (anchored to the graph diagnosis); the other
  three tasks' labels are deterministic functions of the graph.
- The terminology tables ship empty (licensing); see Terminology above.

## License & Citation

Code: [LICENSE](LICENSE). The released data files are fully synthetic and redistributable; the source material the pipeline was built from is not included.

`benchmark_v1.3.db` (88 MB) is committed directly (under GitHub's 100 MB file limit), so a plain `git clone` brings the data with no Git LFS setup; it is also attached to each GitHub release as a downloadable asset.
