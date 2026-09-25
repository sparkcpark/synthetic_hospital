# Data Card — Synthetic Hospital v1.3

## Summary

1,268 synthetic patients and 5,602 clinical encounters, distributed as SQLite
(`patient_profiles.db`) and JSON Lines (`patient_profiles.jsonl`), plus the benchmark
database (`benchmark_v1.3.db`) holding the chart sections, ground truth, relevance
judgments, imaging orders, knowledge graph, and split labels. The data supports evaluation
of clinical AI on patient diagnosis, context summarization, evidence retrieval, and imaging
indication over multi-visit patient timelines, and training on the released training split.

The records are fully synthetic. They are generated from AI-generated question vignettes
combined with a knowledge graph derived from medical facts (see Provenance).

## Files


| File                     | Format     | Contents                                                          |
| ------------------------ | ---------- | ----------------------------------------------------------------- |
| `patient_profiles.db`    | SQLite 3   | Two tables: `patients`, `encounters`                              |
| `patient_profiles.jsonl` | JSON Lines | One patient object per line; encounters nested under `encounters` |
| `benchmark_v1.3.db`      | SQLite 3   | Benchmark tables (below); load into the simulator with `python -m epic_sim.migrate.sqlite_to_pg` |

## Schema



### `patients` (1,268 rows — one per synthetic patient)


| Column              | Description                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `patient_id`        | Unique patient identifier                                                                         |
| `profile`           | JSON blob: demographics, chronic conditions, surgical/family history, allergies, home medications |
| `age`               | Age at first encounter                                                                            |
| `sex`               | Patient sex                                                                                       |
| `race_ethnicity`    | Race / ethnicity                                                                                  |
| `insurance`         | Insurance type                                                                                    |
| `num_encounters`    | Number of encounters for this patient                                                             |
| `primary_diagnoses` | JSON array of the patient's primary diagnoses                                                     |
| `comorbidities`     | JSON array of comorbid conditions                                                                 |
| `generation_seed`   | Seed used for reproducible generation                                                             |


> **Note:** a `pcp_name` field exists in the internal database but is **intentionally
> excluded** from this release — it was a non-informative placeholder (`Dr. Smith-<id>`).
> The meaningful provider field is `encounters.attending_name`.



### `encounters` (5,602 rows — one per clinical visit, ~4.4 per patient)


| Column              | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `encounter_id`      | Unique encounter identifier                                                       |
| `patient_id`        | Foreign key to `patients`                                                         |
| `encounter_date`    | Date of the visit                                                                 |
| `encounter_type`    | `outpatient`, `ed`, `inpatient`, `icu`, `telehealth`, `procedure`, or `follow_up` |
| `chief_complaint`   | Presenting complaint                                                              |
| `attending_name`    | Attending provider (synthetic)                                                    |
| `department`        | Clinical department                                                               |
| `encounter_order`   | 0-based sequence position within the patient's timeline                           |
| `note_text`         | Full clinical note for the visit                                                  |
| `generation_method` | `hybrid` (template assembly + LLM prose polish) or `template`                     |


To reconstruct a patient's full record, join on `patient_id` and order by `encounter_order`.
The JSONL already nests encounters in that order.


### `benchmark_v1.3.db`

| Table | Rows | Contents |
|---|---|---|
| `longitudinal_patients` | 1,268 | Patient profiles (same fields as `patients` above) |
| `longitudinal_encounters` | 5,602 | Encounters with `note_text`; `source_question_ids` are opaque integer keys into the annotation tables |
| `encounter_ehr_sections` | 59,964 | The note split into typed sections (`hpi`, `pmh`, `medications`, `labs`, ...) — the passage unit for retrieval |
| `benchmark_ground_truth` | 12,014 | One row per task instance: `task`, `granularity`, `patient_id` / `encounter_id`, `ground_truth` (JSON), `split` |
| `relevance_judgments` | 58,926 | Graded relevance (0–3) of each chart section for the evidence-retrieval instance it belongs to |
| `imaging_orders` | 1,865 | Underspecified imaging orders for the imaging-indication task, keyed to their ground truth |
| `diagnoses`, `clinical_findings`, `diagnosis_findings` | 9,623 / 36,620 / 52,082 | The ontology-grounded knowledge graph: diagnoses (ICD-10-CM, SNOMED CT), findings (SNOMED CT, LOINC), and typed diagnosis–finding relations |
| `question_findings`, `question_diagnoses`, `board_questions` | 138,777 / 51,075 / 7,003 | Per-source-question annotation links (finding and diagnosis ids with roles) and question metadata (subject, organ system, difficulty). No question text is included |
| `release_info` | — | Version, export date, task list, split definitions, and change notes |

**Tasks** (`benchmark_ground_truth.task`): `patient_diagnosis` (1,268 instances), `context_summarization` (7,613: 1,268 whole-patient plus 6,345 specialty-conditioned), `evidence_retrieval` (1,268), `imaging_indication` (1,865).

**Splits** (`benchmark_ground_truth.split`), assigned at the patient level:

| Split | Patients | Instances | Role |
|---|---|---|---|
| `public` | 200 | 1,859 | The reported benchmark |
| `heldout` | 268 | 2,536 | Private evaluation set; do not train on it |
| `train` | 800 | 7,619 | Training pool, including RL with the graph-derived rewards |

The held-out split was drawn from the same pool as the training split by stratified sampling over dominant ICD-10 chapter and encounter count (seed 20260922); membership is in `scripts/build_rl_split.py`. Every encounter derives from a distinct source question, so no patient shares source material with any other.

**Excluded from the benchmark database**: source text of any kind (raw source cards, board-question vignettes and answers, fact cards and their links, source EHR sections), the licensed SNOMED CT / LOINC terminology tables (load your own with `epic_sim.migrate.load_terminology`), model outputs, and LLM call logs. Relevance judgments are restricted to chart sections; fact-card judgments are not shipped.

**Changes from v1.2**: the single-encounter diagnosis-accuracy task was retired; split labels are `public` / `heldout` / `train` (formerly `val` / `test`); the chart text was cleaned of self-contradictory medication and history statements, sentence fragments, leading whitespace, and duplicate problem-list entries, with every note rebuilt from its sections.

## Provenance

Each patient is formed by deterministically clustering compatible question vignettes into
one person (Stage 8a, no LLM), writing a coherent profile from that cluster (Stage 8b, LLM),
planning a dated visit timeline (Stage 9a, LLM), and generating each visit note by template
assembly plus LLM polish (Stage 9b). Provenance pointers back to source questions
(`source_question_ids`) are removed from the profile files; in `benchmark_v1.3.db` they are kept
as opaque integer keys because the annotation tables are keyed on them, but no question text is
included.

## Copyright & licensing

The released data is fully synthetic and carries no copyright concern. The profile files contain no terminology codes. The benchmark database's knowledge-graph tables carry per-concept ICD-10-CM codes (US public domain), LOINC codes (redistributable with attribution), and SNOMED CT identifiers (concept ids only, no descriptions beyond the display names; use of SNOMED CT content requires a UMLS/affiliate license in non-member countries). The full terminology tables are not included.

## Intended use & limitations

**Known label exposure through the simulator API.** The simulator's problem-list tool and the
`active_problems` field of the chart summary (and the FHIR Condition resource) return the graph-derived
correct diagnoses with ICD-10 codes, i.e. the patient-diagnosis reference. The `/env` endpoints and the
Harbor tasks substitute the chart's documented history and hide assessment/plan sections; agent
evaluations that call the raw tool API directly are not protected.

Intended for benchmarking clinical AI agents. The data is synthetic and must **not** be used
for clinical decision-making or treated as real patient data. Clinical content originates in
medical education material and may contain simplifications or artifacts of the generation
process.