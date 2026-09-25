"""SQLite connection, schema creation. (spec §9: etl/db.py)

The DDL below is copied verbatim from spec.md §4.1.
"""

import sqlite3
from pathlib import Path

from etl.utils.logging import get_logger

log = get_logger("etl.db")

# Full DDL from spec.md §4.1 — verbatim
SCHEMA_DDL = """
-- ============================================================
-- LAYER 0: PROVENANCE & RAW INGESTION
-- ============================================================

CREATE TABLE IF NOT EXISTS source_decks (
    deck_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL UNIQUE,
    deck_name       TEXT,
    deck_type       TEXT NOT NULL CHECK (deck_type IN ('board_exam', 'fact')),
    anki_db_version TEXT,
    note_count      INTEGER,
    card_count      INTEGER,
    model_names     TEXT,
    deck_hierarchy  TEXT,
    ingested_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    sha256_hash     TEXT
);

CREATE TABLE IF NOT EXISTS raw_cards (
    raw_card_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    deck_id         INTEGER NOT NULL REFERENCES source_decks(deck_id),
    anki_note_id    INTEGER NOT NULL,
    anki_card_id    INTEGER,
    anki_model_name TEXT,
    anki_deck_name  TEXT,
    anki_tags       TEXT,
    field_data      TEXT NOT NULL,
    field_data_text TEXT,
    media_refs      TEXT,
    card_ordinal    INTEGER DEFAULT 0,
    -- Stage 2 classification columns (spec §5.2 Stage 2)
    card_type       TEXT CHECK (card_type IN ('board_exam', 'fact')),
    card_format     TEXT,                   -- mcq_vignette, cloze_clinical, basic_qa, cloze_fact, basic_fact, image_occlusion
    classification_method TEXT CHECK (classification_method IN ('rule_based', 'llm')),
    ingested_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(deck_id, anki_note_id, card_ordinal)
);

CREATE INDEX IF NOT EXISTS idx_raw_cards_deck ON raw_cards(deck_id);
CREATE INDEX IF NOT EXISTS idx_raw_cards_anki_note ON raw_cards(anki_note_id);

-- ============================================================
-- LAYER 1: STRUCTURED MEDICAL CONTENT
-- ============================================================

CREATE TABLE IF NOT EXISTS board_questions (
    question_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_card_id         INTEGER NOT NULL REFERENCES raw_cards(raw_card_id),
    source_qid          TEXT,
    question_format     TEXT NOT NULL CHECK (question_format IN (
        'mcq_vignette',
        'cloze_objective',
        'cloze_clinical',
        'basic_qa'
    )),
    vignette_text       TEXT,
    vignette_html       TEXT,
    question_stem       TEXT,
    clinical_setting    TEXT,
    answer_choices      TEXT,
    correct_answer      TEXT,
    correct_explanation TEXT,
    distractor_explanations TEXT,
    cloze_raw           TEXT,
    cloze_answers       TEXT,
    subject             TEXT,
    organ_system        TEXT,
    topic               TEXT,
    step_level          TEXT,
    difficulty          TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    tags_normalized     TEXT,
    extraction_method   TEXT NOT NULL CHECK (extraction_method IN ('rule_based', 'llm', 'hybrid')),
    extraction_model    TEXT,
    extraction_confidence REAL,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_bq_format ON board_questions(question_format);
CREATE INDEX IF NOT EXISTS idx_bq_subject ON board_questions(subject);
CREATE INDEX IF NOT EXISTS idx_bq_system ON board_questions(organ_system);
CREATE INDEX IF NOT EXISTS idx_bq_step ON board_questions(step_level);

CREATE TABLE IF NOT EXISTS fact_cards (
    fact_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_card_id         INTEGER NOT NULL REFERENCES raw_cards(raw_card_id),
    fact_format         TEXT NOT NULL CHECK (fact_format IN (
        'cloze_fact',
        'basic_fact',
        'image_occlusion'
    )),
    fact_text           TEXT NOT NULL,
    fact_text_cloze     TEXT,
    fact_context        TEXT,
    fact_summary        TEXT,
    subject             TEXT,
    organ_system        TEXT,
    topic               TEXT,
    specialty           TEXT,
    resource_refs       TEXT,
    tags_normalized     TEXT,
    extraction_method   TEXT NOT NULL CHECK (extraction_method IN ('rule_based', 'llm', 'hybrid')),
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_fc_subject ON fact_cards(subject);
CREATE INDEX IF NOT EXISTS idx_fc_specialty ON fact_cards(specialty);

-- ============================================================
-- LAYER 2: ONTOLOGY & CLINICAL CODING
-- ============================================================

CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    icd10_code      TEXT,
    icd10_desc      TEXT,
    snomed_id       TEXT,
    snomed_desc     TEXT,
    display_name    TEXT NOT NULL,
    category        TEXT,
    acuity          TEXT CHECK (acuity IN ('acute', 'chronic', 'acute_on_chronic', 'unspecified')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(icd10_code, snomed_id)
);

CREATE INDEX IF NOT EXISTS idx_dx_icd10 ON diagnoses(icd10_code);
CREATE INDEX IF NOT EXISTS idx_dx_snomed ON diagnoses(snomed_id);
CREATE INDEX IF NOT EXISTS idx_dx_category ON diagnoses(category);

CREATE TABLE IF NOT EXISTS clinical_findings (
    finding_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    snomed_id       TEXT,
    snomed_desc     TEXT,
    display_name    TEXT NOT NULL,
    finding_type    TEXT NOT NULL CHECK (finding_type IN (
        'symptom',
        'sign',
        'lab_value',
        'vital_sign',
        'imaging_finding',
        'procedure_result',
        'history_item',
        'medication',
        'demographic'
    )),
    normal_range    TEXT,
    loinc_code      TEXT,
    loinc_desc      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_cf_snomed ON clinical_findings(snomed_id);
CREATE INDEX IF NOT EXISTS idx_cf_type ON clinical_findings(finding_type);
CREATE INDEX IF NOT EXISTS idx_cf_loinc ON clinical_findings(loinc_code);

-- ============================================================
-- LAYER 3: RELATIONSHIP MAPPINGS
-- ============================================================

CREATE TABLE IF NOT EXISTS question_diagnoses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     INTEGER NOT NULL REFERENCES board_questions(question_id),
    diagnosis_id    INTEGER NOT NULL REFERENCES diagnoses(diagnosis_id),
    role            TEXT NOT NULL CHECK (role IN (
        'correct',
        'distractor',
        'secondary',
        'predisposing'
    )),
    confidence      REAL DEFAULT 1.0,
    source          TEXT NOT NULL CHECK (source IN ('rule_based', 'llm', 'manual')),
    UNIQUE(question_id, diagnosis_id, role)
);

CREATE INDEX IF NOT EXISTS idx_qd_question ON question_diagnoses(question_id);
CREATE INDEX IF NOT EXISTS idx_qd_diagnosis ON question_diagnoses(diagnosis_id);

CREATE TABLE IF NOT EXISTS question_findings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     INTEGER NOT NULL REFERENCES board_questions(question_id),
    finding_id      INTEGER NOT NULL REFERENCES clinical_findings(finding_id),
    present         INTEGER NOT NULL DEFAULT 1,
    value_text      TEXT,
    value_numeric   REAL,
    ehr_section     TEXT,
    relevance       TEXT CHECK (relevance IN ('key', 'supporting', 'background', 'distractor')),
    source          TEXT NOT NULL CHECK (source IN ('rule_based', 'llm', 'manual')),
    UNIQUE(question_id, finding_id)
);

CREATE INDEX IF NOT EXISTS idx_qf_question ON question_findings(question_id);
CREATE INDEX IF NOT EXISTS idx_qf_finding ON question_findings(finding_id);

CREATE TABLE IF NOT EXISTS diagnosis_findings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id    INTEGER NOT NULL REFERENCES diagnoses(diagnosis_id),
    finding_id      INTEGER NOT NULL REFERENCES clinical_findings(finding_id),
    relationship    TEXT NOT NULL CHECK (relationship IN (
        'pathognomonic',
        'highly_suggestive',
        'commonly_seen',
        'risk_factor',
        'protective',
        'rules_out'
    )),
    frequency       REAL,
    evidence_source TEXT,
    UNIQUE(diagnosis_id, finding_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_df_diagnosis ON diagnosis_findings(diagnosis_id);
CREATE INDEX IF NOT EXISTS idx_df_finding ON diagnosis_findings(finding_id);

-- Diagnosis<->diagnosis relatedness graph (Phase B, typed-edge rule 2026-06-07).
-- Tier membership is keyed on relationship TYPE/class, not a scalar weight:
--   1_definitional (SNOMED Due to / combination-concept) -> relevant by construction
--   2_associative  (Associated with / After / Pathological process / shared Finding site) -> relevant
--   3_residual     (untyped shared-finding overlap)      -> relevant iff overlap >= tau_residual
CREATE TABLE IF NOT EXISTS diagnosis_relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dx_a            INTEGER NOT NULL REFERENCES diagnoses(diagnosis_id),
    dx_b            INTEGER NOT NULL REFERENCES diagnoses(diagnosis_id),
    relation_class  TEXT NOT NULL,   -- 1_definitional | 2_associative | 3_residual
    relation_type   TEXT NOT NULL,   -- due_to | associated_with | after |
                                     -- pathological_process | shared_finding_site | shared_findings
    direction       TEXT NOT NULL,   -- a_to_b | b_to_a | symmetric
    overlap_score   REAL,            -- class 3 only
    source          TEXT NOT NULL,   -- snomed | icd10cm | derived
    mediated_via    TEXT,            -- SNOMED combination concept id, if mediated
    frozen_version  TEXT,
    UNIQUE(dx_a, dx_b, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_dr_dx_a ON diagnosis_relations(dx_a);
CREATE INDEX IF NOT EXISTS idx_dr_dx_b ON diagnosis_relations(dx_b);
CREATE INDEX IF NOT EXISTS idx_dr_class ON diagnosis_relations(relation_class);

CREATE TABLE IF NOT EXISTS fact_diagnosis_links (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id         INTEGER NOT NULL REFERENCES fact_cards(fact_id),
    diagnosis_id    INTEGER NOT NULL REFERENCES diagnoses(diagnosis_id),
    relevance       TEXT CHECK (relevance IN (
        'defines',
        'differentiates',
        'treatment',
        'epidemiology',
        'mechanism'
    )),
    UNIQUE(fact_id, diagnosis_id)
);

CREATE TABLE IF NOT EXISTS fact_finding_links (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id         INTEGER NOT NULL REFERENCES fact_cards(fact_id),
    finding_id      INTEGER NOT NULL REFERENCES clinical_findings(finding_id),
    relevance       TEXT CHECK (relevance IN (
        'defines',
        'explains',
        'interpretation',
        'normal_variant'
    )),
    UNIQUE(fact_id, finding_id)
);

-- ============================================================
-- LAYER 4: EHR SECTION PARSING
-- ============================================================

CREATE TABLE IF NOT EXISTS ehr_sections (
    section_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     INTEGER NOT NULL REFERENCES board_questions(question_id),
    section_type    TEXT NOT NULL CHECK (section_type IN (
        'demographics',
        'chief_complaint',
        'hpi',
        'pmh',
        'psh',
        'medications',
        'allergies',
        'family_history',
        'social_history',
        'ros',
        'vitals',
        'physical_exam',
        'labs',
        'imaging',
        'pathology',
        'other_studies',
        'assessment',
        'plan'
    )),
    section_text    TEXT NOT NULL,
    section_order   INTEGER NOT NULL,
    extraction_method TEXT NOT NULL CHECK (extraction_method IN ('rule_based', 'llm', 'hybrid')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_ehr_question ON ehr_sections(question_id);
CREATE INDEX IF NOT EXISTS idx_ehr_type ON ehr_sections(section_type);

-- ============================================================
-- LAYER 5: LONGITUDINAL RECORD GENERATION
-- ============================================================

CREATE TABLE IF NOT EXISTS longitudinal_patients (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    profile         TEXT NOT NULL,
    age             INTEGER,
    sex             TEXT CHECK (sex IN ('M', 'F')),
    race_ethnicity  TEXT,
    insurance       TEXT,
    pcp_name        TEXT,
    num_encounters  INTEGER DEFAULT 0,
    primary_diagnoses TEXT,
    comorbidities   TEXT,
    generation_seed INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS longitudinal_encounters (
    encounter_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES longitudinal_patients(patient_id),
    encounter_date  TEXT NOT NULL,
    encounter_type  TEXT NOT NULL CHECK (encounter_type IN (
        'outpatient', 'ed', 'inpatient', 'icu', 'telehealth', 'procedure', 'follow_up'
    )),
    chief_complaint TEXT,
    attending_name  TEXT,
    department      TEXT,
    source_question_ids TEXT,
    encounter_order INTEGER NOT NULL,
    note_text       TEXT,
    generation_method TEXT CHECK (generation_method IN ('template', 'llm', 'hybrid')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_le_patient ON longitudinal_encounters(patient_id);
CREATE INDEX IF NOT EXISTS idx_le_date ON longitudinal_encounters(encounter_date);

CREATE TABLE IF NOT EXISTS encounter_ehr_sections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id    INTEGER NOT NULL REFERENCES longitudinal_encounters(encounter_id),
    section_type    TEXT NOT NULL,
    section_text    TEXT NOT NULL,
    source_section_id INTEGER REFERENCES ehr_sections(section_id),
    is_modified     INTEGER DEFAULT 0,
    section_order   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ees_encounter ON encounter_ehr_sections(encounter_id);

-- ============================================================
-- LAYER 6: BENCHMARK EVALUATION
-- ============================================================

CREATE TABLE IF NOT EXISTS benchmark_ground_truth (
    gt_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task            TEXT NOT NULL CHECK (task IN (
        'patient_diagnosis',
        'context_summarization',
        'evidence_retrieval',
        'imaging_indication'
    )),
    granularity     TEXT NOT NULL CHECK (granularity IN ('question', 'patient', 'encounter')),
    question_id     INTEGER REFERENCES board_questions(question_id),
    patient_id      INTEGER REFERENCES longitudinal_patients(patient_id),
    encounter_id    INTEGER REFERENCES longitudinal_encounters(encounter_id),
    ground_truth    TEXT NOT NULL,
    difficulty      TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    split           TEXT CHECK (split IN ('train', 'val', 'test')),
    num_diagnoses   INTEGER,
    num_evidence    INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    CHECK (
        (granularity = 'question' AND question_id IS NOT NULL) OR
        (granularity = 'patient' AND patient_id IS NOT NULL) OR
        (granularity = 'encounter' AND encounter_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_bgt_task ON benchmark_ground_truth(task);
CREATE INDEX IF NOT EXISTS idx_bgt_split ON benchmark_ground_truth(split);
CREATE INDEX IF NOT EXISTS idx_bgt_question ON benchmark_ground_truth(question_id);
CREATE INDEX IF NOT EXISTS idx_bgt_patient ON benchmark_ground_truth(patient_id);
CREATE INDEX IF NOT EXISTS idx_bgt_encounter ON benchmark_ground_truth(encounter_id);

CREATE TABLE IF NOT EXISTS relevance_judgments (
    judgment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    gt_id           INTEGER NOT NULL REFERENCES benchmark_ground_truth(gt_id),
    passage_id      TEXT NOT NULL,
    passage_source  TEXT NOT NULL CHECK (passage_source IN (
        'ehr_section', 'fact_card', 'encounter_section'
    )),
    relevance_grade INTEGER NOT NULL CHECK (relevance_grade BETWEEN 0 AND 3),
    rationale       TEXT,
    source          TEXT NOT NULL CHECK (source IN ('rule_based', 'llm', 'manual'))
);

CREATE INDEX IF NOT EXISTS idx_rj_gt ON relevance_judgments(gt_id);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name        TEXT NOT NULL,
    method_name     TEXT NOT NULL,
    method_type     TEXT NOT NULL CHECK (method_type IN (
        'llm_direct',
        'rag',
        'fine_tuned',
        'ensemble',
        'sparse_retrieval',
        'dense_retrieval',
        'hybrid_retrieval'
    )),
    task            TEXT NOT NULL CHECK (task IN (
        'patient_diagnosis',
        'context_summarization', 'evidence_retrieval',
        'imaging_indication'
    )),
    split           TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    config          TEXT,
    metrics         TEXT,
    started_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at    TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES evaluation_runs(run_id),
    gt_id           INTEGER NOT NULL REFERENCES benchmark_ground_truth(gt_id),
    prediction      TEXT NOT NULL,
    score           REAL,
    latency_ms      INTEGER,
    token_count     INTEGER,
    raw_output      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_ep_run ON evaluation_predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_ep_gt ON evaluation_predictions(gt_id);

CREATE TABLE IF NOT EXISTS imaging_orders (
    order_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id        INTEGER NOT NULL REFERENCES longitudinal_encounters(encounter_id),
    gt_id               INTEGER NOT NULL REFERENCES benchmark_ground_truth(gt_id),
    modality            TEXT NOT NULL,
    body_region         TEXT NOT NULL,
    clinical_indication TEXT NOT NULL,
    ordering_provider   TEXT,
    order_priority      TEXT CHECK (order_priority IN ('routine', 'stat', 'urgent')),
    order_datetime      TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_io_encounter ON imaging_orders(encounter_id);
CREATE INDEX IF NOT EXISTS idx_io_gt ON imaging_orders(gt_id);

-- ============================================================
-- LAYER 7: PROCESSING METADATA
-- ============================================================

CREATE TABLE IF NOT EXISTS processing_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stage           TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('started', 'completed', 'failed', 'skipped')),
    records_in      INTEGER,
    records_out     INTEGER,
    records_error   INTEGER DEFAULT 0,
    error_message   TEXT,
    duration_sec    REAL,
    started_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at    TEXT,
    config          TEXT
);

CREATE TABLE IF NOT EXISTS llm_call_log (
    call_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stage           TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_template TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,
    latency_ms      INTEGER,
    request_id      TEXT,
    input_hash      TEXT,
    output_json     TEXT,
    raw_response    TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_stage ON llm_call_log(stage);
CREATE INDEX IF NOT EXISTS idx_llm_hash ON llm_call_log(input_hash);

-- ============================================================
-- LAYER 3 VIEWS: MULTI-LAYER DIAGNOSIS ARCHITECTURE
-- ============================================================

-- L1 Benchmark: 1 primary diagnosis per question (scoring diagnosis accuracy)
CREATE VIEW IF NOT EXISTS v_benchmark_diagnoses AS
SELECT qd.question_id, qd.diagnosis_id, d.icd10_code, d.display_name,
       qd.confidence
FROM question_diagnoses qd
JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
WHERE qd.role = 'correct';

-- L2 Encounter: All diagnoses per question (clinical reasoning / fact retrieval)
CREATE VIEW IF NOT EXISTS v_encounter_diagnoses AS
SELECT qd.question_id, qd.diagnosis_id, d.icd10_code, d.display_name,
       qd.role, qd.confidence
FROM question_diagnoses qd
JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id;

-- L3 Concept: ICD-10 3-char categories per question (guideline logic)
CREATE VIEW IF NOT EXISTS v_concept_diagnoses AS
SELECT DISTINCT qd.question_id,
       SUBSTR(d.icd10_code, 1, 3) AS icd10_category,
       d.category AS diagnosis_category
FROM question_diagnoses qd
JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
WHERE d.icd10_code IS NOT NULL;
"""


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""
    if isinstance(db_path, str):
        db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables from spec §4.1 DDL."""
    log.info("Creating database schema (all layers 0-7)")
    conn.executescript(SCHEMA_DDL)
    conn.commit()
    log.info("Schema created successfully")


def migrate_add_loinc_columns(conn: sqlite3.Connection) -> None:
    """Add loinc_code/loinc_desc columns to clinical_findings if missing.

    Safe to call on existing v1.2 databases — checks PRAGMA table_info first.
    """
    cursor = conn.execute("PRAGMA table_info(clinical_findings)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "loinc_code" not in existing_cols:
        conn.execute("ALTER TABLE clinical_findings ADD COLUMN loinc_code TEXT")
        log.info("Added loinc_code column to clinical_findings")
    if "loinc_desc" not in existing_cols:
        conn.execute("ALTER TABLE clinical_findings ADD COLUMN loinc_desc TEXT")
        log.info("Added loinc_desc column to clinical_findings")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_cf_loinc ON clinical_findings(loinc_code)")
    conn.commit()
