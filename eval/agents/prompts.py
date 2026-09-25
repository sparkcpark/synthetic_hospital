"""Agent system prompts for Phase F — 5 tasks × 2 arms.

Each task has a shared preamble + task-specific goal + arm-specific interface description.
"""

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------

PREAMBLE = """You are a clinical AI agent tasked with {task_description}.

You will interact with a patient's medical record through {interface_description}.
After gathering sufficient information, submit your answer using the submission
format specified below.

IMPORTANT CONSTRAINTS:
- You have a budget of {budget} {action_type}. Each query or tool call counts as 1.
- Plan your information gathering strategy before acting.
- Start with high-level overview, then drill into specifics as needed.
- Submit your answer when you have sufficient evidence — do not exhaust your
  budget on marginal queries.

{task_specific_goal}

SUBMISSION SCHEMA:
{output_json_schema}"""


# ---------------------------------------------------------------------------
# Interface descriptions
# ---------------------------------------------------------------------------

STRUCTURED_INTERFACE = """You have access to clinical tools for reviewing \
patient charts, viewing results, searching records, and submitting your \
clinical assessment. Tools are provided via function calling — call them \
directly when you need information.

Start by opening the patient's chart for an overview, then drill into \
specific encounters and results as needed. When you have gathered enough \
evidence, use the appropriate submit tool to record your answer.

Do NOT output JSON tool calls in your text. Simply call the tools directly \
using the function calling interface."""


BASH_INTERFACE = """You have access to a PostgreSQL database via the psql command and basic Unix
utilities (grep, jq, cat, head, wc). Write SQL queries to explore the
patient's medical record.

{orientation}"""


# ---------------------------------------------------------------------------
# Bash orientation prompt (schema + examples + tips)
# ---------------------------------------------------------------------------

BASH_ORIENTATION = """DATABASE SCHEMA (9 accessible tables):

─── longitudinal_patients ───
  patient_id          INTEGER PRIMARY KEY
  profile             JSONB
      → chronic_conditions: [{{icd10, name, onset_date}}]
      → home_medications: [{{name, dose, frequency, route}}]
      → allergies: [{{allergen, reaction, severity}}]
      → family_history: [{{relation, condition}}]
      → social_history: {{smoking_status, alcohol_use, occupation, ...}}
  age                 INTEGER
  sex                 TEXT          -- 'M' | 'F'
  num_encounters      INTEGER
  created_at          TIMESTAMPTZ

─── longitudinal_encounters ───
  encounter_id        INTEGER PRIMARY KEY
  patient_id          INTEGER       → FK longitudinal_patients
  encounter_date      DATE
  encounter_type      TEXT          -- outpatient | ed | inpatient | icu | telehealth
                                    --   | procedure | follow_up
  chief_complaint     TEXT
  attending_name      TEXT
  department          TEXT
  note_text           TEXT          -- full encounter note (concatenated sections)
  source_question_ids JSONB         -- internal reference, not clinically useful
  created_at          TIMESTAMPTZ

─── encounter_ehr_sections ───
  id                  SERIAL PRIMARY KEY
  encounter_id        INTEGER       → FK longitudinal_encounters
  section_type        TEXT          -- demographics | chief_complaint | hpi | pmh | psh
                                    --   | medications | allergies | family_history
                                    --   | social_history | ros | vitals | physical_exam
                                    --   | labs | imaging | pathology | other_studies
                                    -- (assessment/plan sections are not available)
  section_text        TEXT
  section_order       INTEGER       -- display order within encounter

─── diagnoses ───
  diagnosis_id        INTEGER PRIMARY KEY
  icd10_code          TEXT          -- e.g., 'I21.01'
  icd10_desc          TEXT
  snomed_id           BIGINT
  display_name        TEXT
  category            TEXT          -- e.g., 'cardiovascular', 'pulmonary', 'infectious'
  acuity              TEXT          -- acute | chronic | acute_on_chronic | subacute | recurrent

─── clinical_findings ───
  finding_id          INTEGER PRIMARY KEY
  snomed_id           BIGINT
  display_name        TEXT
  finding_type        TEXT          -- symptom | sign | lab_result | imaging_finding | history

─── diagnosis_findings ───
  id                  SERIAL PRIMARY KEY
  diagnosis_id        INTEGER       → FK diagnoses
  finding_id          INTEGER       → FK clinical_findings
  association_type    TEXT          -- e.g., 'pathognomonic', 'supportive', 'common', 'risk_factor'
  specificity         FLOAT         -- 0.0–1.0

─── fact_cards ───
  fact_id             INTEGER PRIMARY KEY
  fact_text           TEXT
  subject             TEXT
  organ_system        TEXT
  topic               TEXT
  created_at          TIMESTAMPTZ

─── imaging_orders ───
  order_id            INTEGER PRIMARY KEY
  encounter_id        INTEGER       → FK longitudinal_encounters
  modality            TEXT          -- CT | MRI | XR | US | NM | PET | Fluoro
  body_region         TEXT
  clinical_indication TEXT
  order_date          DATE
  created_at          TIMESTAMPTZ

─── terminology_codes ───
  code_id             SERIAL PRIMARY KEY
  code_system         TEXT          -- 'ICD10' | 'SNOMED' | 'LOINC' | 'CPT'
  code_value          TEXT
  display_name        TEXT
  parent_code         TEXT
  category            TEXT


EXAMPLE QUERIES:

1. Get patient profile:
   psql -At -c "SELECT profile FROM longitudinal_patients WHERE patient_id = {{pid}}" | jq '.'

2. List encounters chronologically:
   psql -c "SELECT encounter_id, encounter_date, encounter_type, chief_complaint, department
             FROM longitudinal_encounters
             WHERE patient_id = {{pid}}
             ORDER BY encounter_date"

3. Get EHR sections for an encounter:
   psql -c "SELECT section_type, section_text
             FROM encounter_ehr_sections
             WHERE encounter_id = {{eid}}
             ORDER BY section_order"

TIPS:
- Use -At flags for clean output without headers/borders
- Use LIMIT to avoid huge result sets (output is truncated at 8,000 characters)
- Pipe to head/tail/grep to filter long output
- Use jq to navigate JSONB columns (profile, source_question_ids)
- For text search: WHERE section_text ILIKE '%term%'
  or: to_tsvector('english', section_text) @@ plainto_tsquery('english', 'term')
- fact_cards has 53,999 rows — always filter by subject, organ_system, or topic

SUBMISSION FORMAT:
When ready to submit your answer, use curl:
  curl -s -X POST {api_base}/agent/tools \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer {{token}}" \\
    -d '{{submission_json}}'"""


# ---------------------------------------------------------------------------
# Task-specific goals + output schemas
# ---------------------------------------------------------------------------

TASK_GOALS = {
    "patient_diagnosis": {
        "description": "identifying all active diagnoses and chronic conditions for a patient",
        "goal": (
            "Review this patient's complete longitudinal medical record across all encounters. "
            "Identify all active diagnoses and chronic conditions with ICD-10-CM codes and "
            "acuity classifications (acute, chronic, acute_on_chronic)."
        ),
        "schema": """{
  "active_diagnoses": [
    {"icd10": "I21.01", "name": "STEMI involving LAD", "acuity": "acute"},
    ...
  ],
  "chronic_conditions": [
    {"icd10": "I10", "name": "Essential hypertension", "acuity": "chronic"},
    ...
  ]
}""",
    },
    "context_summarization": {
        "description": "producing a comprehensive clinical summary for a patient",
        "goal": (
            "Review this patient's medical record and produce a comprehensive clinical summary "
            "(5–10 sentences) addressing the clinical question: {clinical_question}"
        ),
        "schema": """{
  "summary": "A 65-year-old male with history of ... presented to the ED with ..."
}""",
    },
    "evidence_retrieval": {
        "description": "retrieving and ranking evidence passages for given diagnoses",
        "goal": (
            "For the given diagnoses ({diagnosis_names}), review all available clinical passages "
            "(EHR sections and fact cards) and assign relevance grades (0–3) to each passage.\n"
            "Grade 0: Not relevant. Grade 1: Marginally relevant. "
            "Grade 2: Clearly relevant. Grade 3: Highly specific / defining."
        ),
        "schema": """{
  "rankings": [
    {"passage_id": "ehr_section_1234", "grade": 3},
    {"passage_id": "fact_card_5678", "grade": 2},
    ...
  ]
}""",
    },
    "imaging_indication": {
        "description": "generating a radiology pre-read assessment for an imaging order",
        "goal": (
            "Given this imaging order ({modality} of {body_region}, indication: '{clinical_indication}'), "
            "review the patient's record to infer the underlying clinical question, produce a "
            "pre-read summary, and generate a differential diagnosis. "
            "You may review encounters up to and including encounter_id = {encounter_id} "
            "(do not access future encounters)."
        ),
        "schema": """{
  "clinical_question": "Is there evidence of pulmonary embolism?",
  "pre_read_summary": "65M with acute-onset dyspnea and pleuritic chest pain...",
  "must_include_findings": ["filling defect in pulmonary artery", ...],
  "differential": [
    {"diagnosis": "Pulmonary embolism", "icd10": "I26.99"},
    ...
  ]
}""",
    },
}


# ---------------------------------------------------------------------------
# Task-specific submit tool schemas (override Epic API defaults for Phase F)
# ---------------------------------------------------------------------------

SUBMIT_TOOL_SCHEMAS = {
    "patient_diagnosis": {
        "type": "function",
        "function": {
            "name": "submit_diagnosis",
            "description": (
                "Submit your complete diagnosis list for this patient. Include ALL active "
                "diagnoses and chronic conditions you identified, each with an ICD-10-CM code, "
                "a display name, and acuity classification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "active_diagnoses": {
                        "type": "array",
                        "description": "Active diagnoses (acute or acute_on_chronic)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "icd10": {"type": "string", "description": "ICD-10-CM code (e.g., I21.01)"},
                                "name": {"type": "string", "description": "Diagnosis display name"},
                                "acuity": {"type": "string", "enum": ["acute", "acute_on_chronic"]},
                            },
                            "required": ["icd10", "name", "acuity"],
                        },
                    },
                    "chronic_conditions": {
                        "type": "array",
                        "description": "Chronic conditions",
                        "items": {
                            "type": "object",
                            "properties": {
                                "icd10": {"type": "string", "description": "ICD-10-CM code"},
                                "name": {"type": "string", "description": "Condition display name"},
                                "acuity": {"type": "string", "enum": ["chronic"]},
                            },
                            "required": ["icd10", "name", "acuity"],
                        },
                    },
                },
                "required": ["active_diagnoses", "chronic_conditions"],
            },
        },
    },
    "context_summarization": {
        "type": "function",
        "function": {
            "name": "submit_summary",
            "description": (
                "Submit your comprehensive clinical summary (5-10 sentences) "
                "addressing the clinical question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Clinical summary text (5-10 sentences)",
                    },
                },
                "required": ["summary"],
            },
        },
    },
    "evidence_retrieval": {
        "type": "function",
        "function": {
            "name": "submit_rankings",
            "description": (
                "Submit relevance grades (0-3) for clinical passages. "
                "Grade 0=not relevant, 1=marginal, 2=clearly relevant, 3=highly specific."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rankings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "passage_id": {"type": "string"},
                                "grade": {"type": "integer", "minimum": 0, "maximum": 3},
                            },
                            "required": ["passage_id", "grade"],
                        },
                    },
                },
                "required": ["rankings"],
            },
        },
    },
    "imaging_indication": {
        "type": "function",
        "function": {
            "name": "submit_pre_read",
            "description": (
                "Submit your radiology pre-read assessment including clinical question, "
                "summary, must-include findings, and differential diagnosis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clinical_question": {
                        "type": "string",
                        "description": "The inferred clinical question driving this imaging order",
                    },
                    "pre_read_summary": {
                        "type": "string",
                        "description": "Brief clinical context for the radiologist",
                    },
                    "must_include_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Findings the radiologist must comment on",
                    },
                    "differential": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "diagnosis": {"type": "string"},
                                "icd10": {"type": "string"},
                            },
                            "required": ["diagnosis", "icd10"],
                        },
                    },
                },
                "required": ["clinical_question", "pre_read_summary", "must_include_findings", "differential"],
            },
        },
    },
}


# Bash submission examples per task
BASH_SUBMISSION_EXAMPLES = {
    "patient_diagnosis": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_diagnosis", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"active_diagnoses": [{{"icd10": "...", "name": "...", "acuity": "acute"}}], "chronic_conditions": [{{"icd10": "...", "name": "...", "acuity": "chronic"}}]}}}}}}'""",

    "context_summarization": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_summary", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"summary": "..."}}}}}}'""",

    "evidence_retrieval": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_rankings", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"rankings": [{{"passage_id": "...", "grade": 3}}]}}}}}}'""",

    "imaging_indication": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_pre_read", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"clinical_question": "...", "pre_read_summary": "...", "must_include_findings": [...], "differential": [...]}}}}}}'""",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt(
    task: str,
    arm: str,
    budget: int = 40,
    api_base: str = "http://api:8000",
    token: str = "",
    session_id: str = "",
    **task_kwargs,
) -> str:
    """Build the full system prompt for an agent session.

    Args:
        task: One of the 4 agent eval tasks (no diagnosis_accuracy).
        arm: 'structured' or 'bash'.
        budget: Action budget (default 40).
        api_base: API base URL for bash submissions.
        token: JWT token for bash submissions.
        session_id: Session ID for bash submissions.
        **task_kwargs: Task-specific placeholders (encounter_id, clinical_question, etc.)
    """
    task_info = TASK_GOALS[task]

    if arm in ("structured", "context"):
        interface_desc = STRUCTURED_INTERFACE
        action_type = "tool calls"
    else:
        orientation = BASH_ORIENTATION.format(
            api_base=api_base, pid="{pid}", eid="{eid}",
        )
        interface_desc = BASH_INTERFACE.format(orientation=orientation)
        action_type = "commands"

    goal = task_info["goal"].format(**task_kwargs)
    schema = task_info["schema"]

    prompt = PREAMBLE.format(
        task_description=task_info["description"],
        interface_description=interface_desc,
        budget=budget,
        action_type=action_type,
        task_specific_goal=goal,
        output_json_schema=schema,
    )

    # For bash arm, append the submission example
    if arm == "bash" and task in BASH_SUBMISSION_EXAMPLES:
        example = BASH_SUBMISSION_EXAMPLES[task].format(
            api_base=api_base,
            token=token,
            session_id=session_id,
            gt_id=task_kwargs.get("gt_id", "{gt_id}"),
        )
        prompt += f"\n\nSUBMISSION EXAMPLE:\n{example}"

    return prompt


def build_patient_intro(
    patient_id: int,
    task: str,
    encounter_id: int | None = None,
    diagnosis_names: str | None = None,
    modality: str | None = None,
    body_region: str | None = None,
    clinical_indication: str | None = None,
    **_extra,
) -> str:
    """Build the initial user message introducing the patient assignment."""
    parts = [f"Your assigned patient is patient_id = {patient_id}."]

    if task == "evidence_retrieval" and diagnosis_names:
        parts.append(f"Target diagnoses: {diagnosis_names}")
    elif task == "imaging_indication" and encounter_id:
        parts.append(f"Imaging order encounter_id = {encounter_id}")
        if modality:
            parts.append(f"Modality: {modality}")
        if body_region:
            parts.append(f"Body region: {body_region}")
        if clinical_indication:
            parts.append(f"Clinical indication: {clinical_indication}")

    parts.append("Begin your investigation now.")
    return "\n".join(parts)
