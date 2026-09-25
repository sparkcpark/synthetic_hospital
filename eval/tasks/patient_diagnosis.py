"""Patient-level diagnosis task: data loading, prompt formatting, output parsing.

Evaluates a model's ability to extract a complete problem list (active diagnoses
+ chronic conditions) from a patient's longitudinal medical record.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from eval.config import MODEL_CONTEXT_WINDOWS, TASK_MAX_TOKENS
from eval.prompts import get_prompt
from eval.tasks.diagnosis import normalize_icd10, _strip_markdown

log = logging.getLogger(__name__)


class ParseTier(Enum):
    A = "full_schema_match"
    B = "partial_match"
    C = "unparseable"


@dataclass
class PatientDiagnosisInput:
    gt_id: int
    patient_id: int
    ehr_text: str
    ehr_token_estimate: int
    ground_truth: dict
    organ_systems: list[str] = field(default_factory=list)
    key_findings: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Acuity normalization
# ---------------------------------------------------------------------------

ACUITY_NORMALIZATION = {
    "acute": "acute",
    "chronic": "chronic",
    "acute_on_chronic": "acute_on_chronic",
    "subacute": "acute",
    "resolving": "acute",
    "stable": "chronic",
    "exacerbation": "acute_on_chronic",
    "flare": "acute_on_chronic",
    "acute on chronic": "acute_on_chronic",
    "acute-on-chronic": "acute_on_chronic",
}


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def load_inputs(conn, split: str = "public", granularity: str | None = None,
                pilot: int | None = None) -> list[PatientDiagnosisInput]:
    """Load patient-level diagnosis GT items with assembled EHR text."""
    cur = conn.cursor()

    sql = """
        SELECT gt_id, patient_id, ground_truth
        FROM benchmark_ground_truth
        WHERE task = 'patient_diagnosis' AND is_diagnostic AND split = %s
        ORDER BY gt_id
    """
    params: list = [split]
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)

    cur.execute(sql, params)
    rows = cur.fetchall()
    log.info("Loading %d patient_diagnosis GT items (split=%s)", len(rows), split)

    inputs = []
    for gt_id, pid, gt_json in rows:
        if not pid:
            log.warning("Skipping gt_id=%d: no patient_id", gt_id)
            continue

        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
        ehr_text = _assemble_patient_ehr(cur, pid)
        if not ehr_text:
            log.warning("Empty EHR text for gt_id=%d patient_id=%d", gt_id, pid)
            continue

        organ_systems, key_findings = _load_ontology_hints(cur, pid)

        inputs.append(PatientDiagnosisInput(
            gt_id=gt_id,
            patient_id=pid,
            ehr_text=ehr_text,
            ehr_token_estimate=len(ehr_text) // 4,
            ground_truth=gt,
            organ_systems=organ_systems,
            key_findings=key_findings,
        ))

    log.info("Loaded %d patient_diagnosis inputs with EHR text", len(inputs))
    return inputs


def _assemble_patient_ehr(cur, patient_id: int) -> str:
    """Assemble longitudinal EHR text from encounter_ehr_sections."""
    cur.execute("""
        SELECT le.encounter_date, le.encounter_type, le.chief_complaint,
               ees.section_type, ees.section_text
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON ees.encounter_id = le.encounter_id
        WHERE le.patient_id = %s
          AND ees.section_type NOT IN ('assessment', 'plan')
        ORDER BY le.encounter_order, ees.section_order
    """, (patient_id,))

    parts = []
    current_enc = None
    for enc_date, enc_type, cc, sec_type, sec_text in cur.fetchall():
        enc_key = f"{enc_date}|{enc_type}"
        if enc_key != current_enc:
            current_enc = enc_key
            cc_str = f" — {cc}" if cc else ""
            parts.append(f"\n{'='*60}\nENCOUNTER: {enc_date} ({enc_type}){cc_str}\n{'='*60}")
        header = sec_type.upper().replace("_", " ")
        parts.append(f"[{header}]\n{sec_text}")
    return "\n\n".join(parts)


def _load_ontology_hints(cur, patient_id: int) -> tuple[list[str], list[dict]]:
    """Preload organ systems and key findings for structured hints."""
    # Collect question_ids for this patient
    cur.execute("""
        SELECT source_question_ids FROM longitudinal_encounters
        WHERE patient_id = %s AND source_question_ids IS NOT NULL
    """, (patient_id,))
    all_qids: set[int] = set()
    for (sq_ids_str,) in cur.fetchall():
        qids = json.loads(sq_ids_str)
        if isinstance(qids, list):
            all_qids.update(int(q) for q in qids)

    if not all_qids:
        return [], []

    qid_list = list(all_qids)

    cur.execute("SELECT to_regclass('board_questions')")
    if cur.fetchone()[0] is not None:
        cur.execute("""
            SELECT DISTINCT organ_system FROM board_questions
            WHERE question_id = ANY(%s) AND organ_system IS NOT NULL
        """, (qid_list,))
        organ_systems = [r[0] for r in cur.fetchall()]
    else:
        organ_systems = []

    cur.execute("""
        SELECT DISTINCT cf.display_name, cf.snomed_id, cf.finding_type
        FROM question_findings qf
        JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
        WHERE qf.question_id = ANY(%s) AND qf.relevance = 'key'
          AND cf.snomed_id IS NOT NULL
        LIMIT 5
    """, (qid_list,))
    key_findings = [{"name": r[0], "snomed_id": r[1], "type": r[2]}
                    for r in cur.fetchall()]

    return organ_systems, key_findings


# ---------------------------------------------------------------------------
# Format prompt
# ---------------------------------------------------------------------------

STRATEGY_OVERHEAD = {
    "zero_shot": 200, "few_shot": 1200, "cot": 800, "structured": 500,
}
SYSTEM_PROMPT_BUFFER = 500


def get_input_budget(model_name: str, strategy: str) -> int:
    """Available input tokens after reserving for system, strategy overhead, and output."""
    context = MODEL_CONTEXT_WINDOWS.get(model_name, 128000)
    return (context
            - TASK_MAX_TOKENS["patient_diagnosis"]
            - STRATEGY_OVERHEAD.get(strategy, 500)
            - SYSTEM_PROMPT_BUFFER)


def format_prompt(inp: PatientDiagnosisInput, strategy: str) -> tuple[str, str]:
    """Format the prompt for a patient diagnosis input."""
    template = get_prompt("patient_diagnosis", strategy)

    # Few-shot: load examples or fall back to zero_shot
    few_shot_block = ""
    if strategy == "few_shot":
        from eval.examples import get_few_shot_examples
        few_shot_block = get_few_shot_examples("patient_diagnosis")
        if not few_shot_block:
            template = get_prompt("patient_diagnosis", "zero_shot")

    structured_hints = ""
    if strategy == "structured":
        from eval.hints import format_diagnosis_hints
        structured_hints = format_diagnosis_hints(
            inp.organ_systems, inp.key_findings,
        )

    user = template.user.format(
        ehr_text=inp.ehr_text,
        few_shot_examples=few_shot_block,
        structured_hints=structured_hints,
    )
    return template.system, user


# ---------------------------------------------------------------------------
# Parse output
# ---------------------------------------------------------------------------

def parse_output(raw_text: str) -> dict:
    """Parse model output into structured patient diagnosis prediction.

    Returns dict with active_diagnoses, chronic_conditions, and _parse_tier.
    """
    text = _strip_markdown(raw_text)

    data = None
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                pass

    if data is None:
        return {"active_diagnoses": [], "chronic_conditions": [], "_parse_tier": ParseTier.C.value}

    active = data.get("active_diagnoses", [])
    chronic = data.get("chronic_conditions", [])

    # Tier B: flat "diagnoses" list → treat as active
    if not active and not chronic and "diagnoses" in data:
        active = data["diagnoses"]
        log.warning("Tier B parse: flat 'diagnoses' list used as active_diagnoses")

    # Determine tier
    tier = ParseTier.A
    if "active_diagnoses" not in data or "chronic_conditions" not in data:
        tier = ParseTier.B

    # Normalize entries
    norm_active = _normalize_entries(active, default_acuity="acute")
    norm_chronic = _normalize_entries(chronic, default_acuity="chronic")

    # Validate all have icd10 and acuity
    for entry in norm_active + norm_chronic:
        if not entry.get("icd10"):
            tier = ParseTier.B

    # Deduplicate
    norm_active, norm_chronic = _dedup_across_lists(norm_active, norm_chronic)

    return {
        "active_diagnoses": norm_active,
        "chronic_conditions": norm_chronic,
        "_parse_tier": tier.value,
    }


def _normalize_entries(entries: list, default_acuity: str) -> list[dict]:
    """Normalize a list of diagnosis entries."""
    from eval.icd10_validator import is_valid_icd10, nearest_valid_icd10

    result = []
    for dx in entries:
        if not isinstance(dx, dict):
            continue
        icd10 = normalize_icd10(dx.get("icd10", ""))
        name = dx.get("name", dx.get("display_name", ""))
        raw_acuity = dx.get("acuity", default_acuity)
        norm_acuity = ACUITY_NORMALIZATION.get(
            str(raw_acuity).lower().strip(), None
        )
        entry = {
            "icd10": icd10,
            "name": name,
            "acuity": norm_acuity or default_acuity,
        }
        # Validate and correct invalid ICD-10 codes
        if icd10 and not is_valid_icd10(icd10):
            corrected = nearest_valid_icd10(icd10)
            if corrected:
                entry["icd10_original"] = icd10
                entry["icd10"] = corrected
                entry["icd10_corrected"] = True
        result.append(entry)
    return result


def _dedup_across_lists(
    active: list[dict], chronic: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Deduplicate within and across active/chronic lists at 3-char ICD-10 level.

    If same category appears in both, keep only the active entry.
    """
    ACUITY_RANK = {"acute_on_chronic": 3, "acute": 2, "chronic": 1}

    def _dedup_single(entries: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for e in entries:
            cat = normalize_icd10(e["icd10"])[:3]
            if not cat:
                continue
            # S/T injury: 5-char dedup
            if cat[0] in ("S", "T"):
                cat = normalize_icd10(e["icd10"])[:5]
            existing = seen.get(cat)
            if existing is None:
                seen[cat] = e
            else:
                # Keep higher acuity, more specific code
                if ACUITY_RANK.get(e["acuity"], 0) > ACUITY_RANK.get(existing["acuity"], 0):
                    seen[cat] = e
                elif (ACUITY_RANK.get(e["acuity"], 0) == ACUITY_RANK.get(existing["acuity"], 0)
                      and len(normalize_icd10(e["icd10"])) > len(normalize_icd10(existing["icd10"]))):
                    seen[cat] = e
        return list(seen.values())

    active = _dedup_single(active)
    chronic = _dedup_single(chronic)

    # Cross-list dedup: if same category in both, keep active only
    active_cats = set()
    for e in active:
        cat = normalize_icd10(e["icd10"])[:3]
        if cat and cat[0] in ("S", "T"):
            cat = normalize_icd10(e["icd10"])[:5]
        active_cats.add(cat)

    chronic = [e for e in chronic
               if (normalize_icd10(e["icd10"])[:5] if normalize_icd10(e["icd10"])[:1] in ("S", "T")
                   else normalize_icd10(e["icd10"])[:3]) not in active_cats]

    return active, chronic
