"""RETIRED TASK (v1.3): single-encounter diagnosis accuracy is no longer a benchmark task and is not registered in
eval.tasks. This module is kept only for the shared helpers (_strip_markdown, normalize_icd10, _assemble_*_ehr) that the
remaining task modules import.

Diagnosis accuracy task: data loading, prompt formatting, output parsing."""

import json
import logging
import re
from dataclasses import dataclass, field

from eval.prompts import get_prompt

log = logging.getLogger(__name__)


@dataclass
class DiagnosisInput:
    gt_id: int
    question_id: int | None
    patient_id: int | None
    granularity: str
    ehr_text: str
    ground_truth: dict
    # Ontology hints for structured strategy (preloaded in main thread)
    organ_systems: list[str] = field(default_factory=list)
    key_findings: list[dict] = field(default_factory=list)


def load_inputs(conn, split: str = "public", granularity: str | None = None,
                pilot: int | None = None) -> list[DiagnosisInput]:
    """Load diagnosis accuracy GT items with assembled EHR text."""
    cur = conn.cursor()

    # Determine granularity filter
    gran_filter = ""
    if granularity:
        gran_filter = f"AND granularity = '{granularity}'"

    sql = f"""
        SELECT gt_id, question_id, patient_id, granularity, ground_truth
        FROM benchmark_ground_truth
        WHERE task = 'diagnosis_accuracy' AND is_diagnostic AND split = %s {gran_filter}
        ORDER BY gt_id
    """
    params = [split]
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)

    cur.execute(sql, params)
    rows = cur.fetchall()
    log.info("Loading %d diagnosis GT items (split=%s, granularity=%s)", len(rows), split, granularity)

    inputs = []
    for gt_id, qid, pid, gran, gt_json in rows:
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)

        if gran == "question" and qid:
            ehr_text = _assemble_question_ehr(cur, qid)
        elif gran == "patient" and pid:
            ehr_text = _assemble_patient_ehr(cur, pid)
        else:
            log.warning("Skipping gt_id=%d: no question_id or patient_id", gt_id)
            continue

        if not ehr_text:
            log.warning("Empty EHR text for gt_id=%d", gt_id)
            continue

        # Preload ontology hints for structured strategy
        organ_systems, key_findings = _load_ontology_hints(cur, qid, pid, gran)

        inputs.append(DiagnosisInput(
            gt_id=gt_id, question_id=qid, patient_id=pid,
            granularity=gran, ehr_text=ehr_text, ground_truth=gt,
            organ_systems=organ_systems, key_findings=key_findings,
        ))

    # Guard: verify no patient-level items leaked through
    for inp in inputs:
        gt = inp.ground_truth
        if "active_diagnoses" in gt and "primary_diagnosis" not in gt:
            raise ValueError(
                f"gt_id={inp.gt_id} appears to be patient-level. "
                "Use --task patient_diagnosis instead."
            )

    log.info("Loaded %d diagnosis inputs with EHR text", len(inputs))
    return inputs


def _load_ontology_hints(cur, qid: int | None, pid: int | None,
                         granularity: str) -> tuple[list[str], list[dict]]:
    """Preload organ systems and key findings for structured hints."""
    all_qids = _collect_question_ids(cur, qid, pid, granularity)
    if not all_qids:
        return [], []

    qid_list = list(all_qids)

    # Organ systems
    cur.execute("""
        SELECT DISTINCT organ_system FROM board_questions
        WHERE question_id = ANY(%s) AND organ_system IS NOT NULL
    """, (qid_list,))
    organ_systems = [r[0] for r in cur.fetchall()]

    # Key findings with SNOMED
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


def _collect_question_ids(cur, qid: int | None, pid: int | None,
                          granularity: str) -> set[int]:
    """Collect source question_ids for hint queries."""
    if granularity == "question" and qid:
        return {qid}

    if granularity == "patient" and pid:
        cur.execute("""
            SELECT source_question_ids FROM longitudinal_encounters
            WHERE patient_id = %s AND source_question_ids IS NOT NULL
        """, (pid,))
        all_qids: set[int] = set()
        for (sq_ids_str,) in cur.fetchall():
            qids = json.loads(sq_ids_str)
            if isinstance(qids, list):
                all_qids.update(int(q) for q in qids)
        return all_qids

    return set()


def _assemble_question_ehr(cur, question_id: int) -> str:
    """Assemble EHR text from ehr_sections for a single question."""
    cur.execute("""
        SELECT section_type, section_text FROM ehr_sections
        WHERE question_id = %s AND section_type NOT IN ('assessment', 'plan')
        ORDER BY section_order
    """, (question_id,))
    parts = []
    for sec_type, sec_text in cur.fetchall():
        header = sec_type.upper().replace("_", " ")
        parts.append(f"[{header}]\n{sec_text}")
    return "\n\n".join(parts)


def _assemble_patient_ehr(cur, patient_id: int) -> str:
    """Assemble EHR text from encounter_ehr_sections across all encounters."""
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


def format_prompt(inp: DiagnosisInput, strategy: str) -> tuple[str, str]:
    """Format the prompt for a diagnosis input."""
    template = get_prompt("diagnosis_accuracy", strategy)

    # Few-shot: load examples or fall back to zero_shot
    few_shot_block = ""
    if strategy == "few_shot":
        from eval.examples import get_few_shot_examples
        few_shot_block = get_few_shot_examples("diagnosis_accuracy")
        if not few_shot_block:
            template = get_prompt("diagnosis_accuracy", "zero_shot")

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


def parse_output(raw_text: str) -> dict:
    """Parse model output into structured diagnosis prediction."""
    text = _strip_markdown(raw_text)

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        # Try to extract JSON object from text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                return _fallback_parse(raw_text)
        else:
            return _fallback_parse(raw_text)

    diagnoses = data.get("diagnoses", [])
    normalized = []
    for i, dx in enumerate(diagnoses):
        icd10 = normalize_icd10(dx.get("icd10", ""))
        normalized.append({
            "rank": dx.get("rank", i + 1),
            "icd10": icd10,
            "name": dx.get("name", ""),
            "confidence": dx.get("confidence", 0.0),
        })

    # Validate and correct invalid ICD-10 codes
    from eval.icd10_validator import is_valid_icd10, nearest_valid_icd10
    for dx in normalized:
        if dx["icd10"] and not is_valid_icd10(dx["icd10"]):
            corrected = nearest_valid_icd10(dx["icd10"])
            if corrected:
                dx["icd10_original"] = dx["icd10"]
                dx["icd10"] = corrected
                dx["icd10_corrected"] = True

    return {"diagnoses": normalized}


def normalize_icd10(code: str) -> str:
    """Normalize ICD-10 code: strip dots, uppercase."""
    if not code:
        return ""
    return code.replace(".", "").replace(" ", "").upper()


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    return text.strip()


def _fallback_parse(raw_text: str) -> dict:
    """Fallback: try to extract ICD-10 codes from raw text."""
    # Match ICD-10 patterns like I21.01, J15.9, etc.
    codes = re.findall(r'\b([A-Z]\d{2}\.?\d{0,2})\b', raw_text)
    diagnoses = []
    for i, code in enumerate(codes[:5]):
        diagnoses.append({
            "rank": i + 1,
            "icd10": normalize_icd10(code),
            "name": "",
            "confidence": 0.0,
        })
    if not diagnoses:
        return {"diagnoses": [], "parse_error": "No valid JSON or ICD-10 codes found"}
    return {"diagnoses": diagnoses}
