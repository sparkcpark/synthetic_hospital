"""Imaging clinical indication task: data loading, prompt formatting, output parsing."""

import json
import logging
import re
from dataclasses import dataclass, field

from eval.prompts import get_prompt
from eval.tasks.diagnosis import _strip_markdown

log = logging.getLogger(__name__)


@dataclass
class ImagingInput:
    gt_id: int
    encounter_id: int
    patient_id: int
    modality: str
    body_region: str
    clinical_indication: str
    ehr_text: str
    ground_truth: dict
    # Ontology hints for structured strategy (preloaded in main thread)
    relevant_findings_with_loinc: list[dict] = field(default_factory=list)
    differential_icd10_codes: list[dict] = field(default_factory=list)


def load_inputs(conn, split: str = "public", granularity: str | None = None,
                pilot: int | None = None) -> list[ImagingInput]:
    """Load imaging indication GT items with assembled EHR text (temporal constraint)."""
    cur = conn.cursor()

    sql = """
        SELECT bgt.gt_id, bgt.encounter_id, bgt.ground_truth,
               io.modality, io.body_region, io.clinical_indication,
               le.patient_id, le.encounter_order
        FROM benchmark_ground_truth bgt
        JOIN imaging_orders io ON bgt.gt_id = io.gt_id
        JOIN longitudinal_encounters le ON bgt.encounter_id = le.encounter_id
        WHERE bgt.task = 'imaging_indication' AND bgt.is_diagnostic AND bgt.split = %s
        ORDER BY bgt.gt_id
    """
    params = [split]
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)

    cur.execute(sql, params)
    rows = cur.fetchall()
    log.info("Loading %d imaging GT items (split=%s)", len(rows), split)

    inputs = []
    for gt_id, enc_id, gt_json, modality, body_region, indication, pid, enc_order in rows:
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)

        # Assemble EHR up to and including target encounter (temporal constraint)
        ehr_text = _assemble_temporal_ehr(cur, pid, enc_order)
        if not ehr_text:
            log.warning("Empty EHR text for gt_id=%d", gt_id)
            continue

        # Preload ontology hints for structured strategy
        relevant_findings, differential_codes = _load_imaging_hints(cur, enc_id)

        inputs.append(ImagingInput(
            gt_id=gt_id,
            encounter_id=enc_id,
            patient_id=pid,
            modality=modality,
            body_region=body_region,
            clinical_indication=indication,
            ehr_text=ehr_text,
            ground_truth=gt,
            relevant_findings_with_loinc=relevant_findings,
            differential_icd10_codes=differential_codes,
        ))

    log.info("Loaded %d imaging inputs", len(inputs))
    return inputs


def _assemble_temporal_ehr(cur, patient_id: int, max_encounter_order: int) -> str:
    """Assemble EHR text up to and including the target encounter (no future data)."""
    cur.execute("""
        SELECT le.encounter_date, le.encounter_type, le.chief_complaint,
               ees.section_type, ees.section_text
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON ees.encounter_id = le.encounter_id
        WHERE le.patient_id = %s
          AND le.encounter_order <= %s
          AND ees.section_type NOT IN ('assessment', 'plan')
        ORDER BY le.encounter_order, ees.section_order
    """, (patient_id, max_encounter_order))

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


def _load_imaging_hints(cur, encounter_id: int) -> tuple[list[dict], list[dict]]:
    """Preload findings and differentials for structured hints."""
    # Get source question_ids for this encounter
    cur.execute("""
        SELECT source_question_ids FROM longitudinal_encounters
        WHERE encounter_id = %s AND source_question_ids IS NOT NULL
    """, (encounter_id,))
    row = cur.fetchone()
    if not row:
        return [], []

    qids = json.loads(row[0])
    if not isinstance(qids, list) or not qids:
        return [], []

    src_qid = int(qids[0])

    # Findings with LOINC codes
    cur.execute("""
        SELECT cf.display_name, cf.loinc_code, cf.finding_type
        FROM question_findings qf
        JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
        WHERE qf.question_id = %s AND qf.relevance IN ('key', 'supporting')
        ORDER BY cf.finding_type
        LIMIT 10
    """, (src_qid,))
    findings = [{"name": r[0], "loinc_code": r[1] or "", "type": r[2]}
                for r in cur.fetchall()]

    # Differential diagnoses with ICD-10
    cur.execute("""
        SELECT d.display_name, d.icd10_code, qd.role
        FROM question_diagnoses qd
        JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id
        WHERE qd.question_id = %s AND qd.role IN ('correct', 'distractor')
        ORDER BY qd.role, d.display_name
        LIMIT 8
    """, (src_qid,))
    differentials = [{"name": r[0], "icd10_code": r[1] or "", "role": r[2]}
                     for r in cur.fetchall()]

    return findings, differentials


def format_prompt(inp: ImagingInput, strategy: str) -> tuple[str, str]:
    """Format the prompt for an imaging indication input."""
    template = get_prompt("imaging_indication", strategy)

    # Few-shot: load examples or fall back to zero_shot
    few_shot_block = ""
    if strategy == "few_shot":
        from eval.examples import get_few_shot_examples
        few_shot_block = get_few_shot_examples("imaging_indication")
        if not few_shot_block:
            template = get_prompt("imaging_indication", "zero_shot")

    structured_hints = ""
    if strategy == "structured":
        from eval.hints import format_imaging_hints
        structured_hints = format_imaging_hints(
            inp.relevant_findings_with_loinc, inp.differential_icd10_codes,
        )

    user = template.user.format(
        modality=inp.modality,
        body_region=inp.body_region,
        clinical_indication=inp.clinical_indication,
        ehr_text=inp.ehr_text,
        few_shot_examples=few_shot_block,
        structured_hints=structured_hints,
    )
    return template.system, user


def parse_output(raw_text: str) -> dict:
    """Parse model output into structured imaging indication prediction."""
    text = _strip_markdown(raw_text)

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                return {
                    "clinical_question": raw_text.strip()[:200],
                    "pre_read_summary": "",
                    "must_include_findings": [],
                    "differential": [],
                    "parse_error": "Invalid JSON",
                }
        else:
            return {
                "clinical_question": raw_text.strip()[:200],
                "pre_read_summary": "",
                "must_include_findings": [],
                "differential": [],
                "parse_error": "No JSON found",
            }

    return {
        "clinical_question": data.get("clinical_question", ""),
        "pre_read_summary": data.get("pre_read_summary", ""),
        "must_include_findings": data.get("must_include_findings", []),
        "differential": data.get("differential", []),
    }
