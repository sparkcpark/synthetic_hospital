"""Context summarization task: data loading, prompt formatting, output parsing."""

import json
import logging
import re
from dataclasses import dataclass, field

from eval.prompts import get_prompt
from eval.tasks.diagnosis import _assemble_patient_ehr, _strip_markdown

log = logging.getLogger(__name__)


@dataclass
class SummarizationInput:
    gt_id: int
    patient_id: int
    clinical_question: str
    must_include_findings: list[dict]
    ehr_text: str
    ground_truth: dict
    # "unconditioned" (whole-patient) | "current_visit" (point-in-time) |
    # "specialty_conditioned" (one subspecialty's view).
    variant: str = "unconditioned"
    specialty: str = ""
    # Ontology hints for structured strategy (preloaded in main thread)
    key_findings_hint: list[dict] = field(default_factory=list)


def load_inputs(conn, split: str = "public", granularity: str | None = None,
                pilot: int | None = None,
                tier: str | None = None) -> list[SummarizationInput]:
    """Load context summarization GT items with assembled EHR text.

    granularity='encounter' selects the current-visit variant (point-in-time,
    EHR truncated to encounters <= the index encounter). Otherwise the
    unconditioned whole-patient summary is loaded (default, unchanged behavior).

    tier='small'|'medium'|'large' (specialty variant only) selects rows by the
    frozen patient-aligned cohorts in specialty_dataset_manifest.json instead of
    the DB split column (tiers are nested and span val+test).
    """
    cur = conn.cursor()
    if tier is not None and granularity != "specialty":
        raise ValueError("tier= only applies to the specialty-conditioned variant (granularity='specialty')")
    if granularity == "encounter":
        return _load_current_visit_inputs(cur, split, pilot)
    if granularity == "specialty":
        return _load_specialty_inputs(cur, split, pilot, tier)

    # Unconditioned (whole-patient). Exclude the specialty-conditioned rows, which
    # also live at granularity='patient' but carry a 'variant' marker.
    sql = """
        SELECT gt_id, patient_id, ground_truth
        FROM benchmark_ground_truth
        WHERE task = 'context_summarization' AND is_diagnostic AND granularity = 'patient'
          AND COALESCE(ground_truth->>'variant', 'unconditioned') = 'unconditioned'
          AND split = %s
        ORDER BY gt_id
    """
    params = [split]
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)

    cur.execute(sql, params)
    rows = cur.fetchall()
    log.info("Loading %d summarization GT items (split=%s)", len(rows), split)

    inputs = []
    for gt_id, pid, gt_json in rows:
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
        ehr_text = _assemble_patient_ehr(cur, pid)
        if not ehr_text:
            log.warning("Empty EHR text for gt_id=%d, patient_id=%d", gt_id, pid)
            continue

        # Preload key findings for structured strategy
        key_findings_hint = _load_key_findings(cur, pid)

        inputs.append(SummarizationInput(
            gt_id=gt_id,
            patient_id=pid,
            clinical_question=gt.get("clinical_question", "What is the current active problem list and clinical trajectory?"),
            must_include_findings=gt.get("must_include_findings", []),
            ehr_text=ehr_text,
            ground_truth=gt,
            variant="unconditioned",
            key_findings_hint=key_findings_hint,
        ))

    log.info("Loaded %d summarization inputs", len(inputs))
    return inputs


def _load_current_visit_inputs(cur, split: str,
                               pilot: int | None) -> list[SummarizationInput]:
    """Load current-visit GT (granularity='encounter'); truncate EHR <= index."""
    sql = """
        SELECT gt_id, patient_id, encounter_id, ground_truth
        FROM benchmark_ground_truth
        WHERE task = 'context_summarization' AND is_diagnostic AND granularity = 'encounter'
          AND split = %s
        ORDER BY gt_id
    """
    params = [split]
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)
    cur.execute(sql, params)
    rows = cur.fetchall()
    log.info("Loading %d current-visit summarization GT items (split=%s)", len(rows), split)

    inputs = []
    for gt_id, pid, enc_id, gt_json in rows:
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
        max_order = gt.get("index_encounter_order")
        ehr_text = _assemble_patient_ehr_upto(cur, pid, max_order)
        if not ehr_text:
            log.warning("Empty current-visit EHR for gt_id=%d, patient_id=%d", gt_id, pid)
            continue
        inputs.append(SummarizationInput(
            gt_id=gt_id,
            patient_id=pid,
            clinical_question=gt.get("clinical_question", "Summarize the current encounter."),
            must_include_findings=gt.get("must_include_findings", []),
            ehr_text=ehr_text,
            ground_truth=gt,
            variant="current_visit",
        ))

    log.info("Loaded %d current-visit summarization inputs", len(inputs))
    return inputs


def _tier_patient_ids(tier: str) -> list[int] | None:
    """Frozen patient cohort for a dataset tier (None = no filter, i.e. large/all).

    Read from specialty_dataset_manifest.json at the repo root — the
    reproducibility manifest written by scripts/build_dataset_tiers.py.
    """
    from pathlib import Path
    if tier not in ("small", "medium", "large"):
        raise ValueError(f"Unknown dataset tier: {tier!r} (expected small|medium|large)")
    if tier == "large":
        return None  # all patients — no filter needed
    manifest_path = Path(__file__).resolve().parents[2] / "specialty_dataset_manifest.json"
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    ids = manifest["tiers"][tier]["patient_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Manifest tier {tier!r} has no patient_ids list")
    return ids


def _load_specialty_inputs(cur, split: str, pilot: int | None,
                           tier: str | None = None) -> list[SummarizationInput]:
    """Load specialty-conditioned GT (variant='specialty_conditioned'). Full
    patient EHR + a specialty-framed clinical question. Tiers/involvement for
    scoring travel in `ground_truth`.

    With tier set, membership comes from the frozen manifest patient list
    (spanning val+test) and the split argument is ignored.
    """
    sql = """
        SELECT gt_id, patient_id, ground_truth
        FROM benchmark_ground_truth
        WHERE task = 'context_summarization' AND is_diagnostic AND granularity = 'patient'
          AND ground_truth->>'variant' = 'specialty_conditioned'
    """
    params: list = []
    if tier is not None:
        pids = _tier_patient_ids(tier)
        if pids is not None:
            sql += " AND patient_id = ANY(%s)"
            params.append(pids)
    else:
        sql += " AND split = %s"
        params.append(split)
    sql += " ORDER BY gt_id"
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)
    cur.execute(sql, params)
    rows = cur.fetchall()
    log.info("Loading %d specialty-conditioned GT items (%s)", len(rows),
             f"tier={tier}" if tier else f"split={split}")

    inputs = []
    for gt_id, pid, gt_json in rows:
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)
        ehr_text = _assemble_patient_ehr(cur, pid)
        if not ehr_text:
            log.warning("Empty EHR for specialty gt_id=%d, patient_id=%d", gt_id, pid)
            continue
        inputs.append(SummarizationInput(
            gt_id=gt_id,
            patient_id=pid,
            clinical_question=gt.get("clinical_question",
                                     "Summarize this patient's chart for the requested specialty."),
            must_include_findings=[],
            ehr_text=ehr_text,
            ground_truth=gt,
            variant="specialty_conditioned",
            specialty=gt.get("specialty", ""),
        ))

    log.info("Loaded %d specialty-conditioned summarization inputs", len(inputs))
    return inputs


def _assemble_patient_ehr_upto(cur, patient_id: int, max_order: int | None) -> str:
    """Assemble EHR across encounters with encounter_order <= max_order.

    Enforces the no-future-leakage control: the model sees only encounters up to
    and including the index encounter. Falls back to the full record if
    max_order is None.
    """
    if max_order is None:
        return _assemble_patient_ehr(cur, patient_id)

    cur.execute("""
        SELECT le.encounter_date, le.encounter_type, le.chief_complaint,
               ees.section_type, ees.section_text
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON ees.encounter_id = le.encounter_id
        WHERE le.patient_id = %s
          AND le.encounter_order <= %s
          AND ees.section_type NOT IN ('assessment', 'plan')
        ORDER BY le.encounter_order, ees.section_order
    """, (patient_id, max_order))

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


def _load_key_findings(cur, patient_id: int) -> list[dict]:
    """Preload key findings for structured hints."""
    # Get source question_ids for this patient
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
        return []

    cur.execute("""
        SELECT DISTINCT cf.display_name, cf.finding_type
        FROM question_findings qf
        JOIN clinical_findings cf ON qf.finding_id = cf.finding_id
        WHERE qf.question_id = ANY(%s) AND qf.relevance = 'key'
        LIMIT 10
    """, (list(all_qids),))
    return [{"name": r[0], "type": r[1]} for r in cur.fetchall()]


def format_prompt(inp: SummarizationInput, strategy: str) -> tuple[str, str]:
    """Format the prompt for a summarization input (variant-aware)."""
    if inp.variant == "current_visit":
        task_key = "context_summarization_current_visit"
    elif inp.variant == "specialty_conditioned":
        task_key = "context_summarization_specialty"
    else:
        task_key = "context_summarization"
    template = get_prompt(task_key, strategy)

    # Few-shot: load examples or fall back to zero_shot
    few_shot_block = ""
    if strategy == "few_shot":
        try:
            from eval.examples import get_few_shot_examples
            few_shot_block = get_few_shot_examples(task_key)
        except Exception:
            few_shot_block = ""
        if not few_shot_block:
            template = get_prompt(task_key, "zero_shot")

    structured_hints = ""
    if strategy == "structured":
        from eval.hints import format_summarization_hints
        structured_hints = format_summarization_hints(inp.key_findings_hint)

    user = template.user.format(
        clinical_question=inp.clinical_question,
        ehr_text=inp.ehr_text,
        few_shot_examples=few_shot_block,
        structured_hints=structured_hints,
    )
    return template.system, user


def parse_output(raw_text: str) -> dict:
    """Parse model output into structured summarization prediction."""
    text = _strip_markdown(raw_text)

    try:
        data = json.loads(text, strict=False)
        summary = data.get("summary", "")
        if summary:
            return {"summary": summary}
    except json.JSONDecodeError:
        # Try to extract JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
                summary = data.get("summary", "")
                if summary:
                    return {"summary": summary}
            except json.JSONDecodeError:
                pass

    # Fallback: treat entire text as the summary (skip any reasoning prefix)
    # Look for "summary" keyword and take text after it
    cleaned = raw_text.strip()
    if cleaned:
        return {"summary": cleaned}

    return {"summary": "", "parse_error": "Empty output"}
