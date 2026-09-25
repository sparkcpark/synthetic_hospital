"""Stage 3: Extract Board Questions — populate board_questions from board_exam raw_cards.

  Method: Hybrid (rule-based parsing + LLM enrichment). Each board-exam source
  format has its own field-mapping extractor, selected per raw card below.

  Rule-based extraction (no LLM needed):
    - MCQ format A: Direct field mapping — Question_Stem → vignette_text + question_stem,
      A–E(+) → answer_choices, Correct_Answer → correct_answer,
      per-choice explanations → distractor_explanations
    - MCQ format B: Parse Question field (strip "Question: " prefix, split vignette
      from answer choices), parse Answer field (extract letter via regex
      Correct Answer:\\s*([A-Z])), Explanation field → correct_explanation.
      Extract source_qid from tag QID:nnnnn.
      Extract subject/organ_system/topic from tag hierarchy Root::Subject::System::Topic.
    - Tag parsing: Extract subject, organ_system, step_level from hierarchical tags

  LLM enrichment (batched) — requires glm_api:
    - clinical_setting extraction
    - vignette/stem separation refinement
    - difficulty assignment
"""

import json
import re
import sqlite3
import time

from etl.deck_registry import REGISTRY
from etl.parsers.html_strip import strip_html
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s03_extract_board")


# ---------------------------------------------------------------------------
# Question-stem separation patterns (spec §5.2 Stage 3: rule-based)
# ---------------------------------------------------------------------------

_STEM_PATTERNS = [
    re.compile(r"(Which of the following[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What is the most likely[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What is the most appropriate[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What is the next[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What is the best[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What is the most common[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What is the underlying[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What would be the[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What mechanism[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What additional[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What type of[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What condition[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What diagnosis[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What treatment[^?]*\?)", re.IGNORECASE),
    re.compile(r"(What drug[^?]*\?)", re.IGNORECASE),
]


def _split_vignette_stem(text: str) -> tuple[str, str | None]:
    """Split question text into vignette narrative and question stem.

    The question stem is the final interrogative sentence.
    Returns (vignette_text, question_stem).
    If no clear stem found, returns (full_text, None).
    """
    # Try known question patterns (use the last match, closest to end)
    for pattern in _STEM_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            m = matches[-1]
            stem = m.group(1).strip()
            vignette = text[: m.start()].strip()
            if vignette:
                return vignette, stem

    # Fallback: find the last '?' and extract that sentence
    last_q = text.rfind("?")
    if last_q > 0:
        before = text[:last_q]
        # Find sentence boundary before the question mark
        last_period = max(before.rfind(". "), before.rfind(".\n"))
        if last_period > 0:
            stem = text[last_period + 1 : last_q + 1].strip()
            if 10 < len(stem) < 500:
                vignette = text[: last_period + 1].strip()
                return vignette, stem

    return text, None


# ---------------------------------------------------------------------------
# Answer choice parsing
# ---------------------------------------------------------------------------

# Matches: "A. text" or "A) text" — captures letter and text up to next choice or end
_CHOICE_RE = re.compile(
    r"([A-Z])[.)]\s*(.*?)(?=\n\s*[A-Z][.)]\s|\Z)", re.DOTALL
)


def _parse_answer_choices_text(choices_text: str) -> list[dict]:
    """Parse answer choices from text into [{"letter": "A", "text": "..."}, ...]."""
    choices = []
    for m in _CHOICE_RE.finditer(choices_text):
        letter = m.group(1)
        text = m.group(2).strip()
        if text:
            choices.append({"letter": letter, "text": text})
    return choices


# ---------------------------------------------------------------------------
# Tag parsing (spec §5.2 Stage 3)
# ---------------------------------------------------------------------------

def _parse_tags_fielded(tags_str: str) -> dict:
    """Parse fielded-MCQ hierarchical tags.

    Tag format:
        Root::Subject::embryology
        Root::System::pathology_general_principles

    Returns dict with subject, organ_system, step_level.
    """
    result: dict[str, str | None] = {
        "subject": None,
        "organ_system": None,
        "step_level": None,
    }
    if not tags_str:
        return result

    for tag in tags_str.split():
        parts = tag.split("::")

        # Detect step level from tag prefix (e.g., Root_Step_1 → Step1)
        step_match = re.search(r"Step_(\d+)", parts[0])
        if step_match:
            result["step_level"] = f"Step{step_match.group(1)}"

        if len(parts) >= 3:
            if parts[1] == "Subject":
                result["subject"] = parts[2].replace("_", " ")
            elif parts[1] == "System":
                result["organ_system"] = parts[2].replace("_", " ")

    return result


def _parse_tags_freetext(tags_str: str) -> dict:
    """Parse free-text-MCQ hierarchical tags.

    Tag format: GPT4Anki::Subject::System::SubCategory::Topic::QID:nnnnn
    Example:    GPT4Anki::Surgery::Gastrointestinal__Nutrition::Biliary_Tract_Disorders::Biliary_Cyst::QID:21113

    Returns dict with subject, organ_system, topic, source_qid.
    """
    result: dict[str, str | None] = {
        "subject": None,
        "organ_system": None,
        "topic": None,
        "source_qid": None,
    }
    if not tags_str:
        return result

    for tag in tags_str.split():
        parts = tag.split("::")
        if len(parts) < 2 or parts[0] != "GPT4Anki":
            continue

        result["subject"] = parts[1].replace("_", " ")

        if len(parts) >= 3:
            result["organ_system"] = (
                parts[2].replace("__", " & ").replace("_", " ")
            )

        # Scan remaining parts for QID and topic
        # Topic = last non-QID part after organ_system
        for part in parts[3:]:
            if part.startswith("QID:"):
                result["source_qid"] = part.split(":")[1]
            else:
                # Each successive part is more specific; keep overwriting
                result["topic"] = part.replace("_", " ")

    return result


def _normalize_tags(tags_str: str) -> list[str]:
    """Split Anki tags into a list."""
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split() if t.strip()]


# ---------------------------------------------------------------------------
# Fielded-MCQ extraction
# ---------------------------------------------------------------------------

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _extract_mcq_fielded(
    field_data: dict, field_data_text: dict, tags_str: str
) -> dict | None:
    """Extract board question from a fielded-MCQ card.

    Spec: Direct field mapping — Question_Stem → vignette_text + question_stem,
    A–E → answer_choices, Correct_Answer → correct_answer,
    per-choice explanations → distractor_explanations.

    Returns dict of board_questions column values, or None to skip.
    """
    question_stem_html = field_data.get("Question_Stem", "")
    question_stem_text = field_data_text.get("Question_Stem", "")

    # Skip placeholder card (field value equals field name)
    if question_stem_text == "Question_Stem" or len(question_stem_text.strip()) < 20:
        return None

    correct_answer = field_data_text.get("Correct_Answer", "").strip()
    if correct_answer == "Correct_Answer_Letter" or not correct_answer:
        return None  # Placeholder

    # Split vignette from question stem
    vignette_text, stem = _split_vignette_stem(question_stem_text)

    # Build answer choices JSON array (spec: A–E → answer_choices)
    answer_choices = []
    for letter in _LETTERS:
        choice_text = field_data_text.get(letter, "").strip()
        # Skip empty fields and placeholders (value == field name)
        if choice_text and choice_text != letter:
            answer_choices.append({"letter": letter, "text": choice_text})

    # Correct explanation (spec: Explanation_Correct → correct_explanation)
    correct_explanation = field_data_text.get("Explanation_Correct", "").strip()
    if correct_explanation == "Explanation_Correct" or not correct_explanation:
        correct_explanation = None

    # Distractor explanations (spec: per-choice explanations → distractor_explanations)
    distractor_expl: dict[str, str] = {}
    for letter in _LETTERS:
        key = f"Explanation_{letter}"
        expl = field_data_text.get(key, "").strip()
        # Include non-empty, non-placeholder explanations for distractor letters
        if expl and expl != key and letter != correct_answer:
            distractor_expl[letter] = expl

    tag_info = _parse_tags_fielded(tags_str)

    return {
        "question_format": "mcq_vignette",
        "source_qid": None,
        "vignette_text": vignette_text,
        "vignette_html": question_stem_html,
        "question_stem": stem,
        "clinical_setting": None,  # LLM enrichment
        "answer_choices": json.dumps(answer_choices, ensure_ascii=False)
        if answer_choices
        else None,
        "correct_answer": correct_answer,
        "correct_explanation": correct_explanation,
        "distractor_explanations": json.dumps(distractor_expl, ensure_ascii=False)
        if distractor_expl
        else None,
        "cloze_raw": None,
        "cloze_answers": None,
        "subject": tag_info.get("subject"),
        "organ_system": tag_info.get("organ_system"),
        "topic": None,
        "step_level": tag_info.get("step_level"),
        "difficulty": None,  # LLM enrichment
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
        "extraction_model": None,
        "extraction_confidence": 1.0,
    }


# ---------------------------------------------------------------------------
# UW extraction (spec §5.2 Stage 3)
# ---------------------------------------------------------------------------

_CORRECT_ANS_RE = re.compile(r"Correct Answer:\s*([A-Z])")


def _extract_mcq_freetext(
    field_data: dict, field_data_text: dict, tags_str: str
) -> dict:
    """Extract board question from a free-text-MCQ card.

    Spec: Parse Question field (strip "Question: " prefix, split vignette
    from answer choices), parse Answer field (extract letter via regex),
    Explanation field → correct_explanation.
    """
    question_html = field_data.get("Question", "")
    question_text = field_data_text.get("Question", "")
    answer_text = field_data_text.get("Answer", "")
    explanation_text = field_data_text.get("Explanation", "")

    # Strip "Question: " prefix (spec §5.2 Stage 3)
    if question_text.startswith("Question: "):
        question_text = question_text[len("Question: ") :]
    if question_text.startswith("Question:"):
        question_text = question_text[len("Question:") :].lstrip()

    # Split at "Answer Choices:" or find inline choice markers
    if "Answer Choices:" in question_text:
        idx = question_text.index("Answer Choices:")
        vignette_and_stem = question_text[:idx].strip()
        choices_text = question_text[idx + len("Answer Choices:") :].strip()
    else:
        # Find inline choices: first "\nA." or "\nA)" pattern
        m = re.search(r"\n\s*A[.)]\s", question_text)
        if m:
            vignette_and_stem = question_text[: m.start()].strip()
            choices_text = question_text[m.start() :].strip()
        else:
            vignette_and_stem = question_text
            choices_text = ""

    # Split vignette from question stem
    vignette_text, question_stem = _split_vignette_stem(vignette_and_stem)

    # Parse answer choices
    answer_choices = _parse_answer_choices_text(choices_text)

    # Extract correct answer letter (spec: regex Correct Answer:\s*([A-Z]))
    correct_answer = None
    m = _CORRECT_ANS_RE.search(answer_text)
    if m:
        correct_answer = m.group(1)

    # Correct explanation (strip "Explanation:" prefix)
    correct_explanation = explanation_text
    if correct_explanation.startswith("Explanation:"):
        correct_explanation = correct_explanation[len("Explanation:") :].strip()
    if not correct_explanation:
        correct_explanation = None

    tag_info = _parse_tags_freetext(tags_str)

    return {
        "question_format": "mcq_vignette",
        "source_qid": tag_info.get("source_qid"),
        "vignette_text": vignette_text,
        "vignette_html": question_html,
        "question_stem": question_stem,
        "clinical_setting": None,  # LLM enrichment
        "answer_choices": json.dumps(answer_choices, ensure_ascii=False)
        if answer_choices
        else None,
        "correct_answer": correct_answer,
        "correct_explanation": correct_explanation,
        "distractor_explanations": None,  # UW doesn't have per-choice explanations
        "cloze_raw": None,
        "cloze_answers": None,
        "subject": tag_info.get("subject"),
        "organ_system": tag_info.get("organ_system"),
        "topic": tag_info.get("topic"),
        "step_level": None,
        "difficulty": None,  # LLM enrichment
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
        "extraction_model": None,
        "extraction_confidence": 1.0,
    }


# ---------------------------------------------------------------------------
# Parser registry — maps a deck profile's `parser` key to an extractor.
# A new board-exam format is added by writing a parser here and a profile in
# etl/deck_profiles/ that references its key.
# ---------------------------------------------------------------------------

PARSERS = {
    "mcq_fielded": _extract_mcq_fielded,
    "mcq_freetext": _extract_mcq_freetext,
}


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

_INSERT_SQL = """
    INSERT INTO board_questions
        (raw_card_id, source_qid, question_format,
         vignette_text, vignette_html, question_stem, clinical_setting,
         answer_choices, correct_answer, correct_explanation, distractor_explanations,
         cloze_raw, cloze_answers,
         subject, organ_system, topic, step_level, difficulty,
         tags_normalized,
         extraction_method, extraction_model, extraction_confidence)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def run(conn: sqlite3.Connection) -> dict:
    """Execute Stage 3: Extract Board Questions.

    Processes all raw_cards with card_type='board_exam' and populates
    the board_questions table (Layer 1).

    Returns a summary dict with extraction counts.
    """
    start_time = time.time()

    conn.execute(
        "INSERT INTO processing_log (stage, status) VALUES ('s03_extract_board', 'started')"
    )
    conn.commit()

    # Fetch all board_exam raw_cards with source filename
    rows = conn.execute(
        """
        SELECT rc.raw_card_id, rc.field_data, rc.field_data_text, rc.anki_tags,
               sd.filename
        FROM raw_cards rc
        JOIN source_decks sd ON rc.deck_id = sd.deck_id
        WHERE rc.card_type = 'board_exam'
        """
    ).fetchall()

    total_in = len(rows)
    inserted = 0
    skipped = 0
    errors = 0
    source_counts: dict[str, int] = {}

    for raw_card_id, field_data_json, field_data_text_json, tags_str, filename in rows:
        field_data = json.loads(field_data_json)
        field_data_text = json.loads(field_data_text_json)

        try:
            profile = REGISTRY.resolve(filename=filename)
            parser_fn = PARSERS.get(profile.parser) if profile else None
            if parser_fn is None:
                log.warning(
                    "No board-exam parser for source: %s (raw_card_id=%d)",
                    filename,
                    raw_card_id,
                )
                skipped += 1
                continue
            result = parser_fn(field_data, field_data_text, tags_str)

            if result is None:
                skipped += 1
                continue

            conn.execute(
                _INSERT_SQL,
                (
                    raw_card_id,
                    result["source_qid"],
                    result["question_format"],
                    result["vignette_text"],
                    result["vignette_html"],
                    result["question_stem"],
                    result["clinical_setting"],
                    result["answer_choices"],
                    result["correct_answer"],
                    result["correct_explanation"],
                    result["distractor_explanations"],
                    result["cloze_raw"],
                    result["cloze_answers"],
                    result["subject"],
                    result["organ_system"],
                    result["topic"],
                    result["step_level"],
                    result["difficulty"],
                    result["tags_normalized"],
                    result["extraction_method"],
                    result["extraction_model"],
                    result["extraction_confidence"],
                ),
            )
            inserted += 1
            source_counts[filename] = source_counts.get(filename, 0) + 1

        except Exception as e:
            log.error(
                "Error extracting raw_card_id=%d (%s): %s", raw_card_id, filename, e
            )
            errors += 1

    conn.commit()

    duration = time.time() - start_time

    # LLM enrichment status
    log.warning(
        "LLM enrichment skipped (glm_api not available). "
        "Fields left NULL: clinical_setting, difficulty"
    )

    # Verify extraction counts
    bq_count = conn.execute("SELECT COUNT(*) FROM board_questions").fetchone()[0]
    no_stem = conn.execute(
        "SELECT COUNT(*) FROM board_questions WHERE question_stem IS NULL"
    ).fetchone()[0]
    no_choices = conn.execute(
        "SELECT COUNT(*) FROM board_questions WHERE answer_choices IS NULL"
    ).fetchone()[0]
    no_correct = conn.execute(
        "SELECT COUNT(*) FROM board_questions WHERE correct_answer IS NULL"
    ).fetchone()[0]

    log.info("Verification: %d board_questions rows", bq_count)
    if no_stem:
        log.info("  %d without question_stem (LLM refinement needed)", no_stem)
    if no_choices:
        log.warning("  %d without answer_choices", no_choices)
    if no_correct:
        log.warning("  %d without correct_answer", no_correct)

    # Update processing_log
    conn.execute(
        """
        UPDATE processing_log
        SET status = 'completed',
            records_in = ?,
            records_out = ?,
            records_error = ?,
            duration_sec = ?,
            completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
            config = ?
        WHERE stage = 's03_extract_board' AND status = 'started'
        """,
        (
            total_in,
            inserted,
            errors,
            round(duration, 2),
            json.dumps(
                {
                    "skipped": skipped,
                    "source_counts": source_counts,
                    "no_question_stem": no_stem,
                    "no_answer_choices": no_choices,
                    "no_correct_answer": no_correct,
                    "llm_enrichment": "skipped (glm_api unavailable)",
                }
            ),
        ),
    )
    conn.commit()

    summary = {
        "records_in": total_in,
        "records_out": inserted,
        "records_error": errors,
        "skipped": skipped,
        "source_counts": source_counts,
        "no_question_stem": no_stem,
        "no_answer_choices": no_choices,
        "no_correct_answer": no_correct,
        "llm_enrichment": "skipped",
        "duration_sec": round(duration, 2),
    }

    log.info(
        "Stage 3 complete: %d in, %d extracted, %d skipped, %d errors, %.1fs",
        total_in,
        inserted,
        skipped,
        errors,
        duration,
    )
    for src, cnt in sorted(source_counts.items()):
        log.info("  %s: %d", src, cnt)

    return summary
