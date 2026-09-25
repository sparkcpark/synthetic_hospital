"""Stage 2: Classify Cards — annotate raw_cards with card_type and card_format.

  Method: Rule-based with LLM fallback. Cards are classified by their Anki note
  model name and field content, not by which source deck they came from.
  Classification rules (applied in order):
    1. MCQ-vignette model (board-exam style A)   → board_exam / mcq_vignette
    2. MCQ-vignette model (board-exam style B)    → board_exam / mcq_vignette
    3. Cloze fact model (style A)                 → fact / cloze_fact
    4. Cloze fact model (style B)                 → fact / cloze_fact
    5. Cloze w/ vignette (board_exam decks only)  → board_exam / cloze_clinical
    6. Image Occlusion                            → fact / image_occlusion
    7. All other cloze/basic                      → fact / cloze_fact or basic_fact

  To support a new source, map its note-model name to a rule in _classify_card.
"""

import json
import re
import sqlite3
import time

from etl.deck_registry import REGISTRY
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s02_classify")

# Regex for clinical vignette detection (spec §5.2 Stage 2, Rule 5)
AGE_PATTERN = re.compile(r"\d{1,3}[\s-]year[\s-]old", re.IGNORECASE)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add classification columns to raw_cards if they don't exist (migration)."""
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(raw_cards)").fetchall()
    }
    migrations = [
        ("card_type", "TEXT CHECK (card_type IN ('board_exam', 'fact'))"),
        ("card_format", "TEXT"),
        ("classification_method", "TEXT CHECK (classification_method IN ('rule_based', 'llm'))"),
    ]
    for col_name, col_def in migrations:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE raw_cards ADD COLUMN {col_name} {col_def}")
            log.info("Added column raw_cards.%s", col_name)
    conn.commit()


def _classify_card(
    model_name: str,
    deck_type: str,
    field_data_text: str | None,
) -> tuple[str, str, str]:
    """Classify a single card by rules. Returns (card_type, card_format, method)."""
    model_lower = (model_name or "").lower()
    text = field_data_text or ""

    # Rule 1: profile-driven. If a deck profile matches this note model and
    # declares a card_format, use its role/format directly. Source-specific
    # note-model names live in the profiles' note_model_substrings, not here.
    profile = REGISTRY.resolve(note_model=model_name)
    if profile and profile.card_format:
        return profile.role, profile.card_format, "rule_based"

    # The remaining rules are structural (no source/vendor names) and act as a
    # fallback for cards whose note model no profile matched.

    # Rule 2: cloze-with-vignette inside a board-exam deck -> clinical cloze
    if deck_type == "board_exam" and AGE_PATTERN.search(text):
        return "board_exam", "cloze_clinical", "rule_based"

    # Rule 3: image occlusion (standard Anki note type)
    if "image occlusion" in model_lower:
        return "fact", "image_occlusion", "rule_based"

    # Rule 4: cloze vs basic, detected structurally
    if "cloze" in model_lower or "{{c" in text:
        return "fact", "cloze_fact", "rule_based"

    return "fact", "basic_fact", "rule_based"


def run(conn: sqlite3.Connection) -> dict:
    """Execute Stage 2: Classify Cards.

    Returns a summary dict with classification counts.
    """
    start_time = time.time()

    conn.execute(
        "INSERT INTO processing_log (stage, status) VALUES ('s02_classify', 'started')"
    )
    conn.commit()

    # Ensure classification columns exist (migration for existing DBs)
    _ensure_columns(conn)

    # Fetch all raw_cards with deck_type from source_decks
    rows = conn.execute(
        """
        SELECT rc.raw_card_id, rc.anki_model_name, sd.deck_type, rc.field_data_text
        FROM raw_cards rc
        JOIN source_decks sd ON rc.deck_id = sd.deck_id
        """
    ).fetchall()

    total_in = len(rows)
    counts: dict[str, int] = {}
    rule_based = 0
    llm_classified = 0

    for raw_card_id, model_name, deck_type, field_data_text in rows:
        card_type, card_format, method = _classify_card(
            model_name, deck_type, field_data_text
        )

        conn.execute(
            """
            UPDATE raw_cards
            SET card_type = ?, card_format = ?, classification_method = ?
            WHERE raw_card_id = ?
            """,
            (card_type, card_format, method, raw_card_id),
        )

        key = f"{card_type}/{card_format}"
        counts[key] = counts.get(key, 0) + 1

        if method == "rule_based":
            rule_based += 1
        else:
            llm_classified += 1

    conn.commit()

    # Verify no unclassified cards remain
    unclassified = conn.execute(
        "SELECT COUNT(*) FROM raw_cards WHERE card_type IS NULL"
    ).fetchone()[0]

    duration = time.time() - start_time

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
        WHERE stage = 's02_classify' AND status = 'started'
        """,
        (
            total_in,
            total_in - unclassified,
            unclassified,
            round(duration, 2),
            json.dumps(counts),
        ),
    )
    conn.commit()

    summary = {
        "records_in": total_in,
        "records_out": total_in - unclassified,
        "unclassified": unclassified,
        "rule_based": rule_based,
        "llm_classified": llm_classified,
        "counts": counts,
        "duration_sec": round(duration, 2),
    }

    log.info(
        "Stage 2 complete: %d cards classified (%d rule-based, %d LLM, %d unclassified), %.1fs",
        total_in,
        rule_based,
        llm_classified,
        unclassified,
        duration,
    )
    for key, cnt in sorted(counts.items()):
        log.info("  %s: %d", key, cnt)

    return summary
