"""Stage 1: Ingest APKGs → source_decks, raw_cards.

(spec §5.2 Stage 1):
  Method: Rule-based (Python)
  1. For each .apkg file in apkg/:
     - Compute SHA-256 hash
     - Extract ZIP → find collection.anki21 or collection.anki2
     - Open SQLite, read col.models (JSON) and col.decks (JSON)
     - Insert row into source_decks
  2. For each note in notes table:
     - Parse flds by splitting on \\x1f (Unit Separator)
     - Map field indices to field names using the note's model definition
     - Strip HTML tags from each field → store both raw and plain text
     - Extract media references ([sound:...], <img src="...">)
     - Insert into raw_cards with field_data as JSON
  Deduplication: Same anki_note_id across different exports → keep the one
  from the more specific deck.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Set

from etl.config import APKG_DIR, EXCLUDED_FILES
from etl.deck_registry import REGISTRY
from etl.parsers.apkg import get_primary_deck_name, read_apkg
from etl.parsers.html_strip import extract_media_refs, strip_html
from etl.utils.hashing import sha256_file
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s01_ingest")


def run(conn: sqlite3.Connection) -> dict:
    """Execute Stage 1: Ingest APKGs.

    Returns a summary dict with records_in, records_out, records_error, duplicates_skipped.
    """
    start_time = time.time()

    # Log stage start in processing_log
    conn.execute(
        "INSERT INTO processing_log (stage, status) VALUES ('s01_ingest', 'started')"
    )
    conn.commit()

    # Discover .apkg files. A deck is ingested if a deck profile matches its
    # filename (see etl/deck_profiles/). Ordering is by the profile's
    # dedup_priority so more specific decks are processed first and win dedup.
    all_apkg = [f for f in APKG_DIR.iterdir() if f.suffix == ".apkg"]
    apkg_files = sorted(
        [f for f in all_apkg
         if REGISTRY.role_for(f.name) is not None and f.name not in EXCLUDED_FILES],
        key=lambda f: REGISTRY.priority_for(f.name),
    )

    # Warn about files on disk that no profile matches
    unknown = [f.name for f in all_apkg
               if REGISTRY.role_for(f.name) is None and f.name not in EXCLUDED_FILES]
    if unknown:
        log.warning("Skipping %d .apkg files with no matching deck profile: %s",
                    len(unknown), unknown)

    log.info(
        "Found %d .apkg files to ingest (sorted by specificity)", len(apkg_files)
    )

    # Track seen anki_note_ids for deduplication (spec §5.2 Stage 1)
    seen_note_ids: Set[int] = set()
    total_notes_in = 0
    total_notes_out = 0
    total_duplicates = 0
    total_errors = 0

    for apkg_path in apkg_files:
        filename = apkg_path.name
        deck_type = REGISTRY.role_for(filename)

        # Incremental: skip already-ingested decks
        existing_deck = conn.execute(
            "SELECT deck_id FROM source_decks WHERE filename = ?", (filename,)
        ).fetchone()
        if existing_deck:
            log.info("  %s: already ingested (deck_id=%d), skipping",
                     filename, existing_deck[0])
            continue

        log.info("Ingesting %s (type=%s)", filename, deck_type)

        # Step 1: Compute SHA-256 hash (spec §5.2 Stage 1)
        file_hash = sha256_file(apkg_path)

        # Step 2: Read and parse the APKG (spec §5.2 Stage 1)
        try:
            contents = read_apkg(apkg_path)
        except Exception as e:
            log.error("Failed to read %s: %s", filename, e)
            total_errors += 1
            continue

        # Step 3: Determine primary deck name
        primary_deck_name = get_primary_deck_name(contents.decks)

        # Step 4: Build model_names JSON array and deck_hierarchy JSON
        model_names_json = json.dumps(
            [m.name for m in contents.models.values()], ensure_ascii=False
        )
        deck_hierarchy_json = json.dumps(
            {did: d.name for did, d in contents.decks.items()}, ensure_ascii=False
        )

        # Step 5: Insert into source_decks (spec §4.1 Layer 0)
        cursor = conn.execute(
            """
            INSERT INTO source_decks
                (filename, deck_name, deck_type, anki_db_version,
                 note_count, card_count, model_names, deck_hierarchy, sha256_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                primary_deck_name,
                deck_type,
                contents.db_version,
                contents.note_count,
                contents.card_count,
                model_names_json,
                deck_hierarchy_json,
                file_hash,
            ),
        )
        deck_id = cursor.lastrowid

        # Step 6: Insert notes into raw_cards (spec §5.2 Stage 1)
        deck_notes_in = len(contents.notes)
        deck_notes_out = 0
        deck_duplicates = 0

        for note in contents.notes:
            total_notes_in += 1

            # Note: dedup is disabled for incremental ingestion.
            # All notes from new decks are imported regardless of overlap.

            # field_data: JSON object {field_name: field_value} with original HTML
            # (spec §4.1: "JSON object: {field_name: field_value}")
            field_data_json = json.dumps(note.fields, ensure_ascii=False)

            # field_data_text: Plain-text version (HTML stripped)
            # (spec §4.1: "Plain-text version (HTML stripped)")
            field_data_text_dict = {
                name: strip_html(value) for name, value in note.fields.items()
            }
            field_data_text_json = json.dumps(
                field_data_text_dict, ensure_ascii=False
            )

            # media_refs: JSON array of media filenames
            # (spec §5.2 Stage 1: "Extract media references")
            all_media: list[str] = []
            for value in note.fields.values():
                all_media.extend(extract_media_refs(value))
            media_refs_json = json.dumps(all_media) if all_media else None

            # Insert into raw_cards
            # (spec §4.1: card_ordinal DEFAULT 0, anki_card_id NULL for note-level)
            conn.execute(
                """
                INSERT INTO raw_cards
                    (deck_id, anki_note_id, anki_card_id, anki_model_name,
                     anki_deck_name, anki_tags, field_data, field_data_text,
                     media_refs, card_ordinal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    deck_id,
                    note.note_id,
                    None,  # anki_card_id: NULL for note-level ingestion
                    note.model_name,
                    note.deck_name,
                    note.tags,
                    field_data_json,
                    field_data_text_json,
                    media_refs_json,
                    0,  # card_ordinal: 0 for note-level
                ),
            )
            deck_notes_out += 1

        conn.commit()
        total_notes_out += deck_notes_out

        log.info(
            "  %s: %d notes in, %d inserted, %d duplicates skipped",
            filename,
            deck_notes_in,
            deck_notes_out,
            deck_duplicates,
        )

    duration = time.time() - start_time

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
        WHERE stage = 's01_ingest' AND status = 'started'
        """,
        (
            total_notes_in,
            total_notes_out,
            total_errors,
            round(duration, 2),
            json.dumps({"duplicates_skipped": total_duplicates}),
        ),
    )
    conn.commit()

    summary = {
        "records_in": total_notes_in,
        "records_out": total_notes_out,
        "records_error": total_errors,
        "duplicates_skipped": total_duplicates,
        "duration_sec": round(duration, 2),
    }

    log.info(
        "Stage 1 complete: %d notes in, %d inserted, %d duplicates, %d errors, %.1fs",
        total_notes_in,
        total_notes_out,
        total_duplicates,
        total_errors,
        duration,
    )

    return summary
