"""APKG file reader (ZIP + SQLite).

(spec §2.4): Each .apkg file is a ZIP archive containing collection.anki21
or collection.anki2 (SQLite DBs), a media JSON mapping, and numbered media files.

(spec §9): etl/parsers/apkg.py
"""

import json
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


UNIT_SEPARATOR = "\x1f"


@dataclass
class AnkiModel:
    """An Anki note type (model) definition."""

    model_id: str
    name: str
    field_names: List[str]


@dataclass
class AnkiDeck:
    """An Anki deck definition."""

    deck_id: str
    name: str


@dataclass
class AnkiNote:
    """A single note from an Anki collection."""

    note_id: int
    model_id: str
    model_name: str
    tags: str  # Original space-separated tags
    fields: Dict[str, str]  # {field_name: field_value} with original HTML
    deck_name: str  # Deck name from the first card of this note


@dataclass
class ApkgContents:
    """Parsed contents of a single .apkg file."""

    db_version: str  # 'anki2' or 'anki21'
    models: Dict[str, AnkiModel]  # model_id → AnkiModel
    decks: Dict[str, AnkiDeck]  # deck_id → AnkiDeck
    notes: List[AnkiNote]
    note_count: int
    card_count: int


def read_apkg(apkg_path: Path) -> ApkgContents:
    """Read and parse an .apkg file.

    (spec §2.4): Prefers collection.anki21 over collection.anki2.
    (spec §5.2 Stage 1):
      - Open SQLite, read col.models (JSON) and col.decks (JSON)
      - For each note: parse flds by splitting on \\x1f, map field indices
        to field names using the note's model definition
    """
    with zipfile.ZipFile(apkg_path, "r") as z:
        names = z.namelist()

        # Prefer anki21 over anki2 (spec §2.4)
        if "collection.anki21" in names:
            db_filename = "collection.anki21"
            db_version = "anki21"
        elif "collection.anki2" in names:
            db_filename = "collection.anki2"
            db_version = "anki2"
        else:
            raise ValueError(
                f"No collection.anki21 or collection.anki2 found in {apkg_path}"
            )

        # Extract to temp directory for SQLite access
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extract(db_filename, tmpdir)
            db_path = Path(tmpdir) / db_filename

            conn = sqlite3.connect(str(db_path))
            try:
                return _parse_anki_db(conn, db_version)
            finally:
                conn.close()


def _parse_anki_db(conn: sqlite3.Connection, db_version: str) -> ApkgContents:
    """Parse the SQLite database inside an .apkg."""
    cursor = conn.cursor()

    # Read collection metadata (spec §2.4: col table has models and decks as JSON)
    cursor.execute("SELECT models, decks FROM col LIMIT 1")
    row = cursor.fetchone()
    models_json = json.loads(row[0]) if row[0] else {}
    decks_json = json.loads(row[1]) if row[1] else {}

    # Parse models (spec §2.4: note types with field definitions)
    models: Dict[str, AnkiModel] = {}
    for mid, m in models_json.items():
        field_names = [f["name"] for f in m.get("flds", [])]
        models[str(mid)] = AnkiModel(
            model_id=str(mid),
            name=m.get("name", ""),
            field_names=field_names,
        )

    # Parse decks
    decks: Dict[str, AnkiDeck] = {}
    for did, d in decks_json.items():
        decks[str(did)] = AnkiDeck(
            deck_id=str(did),
            name=d.get("name", ""),
        )

    # Build note_id → deck_name mapping via cards table
    # (spec §2.4: cards.nid = note ID, cards.did = deck ID, cards.ord = card ordinal)
    # Use the first card (ord=0 or minimum ord) to determine the note's deck
    cursor.execute(
        """
        SELECT nid, did, MIN(ord) FROM cards GROUP BY nid
        """
    )
    note_deck_map: Dict[int, str] = {}
    for nid, did, _ in cursor.fetchall():
        deck = decks.get(str(did))
        note_deck_map[nid] = deck.name if deck else "Default"

    # Count cards
    cursor.execute("SELECT count(*) FROM cards")
    card_count = cursor.fetchone()[0]

    # Read notes (spec §5.2 Stage 1: "For each note in notes table")
    cursor.execute("SELECT id, mid, tags, flds FROM notes")
    notes: List[AnkiNote] = []

    for note_id, mid, tags, flds in cursor.fetchall():
        model = models.get(str(mid))
        if model is None:
            # Skip notes with unknown models (should not happen in valid .apkg)
            continue

        # Parse flds by splitting on \x1f (spec §5.2 Stage 1)
        field_values = flds.split(UNIT_SEPARATOR)

        # Map field indices to field names (spec §5.2 Stage 1)
        fields: Dict[str, str] = {}
        for i, name in enumerate(model.field_names):
            if i < len(field_values):
                fields[name] = field_values[i]
            else:
                fields[name] = ""

        # Handle extra fields beyond the model definition
        for i in range(len(model.field_names), len(field_values)):
            if field_values[i].strip():
                fields[f"_extra_field_{i}"] = field_values[i]

        deck_name = note_deck_map.get(note_id, "Default")

        notes.append(
            AnkiNote(
                note_id=note_id,
                model_id=str(mid),
                model_name=model.name,
                tags=tags.strip() if tags else "",
                fields=fields,
                deck_name=deck_name,
            )
        )

    return ApkgContents(
        db_version=db_version,
        models=models,
        decks=decks,
        notes=notes,
        note_count=len(notes),
        card_count=card_count,
    )


def get_primary_deck_name(decks: Dict[str, AnkiDeck]) -> Optional[str]:
    """Get the primary (non-Default) deck name from a collection.

    Returns the non-Default deck name with the shortest hierarchy depth
    (i.e., top-level deck). If only Default exists, returns 'Default'.
    """
    non_default = [d for d in decks.values() if d.name != "Default"]
    if not non_default:
        return "Default"

    # Sort by hierarchy depth (number of :: separators), then alphabetically
    non_default.sort(key=lambda d: (d.name.count("::"), d.name))
    return non_default[0].name
