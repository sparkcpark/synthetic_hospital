# Deck Profiles

One YAML file per source deck/format. The pipeline reads these instead of
hardcoding deck filenames. To add your own deck, copy a file below and edit it.

Fields:

| Field | Meaning |
|---|---|
| `name` | Human-readable profile id |
| `role` | `board_exam` (question vignettes) or `fact` (atomic facts) |
| `parser` | Which extractor Stage 3/4 uses (see keys in s03/s04 `PARSERS`) |
| `card_format` | Classification hint for Stage 2 (`mcq_vignette`, `cloze_fact`, ...) |
| `filename_globs` | Filenames this profile matches (supports `*` wildcards) |
| `note_model_substrings` | Anki note-model name substrings this profile matches |
| `dedup_priority` | Lower = more specific = wins deduplication |
| `specialty_fallback` | Optional fixed specialty for single-specialty decks |
| `tag_root` | Optional namespace token for the source's `::` tag hierarchy |

A card is resolved by filename first, then by note-model substring.
