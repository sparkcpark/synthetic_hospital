# Bringing Your Own Sources

The ETL is source-agnostic: it contains no deck, PDF, or vendor names. Every
source-specific detail lives in a small declarative **profile**. To run the
pipeline on your own material you add a profile — no code changes.

There are two profile kinds, both loaded by `etl/deck_registry.py`:

- **Deck profiles** — `etl/deck_profiles/*.yaml` — for Anki `.apkg` decks.
- **PDF profiles** — `etl/pdf_profiles/*.yaml` — for PDF documents.

Cards/documents are resolved to a profile at runtime; the stages just ask the
registry "what is this, and how do I parse it?"

---

## 1. Deck profiles (`.apkg`)

Each profile tags one source with a role, a parser, match rules, and (optionally)
its tag conventions. Example:

```yaml
name: my_board_deck
role: board_exam            # board_exam (question vignettes) | fact (atomic facts)
parser: mcq_fielded         # which Stage 3/4 parser to use (keys below)
card_format: mcq_vignette   # Stage 2 classification hint
filename_globs: ["my_board_*.apkg"]      # match by filename (supports *)
note_model_substrings: ["my mcq model"]  # or match by Anki note-model name
dedup_priority: 19          # lower = more specific = wins deduplication
specialty_fallback: ""      # fixed specialty for single-specialty decks
tag_config: {}              # source-specific tag tokens (see §3)
```

**Resolution order:** a card is matched by `filename_globs` first, then by
`note_model_substrings`. Config (`DECK_TYPE_MAP`, `DECK_SPECIFICITY`) and Stage 1
discovery are derived from these profiles automatically.

**Available parser keys** (each is a generic format parser in the stage's `PARSERS` map):

| Role | `parser` | Handles |
|---|---|---|
| board_exam | `mcq_fielded` | MCQ with separated fields (stem / answer / explanations) |
| board_exam | `mcq_freetext` | MCQ where vignette + choices share one field |
| fact | `cloze_qid` | Cloze cards keyed by a QID-namespaced tag |
| fact | `cloze_review_hierarchy` | Cloze cards with a `::`-delimited tag hierarchy |
| fact | `cloze_review_specialty` | Cloze cards with specialty in the deck-name hierarchy |
| fact | `cloze_resource` | Cloze cards with resource-reference fields + chapter tags |
| fact | `cloze_specialty_deck` | Single-specialty cloze deck |
| fact | `cloze_podcast` | Cloze cards derived from lecture/podcast notes |

To add a genuinely new *format*, write a parser function in `s03`/`s04`, register
it in that stage's `PARSERS` dict, and reference its key from a profile.

---

## 2. PDF profiles

```yaml
name: my_doc
layout: toc                 # toc | bullets | font_headings — selects the parser
display_name: "My Notes"
filename_globs: ["my_doc_*.pdf"]
topic_from: section_title   # section_title | llm (defer topics to LLM enrichment)
context_style: hierarchy    # hierarchy (parent: title) | episode (episode/caps header)
```

| `layout` | Parser | For |
|---|---|---|
| `toc` | `parse_toc` | Table-of-contents driven documents |
| `bullets` | `parse_bullets` | Bullet/point-per-line documents (set `topic_from: llm`) |
| `font_headings` | `parse_font_headings` | Documents whose headings are font-based |

`s01d` reads `topic_from` / `context_style` from the profile, and its
`--enrich-topics` step processes any PDF whose profile sets `topic_from: llm`.

---

## 3. Tag conventions (`tag_config`)

Source-specific tag/field tokens are **configuration, not code**. A fact deck's
profile carries a `tag_config` block that the Stage 4 parsers read. Keys by parser:

```yaml
# cloze_qid
tag_config:
  qid_namespace: "MYNS"            # leading token: MYNS::pos::Subject::System::Topic

# cloze_review_hierarchy
tag_config:
  hier_root: "ROOT"                # ROOT::L2::Specialty::SubSpecialty
  hier_level2: "L2"
  subject_namespace: "SUBJ"        # SUBJ::Subject

# cloze_resource
tag_config:
  resource_fields: ["Resource A", "Resource B"]   # field names holding references
  shelf_markers: ["!Shelf", "Shelf"]              # marker -> next part is a subject code
  topic_exclude_prefixes: ["no ", "!"]            # topic-candidate prefixes to ignore

# cloze_podcast
tag_config:
  podcast_root: "PodDeck"          # leading tag token
  podcast_branch: "MAIN"           # main-content branch
  optional_branch: "Optional"      # supplementary-source branch
  fixed_source_subjects: {SourceP: "Pathology"}   # exact source token -> subject
  subcat_source: "SourceQ"         # source whose subcategory is mapped
  keyword_source_prefix: "srck"    # source prefix -> keyword match on the tag
```

If a needed key is absent, that parsing strategy is simply skipped — the parser
falls back to keyword matching or leaves the field null. The shared taxonomy maps
(`_CHAPTER_ORGAN_MAP`, `_TAG_KEYWORD_SUBJECT`, etc.) are generic medical reference
data and need no per-source configuration.

Stage 2 classification is likewise data-driven: a deck's `note_model_substrings`
and `card_format` classify its cards; only structural fallbacks (cloze detection,
age-pattern vignette detection) remain in code.

---

## Checklist: add a new source

1. Drop a YAML file in `deck_profiles/` (or `pdf_profiles/`).
2. Set `role`/`layout`, a `parser`, and `filename_globs` (and `tag_config` if the
   chosen parser needs tokens).
3. Run the stages — no code edits. If your format matches none of the existing
   parsers, add one to the stage `PARSERS` map and reference its key.
