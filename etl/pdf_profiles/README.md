# PDF Profiles

One YAML file per source PDF (or family of PDFs). Stage 1d reads these instead of
hardcoding filenames. To add your own document, copy a file below and edit it.

| Field | Meaning |
|---|---|
| `name` | Human-readable profile id |
| `layout` | Which parser to use: `toc`, `bullets`, or `font_headings` |
| `display_name` | Human-readable source name stored on the deck |
| `filename_globs` | Filenames this profile matches (supports `*`) |
| `topic_from` | `section_title` (use the section heading) or `llm` (enrich later) |
| `context_style` | `hierarchy` (parent: title) or `episode` (episode/caps header) |

Layouts:
- `toc` — a table-of-contents driven document (headings from the TOC).
- `bullets` — a bullet/point-per-line document; titles aren't reliable topics,
  so set `topic_from: llm` and `context_style: episode`.
- `font_headings` — headings detected by font size/weight.
