"""PDF section parser for document ingestion (Stage 1d).

Extracts structured sections from medical education PDFs:
- TOC layout: ALL-CAPS TOC entries with page numbers -> per-section
- Bullet layout: per top-level bullet point within episodes
- Font-heading layout: font-based heading detection for granular sub-sections

Each source's parser returns a list of Section objects that define
the title, parent heading, page range, and optionally pre-extracted text.
"""

import re

# Footer lines that carry a distribution URL are boilerplate, not content.
_FOOTER_URL_RE = re.compile(r"https?://\S+")
from dataclasses import dataclass, field

import pdfplumber

from etl.deck_registry import PDF_REGISTRY
from etl.utils.logging import get_logger

log = get_logger("etl.parsers.pdf_sections")

# Max pages per section before splitting
MAX_SECTION_PAGES = 20


@dataclass
class Section:
    """A document section with its metadata and page range."""
    title: str
    parent: str | None
    page_start: int      # 0-indexed inclusive
    page_end: int         # 0-indexed exclusive
    source_type: str      # 'toc' | 'bullets' | 'font_headings' (layout)
    text: str = ""        # Pre-extracted text (bullet layout carries its own text)
    episode_title: str = ""  # Full episode title (bullet layout, for subject derivation)
    caps_header: str = ""    # ALL-CAPS sub-header within episode (bullet layout)
    image_count: int = 0     # Non-trivial images in section pages (for QA logging)


# ── TOC top-level headings (organ-system groupings in multi-level TOCs) ──
# These serve as parents for sub-sections but are NOT sections themselves
# (they share a page number with their first child).
_TOC_TOP_LEVEL_PARENTS = {
    "CARDIOLOGY", "NEUROLOGY", "OPHTHALMOLOGY", "OTORHINOLARYNGOLOGY",
    "GASTROENTEROLOGY", "PULMONOLOGY", "NEPHROLOGY", "ENDOCRINOLOGY",
    "HEMATOLOGY ONCOLOGY", "INFECTIOUS DISEASE", "RHEUMATOLOGY",
    "ORTHOPEDICS", "DERMATOLOGY", "MALE REPRODUCTIVE",
    "ENVIRONMENTAL PATHOLOGY", "ANESTHESIA", "SURGERY", "NUTRITION",
}

# TOC line pattern: title (ALL-CAPS or Title-case) followed by a page number
# Matches both "CARDIOLOGY 7" and "Prevention 2"
_TOC_LINE_RE = re.compile(r"^([A-Z][A-Za-z0-9 &/\-'(),]+?)\s+(\d+)$")

# Font-heading layout: chapter names and their expected order
_FONT_HEADING_CHAPTERS = [
    "Renal", "Pulmonary", "CVS", "Endocrine",
    "Gynae & Reproductive Health", "Obs", "GIT",
    "Blood & Oncology", "Musculoskeletal", "Dermatology",
    "CNS, Eye, Ear", "Psychiatry & Substance Abuse",
    "Short Subjects", "General Paeds", "Geriatrics",
    "Public Health", "Ethics & Communication",
]


# ── TOC Parser ────────────────────────────────────────────────────

def parse_toc(pdf: pdfplumber.PDF, filename: str) -> list[Section]:
    """Parse a table-of-contents style PDF.

    Documents whose TOC contains recognized top-level headings (see
    _TOC_TOP_LEVEL_PARENTS) are treated as multi-level, with those headings used
    as parents of their sub-sections. Documents without such headings are treated
    as single-level, and a parent is derived from the filename.
    """
    toc_entries = []
    current_parent = None

    # First, find which pages contain "Table of Contents" / "TABLE OF CONTENTS"
    toc_page_indices = []
    for page_idx in range(1, min(10, len(pdf.pages))):
        text = pdf.pages[page_idx].extract_text() or ""
        if "table of contents" in text.lower():
            toc_page_indices.append(page_idx)

    if not toc_page_indices:
        # Fallback: scan pages 1-2
        toc_page_indices = [1]

    # A multi-level TOC can span multiple pages after the header page
    first_toc = toc_page_indices[0]
    # Scan TOC header page + up to 6 following pages (some TOCs span several pages)
    scan_pages = list(range(first_toc, min(first_toc + 7, len(pdf.pages))))

    first_content_page = None  # will be set to the smallest page_num in TOC

    for page_idx in scan_pages:
        text = pdf.pages[page_idx].extract_text() or ""
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Skip noise lines
            if not line or line.startswith("pg.") or line in ("Table of Contents", "TABLE OF CONTENTS"):
                continue
            if line in ("<\u00ae>", "tut"):
                continue

            m = _TOC_LINE_RE.match(line)
            if not m:
                continue

            title = m.group(1).strip()
            page_num = int(m.group(2))

            # Track first content page to know when to stop scanning
            if first_content_page is None:
                first_content_page = page_num

            # Stop scanning if we've reached a page that IS content (beyond TOC)
            if page_idx >= (first_content_page or 999):
                break

            if title.upper() in _TOC_TOP_LEVEL_PARENTS:
                # This is a parent heading, not a standalone section
                current_parent = title.upper()
                continue

            toc_entries.append({
                "title": _title_case(title),
                "parent": current_parent,
                "page_num": page_num,
            })

        # Stop scanning pages once we've reached content area
        if first_content_page is not None and page_idx >= first_content_page:
            break

    if not toc_entries:
        log.warning(f"No TOC entries found in {filename}")
        return []

    # Single-level docs (no top-level headings found): derive parent from filename
    if not any(e["parent"] for e in toc_entries):
        parent_name = _toc_parent_from_filename(filename)
        for entry in toc_entries:
            entry["parent"] = parent_name

    # Convert to Sections with computed page ranges
    sections = _entries_to_sections(toc_entries, len(pdf.pages), "toc", pdf)
    log.info(f"Parsed {len(sections)} sections from {filename}")
    return sections


def _toc_parent_from_filename(filename: str) -> str | None:
    """Derive the parent topic from the PDF filename."""
    fn = filename.lower()
    if "biostats" in fn:
        return "BIOSTATS & SOCIAL SCIENCES"
    elif "psychiatry" in fn:
        return "PSYCHIATRY"
    elif "obgyn" in fn:
        return "OBSTETRICS & GYNECOLOGY"
    elif "pediatrics" in fn:
        return "PEDIATRICS"
    elif "im" in fn:
        return None  # IM uses per-section parents
    return None


def _title_case(s: str) -> str:
    """Convert ALL-CAPS to title case, preserving common abbreviations."""
    abbrevs = {"ACLS", "ARDS", "COPD", "DPLD", "ENT", "GI", "HIV",
               "MEN", "OCD", "PTSD", "UTI", "RBCS", "WBCS"}
    words = s.split()
    result = []
    for w in words:
        if w in abbrevs:
            result.append(w)
        elif w == "&":
            result.append("&")
        else:
            result.append(w.capitalize())
    return " ".join(result)


# ── Bullet Parser ─────────────────────────────────────────────────

def parse_bullets(pdf: pdfplumber.PDF, filename: str) -> list[Section]:
    """Parse a bullet/point-per-line PDF into per-bullet sections.

    Uses a state machine to process all pages from page 5 onward,
    detecting episode boundaries, ALL-CAPS headers, and top-level
    bullet points (marked with the bullet character).

    Each top-level bullet (plus its sub-bullets and continuation lines)
    becomes one Section with pre-extracted text. Tables on each page
    are extracted separately and emitted as standalone Sections.
    """
    sections: list[Section] = []
    current_episode = ""
    current_caps = ""
    bullet_lines: list[str] = []
    bullet_page = 5
    table_count = 0

    for page_idx in range(5, len(pdf.pages)):
        page = pdf.pages[page_idx]

        # Detect tables on this page
        tables = _find_outermost_tables(page)

        if tables:
            # Extract non-table text via filter
            table_bboxes = [t.bbox for t in tables]

            def not_in_any_table(obj, bboxes=table_bboxes):
                if "x0" not in obj or "top" not in obj:
                    return True
                ox, ot = obj["x0"], obj["top"]
                for x0, top, x1, bottom in bboxes:
                    if x0 - 2 <= ox <= x1 + 2 and top - 2 <= ot <= bottom + 2:
                        return False
                return True

            filtered_page = page.filter(not_in_any_table)
            raw_text = filtered_page.extract_text() or ""

            # Extract table text separately
            page_table_texts = []
            for t in tables:
                data = t.extract()
                if data and len(data) > 1:
                    formatted = _format_table_text(data)
                    if formatted:
                        page_table_texts.append((t.bbox[1], formatted))
        else:
            raw_text = page.extract_text() or ""
            page_table_texts = []

        lines = raw_text.split("\n")

        for line in lines:
            stripped = line.strip()

            # Skip empty and noise lines
            if not stripped:
                continue
            if re.match(r"^\d{1,4}$", stripped):
                continue
            if stripped.startswith("pg."):
                continue

            # Episode separator (long dashes)
            if re.match(r"^-{15,}", stripped):
                _flush_bullet(
                    sections, bullet_lines, current_episode,
                    current_caps, bullet_page, page_idx,
                )
                bullet_lines = []
                current_caps = ""
                continue

            # Episode header
            m = re.match(
                r"(?:Episode|Ep)\s+(\d+)\s*[:\[\s]+\s*(.+)",
                stripped, re.IGNORECASE,
            )
            if m:
                _flush_bullet(
                    sections, bullet_lines, current_episode,
                    current_caps, bullet_page, page_idx,
                )
                bullet_lines = []
                current_episode = stripped.rstrip("] ").strip()
                current_caps = ""
                continue

            # ALL-CAPS header (not a bullet, at least 5 uppercase chars)
            if (re.match(r"^[A-Z][A-Z \-/&:]{4,}$", stripped)
                    and not stripped.startswith(("\u25cf", "\u25cb", "\u25a0", "\u25aa"))):
                _flush_bullet(
                    sections, bullet_lines, current_episode,
                    current_caps, bullet_page, page_idx,
                )
                bullet_lines = []
                current_caps = stripped
                continue

            # Top-level bullet (black circle)
            if stripped.startswith("\u25cf"):
                _flush_bullet(
                    sections, bullet_lines, current_episode,
                    current_caps, bullet_page, page_idx,
                )
                bullet_lines = [stripped]
                bullet_page = page_idx
                continue

            # Sub-bullet, sub-sub-bullet, or continuation line
            if bullet_lines:
                bullet_lines.append(stripped)

        # After processing non-table lines, emit tables as standalone sections
        for _, table_text in sorted(page_table_texts, key=lambda x: x[0]):
            _flush_bullet(
                sections, bullet_lines, current_episode,
                current_caps, bullet_page, page_idx + 1,
            )
            bullet_lines = []

            parent = _bullet_parent_from_title(current_episode) if current_episode else "GENERAL PRINCIPLES"
            label = current_caps or current_episode or "Comparison"
            title = f"[Table] {label}"[:100]

            sections.append(Section(
                title=title,
                parent=parent,
                page_start=page_idx,
                page_end=page_idx + 1,
                source_type="bullets",
                text=table_text,
                episode_title=current_episode,
                caps_header=current_caps,
            ))
            table_count += 1

    # Flush final bullet
    _flush_bullet(
        sections, bullet_lines, current_episode,
        current_caps, bullet_page, len(pdf.pages),
    )

    log.info(f"Parsed {len(sections)} bullet sections ({table_count} tables) from {filename}")
    return sections


def _flush_bullet(
    sections: list[Section],
    lines: list[str],
    episode: str,
    caps_header: str,
    page_start: int,
    page_end: int,
) -> None:
    """Emit accumulated bullet lines as a Section (if non-empty)."""
    if not lines:
        return

    text = "\n".join(lines)

    # Title: first line stripped of bullet char, truncated to 100 chars
    first_line = lines[0].lstrip("\u25cf ").strip()
    title = first_line[:100]

    # Parent: derive subject category from episode title
    parent = _bullet_parent_from_title(episode) if episode else "GENERAL PRINCIPLES"

    sections.append(Section(
        title=title,
        parent=parent,
        page_start=page_start,
        page_end=max(page_start + 1, page_end),
        source_type="bullets",
        text=text,
        episode_title=episode,
        caps_header=caps_header,
    ))


def _bullet_parent_from_title(title: str) -> str:
    """Infer parent subject from episode-title keywords."""
    t = title.lower()
    # Check specific keywords in priority order
    keyword_map = [
        ("neurology", "NEUROLOGY"),
        ("neuro", "NEUROLOGY"),
        ("peds", "PEDIATRICS"),
        ("pediatric", "PEDIATRICS"),
        ("ob/gyn", "OBSTETRICS & GYNECOLOGY"),
        ("obgyn", "OBSTETRICS & GYNECOLOGY"),
        ("ob,", "OBSTETRICS & GYNECOLOGY"),
        ("gyn", "OBSTETRICS & GYNECOLOGY"),
        ("surgery", "SURGERY"),
        ("surg", "SURGERY"),
        ("ophthalmology", "OPHTHALMOLOGY"),
        ("ophtho", "OPHTHALMOLOGY"),
        ("psych", "PSYCHIATRY"),
        ("cardio", "CARDIOLOGY"),
        ("cardiac", "CARDIOLOGY"),
        ("hematology", "HEMATOLOGY ONCOLOGY"),
        ("cancer", "HEMATOLOGY ONCOLOGY"),
        ("pharmacology", "PHARMACOLOGY"),
        ("pharm", "PHARMACOLOGY"),
        ("micro", "INFECTIOUS DISEASE"),
        ("antibiotic", "INFECTIOUS DISEASE"),
        ("biostats", "BIOSTATS & SOCIAL SCIENCES"),
        ("ethics", "BIOSTATS & SOCIAL SCIENCES"),
        ("social science", "BIOSTATS & SOCIAL SCIENCES"),
        ("radiology", "RADIOLOGY"),
        ("toxicology", "ENVIRONMENTAL PATHOLOGY"),
        ("electrolyte", "NEPHROLOGY"),
        ("nephr", "NEPHROLOGY"),
        ("genetic", "GENETICS"),
        ("immunodeficiency", "IMMUNE"),
        ("pulmonary", "PULMONOLOGY"),
        ("ventilator", "PULMONOLOGY"),
        ("emergency medicine", "EMERGENCY MEDICINE"),
        ("medicine", "INTERNAL MEDICINE"),
        ("shelf review", "INTERNAL MEDICINE"),
        ("rapid review", "INTERNAL MEDICINE"),
        ("breast", "SURGERY"),
        ("risk factor", "GENERAL PRINCIPLES"),
        ("acls", "CARDIOLOGY"),
        ("antibody", "IMMUNE"),
    ]
    for keyword, parent in keyword_map:
        if keyword in t:
            return parent
    return "GENERAL PRINCIPLES"


# ── Font-Based Heading Parser ───────────────────────────────

def parse_font_headings(pdf: pdfplumber.PDF, filename: str) -> list[Section]:
    """Parse a PDF using font-based heading detection.

    Strategy:
    1. Detect chapter boundaries from footer text (for parent assignment)
    2. Detect all headings by font properties (size >= 11pt, Bold/SemiBold)
    3. Build sections from consecutive headings
    4. Split any oversized sections
    """
    # Phase 1: Map each page to its chapter using footer detection
    page_chapters = _detect_fh_chapters(pdf)

    if not page_chapters:
        log.warning(f"No chapters detected in {filename}")
        return []

    # Phase 2: Detect headings by font properties
    headings = _detect_fh_headings(pdf)

    if not headings:
        log.warning(f"No font-based headings detected in {filename}, "
                     "falling back to chapter-level sections")
        return _fh_chapter_fallback(pdf, page_chapters)

    # Phase 3: Build sections from consecutive headings
    sections = []
    for i, (page_idx, y_pos, font_size, text) in enumerate(headings):
        # Determine page_end from next heading
        if i + 1 < len(headings):
            next_page = headings[i + 1][0]
            page_end = max(page_idx + 1, next_page)
        else:
            page_end = len(pdf.pages)

        # Assign parent chapter
        parent = _find_chapter_for_page(page_chapters, page_idx)

        sections.append(Section(
            title=text,
            parent=parent,
            page_start=page_idx,
            page_end=page_end,
            source_type="font_headings",
        ))

    # Phase 4: Split oversized sections
    final_sections = []
    for sec in sections:
        span = sec.page_end - sec.page_start
        if span > MAX_SECTION_PAGES:
            parts = _split_long_section(
                pdf, sec.title, sec.parent,
                sec.page_start, sec.page_end, "font_headings",
            )
            final_sections.extend(parts)
        else:
            final_sections.append(sec)

    log.info(f"Parsed {len(final_sections)} sections from {filename} "
             f"({len(headings)} headings detected)")
    return final_sections


def _detect_fh_chapters(pdf: pdfplumber.PDF) -> dict[int, str]:
    """Detect which chapter each page belongs to via footer text."""
    page_chapters = {}
    # Known footer patterns: "chapter_name <distribution URL>"
    # Also: some pages just have the chapter name at the bottom

    # Build regex from known chapter names
    chapter_patterns = {}
    for ch in _FONT_HEADING_CHAPTERS:
        # Escape for regex, match at end of page text
        safe = re.escape(ch)
        chapter_patterns[ch] = re.compile(
            rf"(?:^|\n)\s*{safe}\s*(?:https?://\S+)?\s*$",
            re.IGNORECASE | re.MULTILINE
        )

    for page_idx in range(len(pdf.pages)):
        text = pdf.pages[page_idx].extract_text() or ""
        # Check last 300 chars for footer
        footer = text[-300:] if len(text) > 300 else text

        for ch_name, pattern in chapter_patterns.items():
            if pattern.search(footer):
                page_chapters[page_idx] = ch_name
                break

    return page_chapters


def _detect_fh_headings(
    pdf: pdfplumber.PDF,
) -> list[tuple[int, float, float, str]]:
    """Detect headings by font properties (size >= 11pt, Bold/SemiBold).

    Returns list of (page_idx, y_pos, avg_font_size, text) sorted by position.
    """
    headings: list[tuple[int, float, float, str]] = []

    for page_idx in range(len(pdf.pages)):
        page = pdf.pages[page_idx]
        chars = page.chars
        if not chars:
            continue

        # Group characters by y-position (round to nearest 2pt for baseline tolerance)
        by_y: dict[int, list] = {}
        for c in chars:
            y_key = round(c["top"] / 2) * 2
            if y_key not in by_y:
                by_y[y_key] = []
            by_y[y_key].append(c)

        for y_key in sorted(by_y.keys()):
            line_chars = [c for c in by_y[y_key] if c["text"].strip()]
            if len(line_chars) < 3:
                continue

            # Compute average font size
            sizes = [c["size"] for c in line_chars]
            avg_size = sum(sizes) / len(sizes)

            # Skip too small (body text) or too large (artifacts)
            if avg_size < 11.0 or avg_size > 25.0:
                continue

            # Check bold percentage (need majority bold)
            bold_count = sum(
                1 for c in line_chars
                if "Bold" in c.get("fontname", "") or "SemiBold" in c.get("fontname", "")
            )
            if bold_count / len(line_chars) < 0.5:
                continue

            # Reconstruct text from characters
            text = _reconstruct_line_text(line_chars)

            # Filter out noise
            if len(text) < 3:
                continue
            if text in ("Version 2",):
                continue
            if re.match(r"^\d+$", text):
                continue
            if "http" in text.lower():
                continue
            if text.startswith(("\u00a9", "Step 2", "Step 3")):
                continue
            # Skip if text matches a chapter name exactly (likely TOC header)
            if text in _FONT_HEADING_CHAPTERS:
                continue

            headings.append((page_idx, float(y_key), avg_size, text))

    return headings


def _reconstruct_line_text(chars: list) -> str:
    """Reconstruct a text line from character objects, inserting spaces at word gaps."""
    if not chars:
        return ""
    sorted_chars = sorted(chars, key=lambda c: c["x0"])
    parts = [sorted_chars[0]["text"]]
    for i in range(1, len(sorted_chars)):
        prev = sorted_chars[i - 1]
        curr = sorted_chars[i]
        gap = curr["x0"] - prev.get("x1", prev["x0"] + prev["size"] * 0.6)
        if gap > 2:
            parts.append(" ")
        parts.append(curr["text"])
    return "".join(parts).strip()


def _find_chapter_for_page(page_chapters: dict[int, str], page_idx: int) -> str | None:
    """Find the chapter for a page by checking it and nearby pages (backward)."""
    if page_idx in page_chapters:
        return page_chapters[page_idx]
    for offset in range(1, 15):
        if page_idx - offset in page_chapters:
            return page_chapters[page_idx - offset]
    return None


def _fh_chapter_fallback(
    pdf: pdfplumber.PDF,
    page_chapters: dict[int, str],
) -> list[Section]:
    """Fallback: create one section per chapter when heading detection fails."""
    chapter_ranges: dict[str, tuple[int, int]] = {}
    current_chapter = None
    start_page = None

    for page_idx in range(len(pdf.pages)):
        ch = page_chapters.get(page_idx)
        if ch and ch != current_chapter:
            if current_chapter is not None:
                chapter_ranges[current_chapter] = (start_page, page_idx)
            current_chapter = ch
            start_page = page_idx
    if current_chapter is not None:
        chapter_ranges[current_chapter] = (start_page, len(pdf.pages))

    sections = []
    for chapter_name, (ch_start, ch_end) in chapter_ranges.items():
        if ch_end - ch_start > MAX_SECTION_PAGES:
            parts = _split_long_section(
                pdf, chapter_name, chapter_name,
                ch_start, ch_end, "font_headings",
            )
            sections.extend(parts)
        else:
            sections.append(Section(
                title=chapter_name,
                parent=chapter_name,
                page_start=ch_start,
                page_end=ch_end,
                source_type="font_headings",
            ))

    return sections


# ── Table Extraction Utilities ────────────────────────────────────

def _find_outermost_tables(page) -> list:
    """Find tables on a page, deduplicating nested sub-tables.

    Returns list of pdfplumber Table objects, keeping only the outermost
    (largest-area) table when tables are spatially nested.
    """
    tables = page.find_tables()
    if len(tables) <= 1:
        return tables

    # Sort by area descending
    tables_sorted = sorted(
        tables,
        key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]),
        reverse=True,
    )

    outermost = []
    for t in tables_sorted:
        x0, top, x1, bottom = t.bbox
        contained = False
        for kept in outermost:
            kx0, ktop, kx1, kbottom = kept.bbox
            if kx0 <= x0 and ktop <= top and kx1 >= x1 and kbottom >= bottom:
                contained = True
                break
        if not contained:
            outermost.append(t)

    return outermost


def _format_table_text(table_data: list[list]) -> str:
    """Format raw table extraction data into readable pipe-separated text.

    Handles: None cells, continuation rows (first cell empty),
    empty columns, and newlines within cells.
    """
    if not table_data:
        return ""

    # Step 1: Clean None -> ""
    cleaned = []
    for row in table_data:
        cleaned.append([(cell or "").strip() for cell in row])

    # Step 2: Remove entirely empty rows
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    # Step 3: Merge continuation rows (rows where first cell is empty)
    merged = []
    for row in cleaned:
        if merged and not row[0] and any(row):
            for i, cell in enumerate(row):
                if cell and i < len(merged[-1]):
                    if merged[-1][i]:
                        merged[-1][i] += " " + cell
                    else:
                        merged[-1][i] = cell
        else:
            merged.append(list(row))

    # Step 4: Remove empty columns
    if merged:
        non_empty_cols = set()
        for row in merged:
            for i, cell in enumerate(row):
                if cell:
                    non_empty_cols.add(i)
        if non_empty_cols:
            indices = sorted(non_empty_cols)
            merged = [[row[i] for i in indices if i < len(row)] for row in merged]

    # Step 5: Replace internal newlines with semicolons for flat text
    for row in merged:
        for i, cell in enumerate(row):
            row[i] = cell.replace("\n", "; ")

    # Step 6: Format as pipe-separated lines
    lines = []
    for row in merged:
        line = " | ".join(row)
        if line.strip():
            lines.append(line)

    if not lines:
        return ""
    return "[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]"


def _extract_page_text_with_tables(page, source_type: str) -> str:
    """Extract text from a page, handling tables separately.

    1. Detect tables on the page
    2. Extract non-table text (filtered)
    3. Extract and format table text
    4. Combine in vertical order (table position on page)

    Returns clean text with tables formatted as pipe-separated blocks.
    """
    tables = _find_outermost_tables(page)

    if not tables:
        return page.extract_text() or ""

    # Build table bboxes for exclusion
    table_bboxes = [t.bbox for t in tables]

    # Extract non-table text using filter
    def not_in_any_table(obj):
        if "x0" not in obj or "top" not in obj:
            return True
        ox, ot = obj["x0"], obj["top"]
        for x0, top, x1, bottom in table_bboxes:
            if x0 - 2 <= ox <= x1 + 2 and top - 2 <= ot <= bottom + 2:
                return False
        return True

    filtered_page = page.filter(not_in_any_table)
    non_table_text = filtered_page.extract_text() or ""

    # Extract and format each table, noting its vertical position
    table_blocks = []
    for t in tables:
        data = t.extract()
        if data and len(data) > 1:  # skip single-row "tables" (likely false positive)
            formatted = _format_table_text(data)
            if formatted:
                table_blocks.append((t.bbox[1], formatted))  # (y_top, text)

    if not table_blocks:
        return non_table_text

    # Append table blocks after non-table text (sorted by vertical position)
    parts = [non_table_text.rstrip()]
    for _, table_text in sorted(table_blocks, key=lambda x: x[0]):
        parts.append(table_text)

    return "\n\n".join(p for p in parts if p.strip())


def _count_page_images(page, min_size: int = 50) -> int:
    """Count non-trivial images on a page (excluding backgrounds and tiny icons)."""
    count = 0
    pw, ph = float(page.width), float(page.height)
    for img in page.images:
        w = abs(img.get("x1", 0) - img.get("x0", 0))
        h = abs(img.get("y1", 0) - img.get("y0", 0))
        # Skip full-page backgrounds
        if w >= pw * 0.9 and h >= ph * 0.9:
            continue
        # Skip tiny icons (bullet dots, etc.)
        if w < min_size and h < min_size:
            continue
        count += 1
    return count


# ── Shared Utilities ──────────────────────────────────────────────

def _entries_to_sections(
    entries: list[dict],
    total_pages: int,
    source_type: str,
    pdf: pdfplumber.PDF,
) -> list[Section]:
    """Convert TOC entries (title, parent, page_num) to Section objects.

    Computes page_end from the next entry's page_start.
    Handles long sections by splitting at page boundaries.
    """
    if not entries:
        return []

    # Sort by page number
    entries.sort(key=lambda e: e["page_num"])

    sections = []
    for i, entry in enumerate(entries):
        page_start = entry["page_num"]

        if i + 1 < len(entries):
            page_end = entries[i + 1]["page_num"]
        else:
            page_end = total_pages

        # Clamp to valid range
        page_start = max(0, min(page_start, total_pages - 1))
        page_end = max(page_start + 1, min(page_end, total_pages))

        if page_end <= page_start:
            continue

        # Split long sections
        if page_end - page_start > MAX_SECTION_PAGES:
            sub_sections = _split_long_section(
                pdf, entry["title"], entry["parent"],
                page_start, page_end, source_type,
            )
            sections.extend(sub_sections)
        else:
            sections.append(Section(
                title=entry["title"],
                parent=entry["parent"],
                page_start=page_start,
                page_end=page_end,
                source_type=source_type,
            ))

    return sections


def _split_long_section(
    pdf: pdfplumber.PDF,
    title: str,
    parent: str | None,
    page_start: int,
    page_end: int,
    source_type: str,
) -> list[Section]:
    """Split a section that exceeds MAX_SECTION_PAGES.

    Splits at MAX_SECTION_PAGES intervals, creating parts like
    "Original Title (Part 1)", "Original Title (Part 2)", etc.
    """
    sections = []
    part = 1
    current_start = page_start

    while current_start < page_end:
        current_end = min(current_start + MAX_SECTION_PAGES, page_end)
        part_title = f"{title} (Part {part})" if page_end - page_start > MAX_SECTION_PAGES else title

        sections.append(Section(
            title=part_title,
            parent=parent,
            page_start=current_start,
            page_end=current_end,
            source_type=source_type,
        ))
        current_start = current_end
        part += 1

    return sections


def extract_section_text(
    pdf: pdfplumber.PDF,
    page_start: int,
    page_end: int,
    source_type: str,
) -> str:
    """Extract and clean text for a page range (table-aware)."""
    parts = []
    for page_idx in range(page_start, min(page_end, len(pdf.pages))):
        text = _extract_page_text_with_tables(pdf.pages[page_idx], source_type)
        cleaned = clean_page_text(text, source_type)
        if cleaned.strip():
            parts.append(cleaned.strip())

    return "\n\n".join(parts)


def clean_page_text(text: str, source_type: str) -> str:
    """Remove artifacts from extracted PDF page text."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Preserve table markers
        if stripped in ("[TABLE]", "[/TABLE]"):
            cleaned.append(line)
            continue

        # Remove common artifacts
        if re.match(r"^pg\.\s*\d+$", stripped):
            continue
        if re.match(r"^\d+$", stripped) and len(stripped) <= 4:
            continue
        if stripped in ("Version 2", "<\u00ae>", "tut"):
            continue
        if _FOOTER_URL_RE.search(stripped):
            continue
        if stripped.startswith("\u00a9UWo") or stripped.startswith("\u00a9uwo"):
            continue
        if stripped in ("Table of Contents", "TABLE OF CONTENTS"):
            continue
        if stripped.startswith("Step 2 & 3 Notes"):
            continue

        # Source-specific cleaning
        if source_type == "bullets":
            # Remove standalone episode header lines that are just numbers
            if re.match(r"^\d{1,3}$", stripped):
                continue
        elif source_type == "font_headings":
            # Remove chapter footer lines
            if re.match(r"^[A-Z][\w &,]+\s*https?://", stripped):
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ── Main dispatch ─────────────────────────────────────────────────

def parse_pdf_sections(
    pdf_path: str,
    pdf: pdfplumber.PDF,
) -> list[Section]:
    """Parse sections from a PDF using its profile-declared layout.

    The document's PDF profile (etl/pdf_profiles/) selects the parser via its
    `layout` field: `toc`, `bullets`, or `font_headings`.
    """
    from pathlib import Path
    filename = Path(pdf_path).name

    profile = PDF_REGISTRY.resolve(filename)
    if profile is None:
        log.warning(f"No PDF profile matches {filename}, skipping")
        return []

    parsers = {
        "toc": parse_toc,
        "bullets": parse_bullets,
        "font_headings": parse_font_headings,
    }
    parser_fn = parsers.get(profile.layout)
    if parser_fn is None:
        log.warning(f"Unknown layout '{profile.layout}' for {filename}, skipping")
        return []
    sections = parser_fn(pdf, filename)

    # Populate image_count for each section
    total_images = 0
    for sec in sections:
        count = 0
        for page_idx in range(sec.page_start, min(sec.page_end, len(pdf.pages))):
            count += _count_page_images(pdf.pages[page_idx])
        sec.image_count = count
        total_images += count

    if total_images:
        sections_with_imgs = sum(1 for s in sections if s.image_count > 0)
        log.info(f"  {filename}: {total_images} images across {sections_with_imgs} sections")

    return sections
