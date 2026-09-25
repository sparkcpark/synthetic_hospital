"""HTML → plain text with media reference extraction.

(spec §5.2 Stage 1): "Strip HTML tags from each field → store both raw and plain text"
(spec §5.2 Stage 1): "Extract media references ([sound:...], <img src='...'>)"
(spec §9): etl/parsers/html_strip.py
"""

import re
from typing import List


# Anki sound reference pattern: [sound:filename.mp3]
_SOUND_RE = re.compile(r"\[sound:([^\]]+)\]")

# HTML img tag src extraction
_IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)


def strip_html(html: str) -> str:
    """Strip HTML tags from a string, returning plain text.

    Preserves whitespace structure (newlines for <br>, <div>, <p>).
    Does NOT modify Anki-specific markup like {{c1::...}} — those are
    preserved verbatim in the plain text output.
    """
    if not html:
        return ""

    text = html

    # Replace <br>, <br/>, <br /> with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Replace block-level closing tags with newlines
    text = re.sub(r"</(?:div|p|li|tr|h[1-6])>", "\n", text, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Remove sound references from plain text (they are captured separately)
    text = _SOUND_RE.sub("", text)

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line, then overall
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text


def extract_media_refs(html: str) -> List[str]:
    """Extract media filenames from Anki HTML content.

    Returns a list of filenames referenced via [sound:...] or <img src="...">.
    """
    if not html:
        return []

    refs: List[str] = []

    # Sound references: [sound:filename.mp3]
    for match in _SOUND_RE.finditer(html):
        refs.append(match.group(1))

    # Image references: <img src="filename">
    for match in _IMG_SRC_RE.finditer(html):
        refs.append(match.group(1))

    return refs
