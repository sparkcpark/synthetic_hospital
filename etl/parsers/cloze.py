"""Cloze deletion resolution. (spec §9: etl/parsers/cloze.py)

Resolves Anki cloze markup:
  Standard:    {{c1::answer}}       -> answer
  With hint:   {{c1::answer::hint}} -> answer
  Overlapping: [[oc1::answer::hint]] -> answer  (overlapping-cloze format)
"""

import re

# Standard Anki cloze: {{c<digits>::<answer>(::optional_hint)}}
_CLOZE_RE = re.compile(r"\{\{c\d+::(.*?)(?:::[^}]*)?\}\}", re.DOTALL)

# Overlapping cloze: [[oc<digits>::<answer>(::optional_hint)]]
_OVERLAP_RE = re.compile(r"\[\[oc\d+::(.*?)(?:::[^\]]*?)?\]\]", re.DOTALL)


def resolve_cloze(text: str) -> str:
    """Replace all standard cloze deletions with their answers.

    {{c1::myocardial infarction}} -> myocardial infarction
    {{c1::answer::hint}}         -> answer
    """
    if not text:
        return ""
    return _CLOZE_RE.sub(r"\1", text).strip()


def resolve_overlapping_cloze(text: str) -> str:
    """Replace all overlapping cloze deletions with their answers.

    [[oc1::Normal pressure hydrocephalus::(most likely)]] -> Normal pressure hydrocephalus
    """
    if not text:
        return ""
    return _OVERLAP_RE.sub(r"\1", text).strip()


def resolve_all_cloze(text: str) -> str:
    """Resolve both standard and overlapping cloze formats."""
    if not text:
        return ""
    result = _CLOZE_RE.sub(r"\1", text)
    result = _OVERLAP_RE.sub(r"\1", result)
    return result.strip()


def has_cloze(text: str) -> bool:
    """Return True if text contains any cloze deletion markup."""
    if not text:
        return False
    return "{{c" in text or "[[oc" in text
