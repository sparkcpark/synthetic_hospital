"""ICD-10 code validation and prefix-based correction for parse-time use.

Lazy-loads the ICD10Dictionary singleton from etl/ontology/icd10.py.
"""

_icd10_dict = None


def _get_dict():
    global _icd10_dict
    if _icd10_dict is None:
        from etl.ontology.icd10 import ICD10Dictionary
        _icd10_dict = ICD10Dictionary()
    return _icd10_dict


def is_valid_icd10(code: str) -> bool:
    """Check if normalized code exists in ICD-10-CM 2025."""
    if not code:
        return False
    return _get_dict().lookup_code(code) is not None


def nearest_valid_icd10(code: str) -> str | None:
    """Try prefix-based correction for an invalid code.

    Fallback chain:
      1. Progressive truncation from len-1 down to 4 chars → first billable match
      2. 3-char parent category (may be header)
      3. None if nothing matches
    """
    d = _get_dict()
    norm = code.replace(".", "").replace(" ", "").upper()
    if not norm or len(norm) < 3:
        return None

    # Try progressively shorter truncations (prefer billable)
    for length in range(len(norm) - 1, 3, -1):
        entry = d.lookup_code(norm[:length])
        if entry and not entry.is_header:
            return norm[:length]

    # Fall back to 3-char parent (header)
    entry = d.lookup_code(norm[:3])
    if entry:
        return norm[:3]

    return None
