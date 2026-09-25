"""ICD-10-CM Tabular instructional-note edges (Phase B Class 1/2 sources).

Parses the CMS ICD-10-CM Tabular XML (public domain) for the cross-code
instructional notes that encode etiologic/associative relationships:

  codeFirst        -> Class 1 (the referenced code is the underlying etiology)
  useAdditionalCode-> Class 2 (associative — code the additional related condition)
  codeAlso         -> Class 2 (both conditions coded together)
  excludes2        -> Class 3 candidate hint only (weak co-occurrence; not admitted)

Each note's free text is scanned for ICD-10 code references (incl. ranges and
`X.-` categories), resolved at 3-char category granularity. Returns (code, ref,
relation_class, relation_type) tuples; the builder resolves these to diagnoses.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

TABULAR_XML = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "ontology" / "icd10cm_tabular_2025.xml"
)

# note container tag -> (relation_class, relation_type)
NOTE_MAP = {
    "codeFirst": ("1_definitional", "code_first"),
    "useAdditionalCode": ("2_associative", "use_additional"),
    "codeAlso": ("2_associative", "code_also"),
}
# Excludes2 = "not coded here but may co-occur" -> Class 3 candidate hint only.

_CODE = re.compile(r"[A-TV-Z]\d{2}[0-9A-Z]?(?:\.[0-9A-Z]+)?")


def _norm(code: str) -> str:
    return code.replace(".", "").replace("-", "").strip().upper()


def _refs(text: str) -> set[str]:
    """3-char ICD categories referenced in a note (captures ranges/categories)."""
    return {_norm(m)[:3] for m in _CODE.findall(text) if len(_norm(m)) >= 3}


def load_icd_note_edges(xml_path: Path = TABULAR_XML):
    """Return list of (diag_code_norm, ref_category, relation_class, relation_type)."""
    root = ET.parse(xml_path).getroot()
    edges = []
    excludes2_hints = 0
    for d in root.iter("diag"):
        name = d.find("name")
        if name is None or not name.text:
            continue
        code = _norm(name.text)
        cat = code[:3]
        for tag, (rclass, rtype) in NOTE_MAP.items():
            for nb in d.findall(tag):
                for note in nb.findall("note"):
                    if not note.text:
                        continue
                    for ref in _refs(note.text):
                        if ref != cat:  # skip same-category (same disease family)
                            edges.append((code, ref, rclass, rtype))
        for nb in d.findall("excludes2"):
            excludes2_hints += len(nb.findall("note"))
    return edges, excludes2_hints
