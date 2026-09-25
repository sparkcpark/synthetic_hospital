"""LOINC 2.82 dictionary with exact and fuzzy matching.

Loads the Regenstrief LOINC CSV and provides:
1. Exact lookup by LOINC number
2. Exact case-insensitive match by name (LONG_COMMON_NAME or COMPONENT)
3. Fuzzy match via token-overlap prefilter + SequenceMatcher
4. Candidate list for disambiguation

No external dependencies — uses stdlib only (csv, re, difflib).
"""

import csv
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from etl.utils.logging import get_logger

log = get_logger("etl.ontology.loinc")

LOINC_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "ontology"
    / "Loinc_2.82"
    / "LoincTable"
    / "Loinc.csv"
)

# Increase csv field size limit for LOINC descriptions
csv.field_size_limit(sys.maxsize)


@dataclass
class LOINCCode:
    loinc_num: str          # e.g., "2160-0"
    component: str          # e.g., "Creatinine"
    long_common_name: str   # e.g., "Creatinine [Mass/volume] in Serum or Plasma"
    loinc_class: str        # e.g., "CHEM"
    system: str             # Specimen: e.g., "Ser/Plas"
    scale_typ: str          # e.g., "Qn" (Quantitative)
    units: str              # e.g., "mg/dL"
    property: str           # e.g., "MCnc" (Mass concentration)


class LOINCDictionary:
    """In-memory LOINC dictionary for lookup and fuzzy matching."""

    def __init__(self, filepath: Path = LOINC_FILE):
        self.codes: dict[str, LOINCCode] = {}
        self._desc_index: dict[str, list[str]] = {}   # lowercase name → loinc_nums
        self._token_index: dict[str, set[str]] = {}    # token → loinc_nums
        self._component_index: dict[str, list[str]] = {}  # lowercase component → loinc_nums
        self._load(filepath)

    def _load(self, filepath: Path):
        """Load LOINC codes from CSV (ACTIVE only)."""
        if not filepath.exists():
            log.error(f"LOINC file not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("STATUS") != "ACTIVE":
                    continue

                loinc_num = row["LOINC_NUM"]
                component = row.get("COMPONENT", "")
                long_name = row.get("LONG_COMMON_NAME", "")

                entry = LOINCCode(
                    loinc_num=loinc_num,
                    component=component,
                    long_common_name=long_name,
                    loinc_class=row.get("CLASS", ""),
                    system=row.get("SYSTEM", ""),
                    scale_typ=row.get("SCALE_TYP", ""),
                    units=row.get("EXAMPLE_UCUM_UNITS", ""),
                    property=row.get("PROPERTY", ""),
                )
                self.codes[loinc_num] = entry

                # Description index (LONG_COMMON_NAME, lowercase)
                if long_name:
                    name_lower = long_name.lower().strip()
                    self._desc_index.setdefault(name_lower, []).append(loinc_num)

                # Component index (lowercase)
                if component:
                    comp_lower = component.lower().strip()
                    self._component_index.setdefault(comp_lower, []).append(loinc_num)
                    # Also add component to desc_index for broader matching
                    self._desc_index.setdefault(comp_lower, []).append(loinc_num)

                # Token index for fuzzy search (from both name and component)
                text = f"{long_name} {component}".lower()
                for token in self._tokenize(text):
                    self._token_index.setdefault(token, set()).add(loinc_num)

        log.info(f"Loaded {len(self.codes)} active LOINC codes")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into searchable tokens (min 3 chars)."""
        return [t for t in re.split(r"\W+", text.lower()) if len(t) > 2]

    def lookup_code(self, loinc_num: str) -> LOINCCode | None:
        """Exact lookup by LOINC number."""
        return self.codes.get(loinc_num.strip())

    def exact_match(self, name: str) -> list[LOINCCode]:
        """Exact case-insensitive match by LONG_COMMON_NAME or COMPONENT."""
        codes = self._desc_index.get(name.lower().strip(), [])
        return [self.codes[c] for c in codes]

    def component_match(self, component: str) -> list[LOINCCode]:
        """Exact case-insensitive match by COMPONENT only."""
        codes = self._component_index.get(component.lower().strip(), [])
        return [self.codes[c] for c in codes]

    def fuzzy_match(
        self, query: str, top_k: int = 10, min_score: float = 0.4
    ) -> list[tuple[LOINCCode, float]]:
        """Fuzzy match by name.

        Two-stage: token-overlap prefilter → SequenceMatcher scoring.
        Returns [(LOINCCode, score), ...] sorted by score desc.
        """
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        # Stage 1: Token overlap to find candidates
        candidate_scores: dict[str, int] = {}
        for token in query_tokens:
            for code in self._token_index.get(token, set()):
                candidate_scores[code] = candidate_scores.get(code, 0) + 1

        # Require at least 1 token overlap, sort by overlap count
        candidates = sorted(
            [(c, s) for c, s in candidate_scores.items() if s >= 1],
            key=lambda x: x[1],
            reverse=True,
        )[:500]  # Cap for performance

        # Stage 2: SequenceMatcher scoring against LONG_COMMON_NAME
        scored = []
        query_lower = query.lower().strip()
        for code, _ in candidates:
            entry = self.codes[code]
            # Score against both long name and component, take best
            ratio_name = SequenceMatcher(
                None, query_lower, entry.long_common_name.lower()
            ).ratio()
            ratio_comp = SequenceMatcher(
                None, query_lower, entry.component.lower()
            ).ratio()
            ratio = max(ratio_name, ratio_comp)
            if ratio >= min_score:
                scored.append((entry, ratio))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_candidates(self, query: str, top_k: int = 5) -> list[LOINCCode]:
        """Get top candidate codes for disambiguation."""
        results = self.fuzzy_match(query, top_k=top_k, min_score=0.35)
        return [entry for entry, score in results]

    def validate_code(self, loinc_num: str) -> bool:
        """Check if a LOINC number exists and is active."""
        return loinc_num.strip() in self.codes

    def disambiguate(self, matches: list[LOINCCode]) -> LOINCCode | None:
        """Pick the best LOINC code when multiple match.

        Preference order:
        1. Serum/Plasma system over others (most common for blood tests)
        2. Quantitative scale over ordinal
        3. Shorter LOINC number (often = more common/established code)
        """
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        def _score(entry: LOINCCode) -> tuple:
            # Higher = better
            sys_score = 2 if "Ser/Plas" in entry.system else (1 if "Bld" in entry.system else 0)
            scale_score = 2 if entry.scale_typ == "Qn" else (1 if entry.scale_typ == "Ord" else 0)
            # Prefer shorter LOINC numbers (more established)
            len_score = -len(entry.loinc_num)
            return (sys_score, scale_score, len_score)

        return max(matches, key=_score)
