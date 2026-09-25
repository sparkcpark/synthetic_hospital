"""ICD-10-CM 2025 dictionary with exact and fuzzy matching.

Loads the parsed CMS ICD-10-CM CSV file and provides:
1. Exact lookup by code (with dot normalization)
2. Exact case-insensitive match by description
3. Fuzzy match via token-overlap prefilter + SequenceMatcher
4. Candidate list for LLM disambiguation

No external dependencies — uses stdlib only (csv, re, difflib).
"""

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from etl.utils.logging import get_logger

log = get_logger("etl.ontology.icd10")

ICD10_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "ontology" / "icd10cm_2025.csv"


@dataclass
class ICD10Code:
    code: str          # e.g., "I21.01"
    description: str   # Long description
    is_header: bool    # True = category header (not billable)


class ICD10Dictionary:
    """In-memory ICD-10-CM dictionary for lookup and fuzzy matching."""

    def __init__(self, filepath: Path = ICD10_FILE):
        self.codes: dict[str, ICD10Code] = {}
        self._desc_index: dict[str, list[str]] = {}
        self._token_index: dict[str, set[str]] = {}
        self._load(filepath)

    def _load(self, filepath: Path):
        """Load ICD-10-CM codes from CSV."""
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row["code"]
                desc = row["description"]
                is_header = row.get("is_header", "0") == "0"  # 0 = header in CMS
                entry = ICD10Code(code=code, description=desc, is_header=is_header)
                self.codes[code] = entry
                # Description index (lowercase)
                desc_lower = desc.lower().strip()
                self._desc_index.setdefault(desc_lower, []).append(code)
                # Token index for fuzzy search
                for token in self._tokenize(desc_lower):
                    self._token_index.setdefault(token, set()).add(code)
        log.info(f"Loaded {len(self.codes)} ICD-10-CM codes "
                 f"({sum(1 for c in self.codes.values() if not c.is_header)} billable)")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into searchable tokens (min 3 chars)."""
        return [t for t in re.split(r"\W+", text.lower()) if len(t) > 2]

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize an ICD-10 code to standard format with dot."""
        code = code.strip().upper()
        # Remove existing dots for re-formatting
        code_nodot = code.replace(".", "")
        if len(code_nodot) > 3:
            return code_nodot[:3] + "." + code_nodot[3:]
        return code_nodot

    def lookup_code(self, code: str) -> ICD10Code | None:
        """Exact lookup by ICD-10 code (normalizes dots)."""
        normalized = self._normalize_code(code)
        return self.codes.get(normalized)

    def exact_match(self, description: str) -> list[ICD10Code]:
        """Exact case-insensitive match by description."""
        codes = self._desc_index.get(description.lower().strip(), [])
        return [self.codes[c] for c in codes]

    def fuzzy_match(self, query: str, top_k: int = 10,
                    min_score: float = 0.4) -> list[tuple[ICD10Code, float]]:
        """Fuzzy match by description.

        Two-stage: token-overlap prefilter → SequenceMatcher scoring.
        Returns [(ICD10Code, score), ...] sorted by score desc.
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
            key=lambda x: x[1], reverse=True,
        )[:300]  # Cap for performance

        # Stage 2: SequenceMatcher scoring
        scored = []
        query_lower = query.lower().strip()
        for code, _ in candidates:
            entry = self.codes[code]
            if entry.is_header:
                continue
            ratio = SequenceMatcher(None, query_lower, entry.description.lower()).ratio()
            if ratio >= min_score:
                scored.append((entry, ratio))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_candidates(self, query: str, top_k: int = 5) -> list[ICD10Code]:
        """Get top candidate codes for LLM disambiguation."""
        results = self.fuzzy_match(query, top_k=top_k, min_score=0.35)
        return [entry for entry, score in results]

    def validate_code(self, code: str) -> bool:
        """Check if a code exists in the dictionary (billable only)."""
        entry = self.lookup_code(code)
        return entry is not None and not entry.is_header
