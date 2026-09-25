"""SNOMED CT US Edition dictionary with validation, lookup, and ICD-10 reverse mapping.

Loads the SNOMED CT Snapshot release and provides:
1. Concept validation (active concept check)
2. Preferred term lookup
3. Exact case-insensitive match by description
4. Fuzzy match via token-overlap prefilter + SequenceMatcher
5. ICD-10 → SNOMED reverse mapping via Extended Map refset

No external dependencies — uses stdlib only (csv, re, difflib).
"""

import csv
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# Some SNOMED description rows exceed default csv field limit
csv.field_size_limit(sys.maxsize)

from etl.utils.logging import get_logger

log = get_logger("etl.ontology.snomed")

# Default SNOMED CT release path
SNOMED_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "ontology"
    / "SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20250901T120000Z"
    / "Snapshot"
)

CONCEPT_FILE = SNOMED_DIR / "Terminology" / "sct2_Concept_Snapshot_US1000124_20250901.txt"
DESCRIPTION_FILE = SNOMED_DIR / "Terminology" / "sct2_Description_Snapshot-en_US1000124_20250901.txt"
EXTENDED_MAP_FILE = SNOMED_DIR / "Refset" / "Map" / "der2_iisssccRefset_ExtendedMapSnapshot_US1000124_20250901.txt"
RELATIONSHIP_FILE = SNOMED_DIR / "Terminology" / "sct2_Relationship_Snapshot_US1000124_20250901.txt"

# Ontology release tag (for freezing/versioning the relatedness graph).
SNOMED_RELEASE = "SNOMED US 20250901"

# Concept-model relationship typeIds that express disease-disease relatedness
# (spec_v1.33 §17.5). Mapped to diagnosis_relations.edge_type names.
ISA_TYPE = "116680003"  # Is a (subtype hierarchy)
RELEVANT_REL_TYPES: dict[str, str] = {
    "42752001": "snomed_due_to",
    "47429007": "snomed_associated_with",
    "246075003": "snomed_causative_agent",
    "370135005": "snomed_pathological_process",
}

# Description typeIds
FSN_TYPE = "900000000000003001"      # Fully Specified Name
SYNONYM_TYPE = "900000000000013009"  # Synonym

# Regex to strip FSN semantic tag: "Bacterial sepsis (disorder)" → "Bacterial sepsis"
_SEMANTIC_TAG_RE = re.compile(r"\s*\([^)]+\)\s*$")


@dataclass
class SNOMEDConcept:
    concept_id: str       # e.g., "56905008"
    preferred_term: str   # FSN sans semantic tag, or best synonym
    is_active: bool


class SNOMEDDictionary:
    """In-memory SNOMED CT dictionary for validation and matching."""

    def __init__(self, snomed_dir: Path = SNOMED_DIR):
        self.concepts: dict[str, SNOMEDConcept] = {}
        self._desc_index: dict[str, list[str]] = {}   # term_lower → [concept_ids]
        self._token_index: dict[str, set[str]] = {}    # token → {concept_ids}
        self._icd10_to_snomed: dict[str, list[str]] = {}  # icd10_code → [concept_ids]

        concept_file = snomed_dir / "Terminology" / "sct2_Concept_Snapshot_US1000124_20250901.txt"
        desc_file = snomed_dir / "Terminology" / "sct2_Description_Snapshot-en_US1000124_20250901.txt"
        map_file = snomed_dir / "Refset" / "Map" / "der2_iisssccRefset_ExtendedMapSnapshot_US1000124_20250901.txt"

        self._load_concepts(concept_file)
        self._load_descriptions(desc_file)
        self._load_extended_map(map_file)

    def _load_concepts(self, filepath: Path):
        """Load active concept IDs from Concept Snapshot."""
        active_count = 0
        total = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total += 1
                cid = row["id"]
                is_active = row["active"] == "1"
                if is_active:
                    active_count += 1
                # Store all concepts but mark active status
                self.concepts[cid] = SNOMEDConcept(
                    concept_id=cid, preferred_term="", is_active=is_active
                )
        log.info(f"Loaded {active_count} active / {total} total SNOMED concepts")

    def _load_descriptions(self, filepath: Path):
        """Load descriptions, build preferred terms and search indices."""
        # First pass: collect FSN and synonyms per concept
        fsn_map: dict[str, str] = {}       # concept_id → FSN term
        synonym_map: dict[str, str] = {}   # concept_id → first synonym

        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["active"] != "1":
                    continue
                cid = row["conceptId"]
                term = row["term"]
                type_id = row["typeId"]

                # Only index active concepts
                if cid not in self.concepts or not self.concepts[cid].is_active:
                    continue

                loaded += 1

                if type_id == FSN_TYPE:
                    fsn_map[cid] = term
                elif type_id == SYNONYM_TYPE:
                    if cid not in synonym_map:
                        synonym_map[cid] = term

                # Build description index (all active terms)
                term_lower = term.lower().strip()
                # Also index FSN without semantic tag
                term_clean = _SEMANTIC_TAG_RE.sub("", term).strip().lower()

                for t in {term_lower, term_clean}:
                    if t:
                        self._desc_index.setdefault(t, []).append(cid)

                # Build token index
                for token in self._tokenize(term_clean if term_clean else term_lower):
                    self._token_index.setdefault(token, set()).add(cid)

        # Set preferred terms: FSN (sans semantic tag) > synonym
        for cid, concept in self.concepts.items():
            if not concept.is_active:
                continue
            if cid in fsn_map:
                concept.preferred_term = _SEMANTIC_TAG_RE.sub("", fsn_map[cid]).strip()
            elif cid in synonym_map:
                concept.preferred_term = synonym_map[cid]

        terms_set = sum(1 for c in self.concepts.values() if c.preferred_term)
        log.info(f"Loaded {loaded} active descriptions, "
                 f"{terms_set} concepts with preferred terms, "
                 f"{len(self._desc_index)} description index entries")

    def _load_extended_map(self, filepath: Path):
        """Load SNOMED→ICD-10 extended map and build reverse index."""
        active = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["active"] != "1":
                    continue
                icd10_code = row["mapTarget"].strip()
                snomed_id = row["referencedComponentId"].strip()
                if not icd10_code or not snomed_id:
                    continue
                active += 1
                self._icd10_to_snomed.setdefault(icd10_code, []).append(snomed_id)

        log.info(f"Loaded {active} active extended map entries, "
                 f"{len(self._icd10_to_snomed)} unique ICD-10 codes mapped")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into searchable tokens (min 3 chars)."""
        return [t for t in re.split(r"\W+", text.lower()) if len(t) > 2]

    def validate_concept(self, concept_id: str) -> bool:
        """Check if concept_id exists and is active."""
        if not concept_id:
            return False
        c = self.concepts.get(str(concept_id))
        return c is not None and c.is_active

    def lookup_concept(self, concept_id: str) -> SNOMEDConcept | None:
        """Exact lookup by SNOMED concept ID."""
        c = self.concepts.get(str(concept_id))
        if c and c.is_active:
            return c
        return None

    def get_preferred_term(self, concept_id: str) -> str | None:
        """Get preferred term for a concept (FSN sans semantic tag)."""
        c = self.lookup_concept(concept_id)
        return c.preferred_term if c and c.preferred_term else None

    def exact_match(self, name: str) -> list[SNOMEDConcept]:
        """Exact case-insensitive match by description term."""
        name_lower = name.lower().strip()
        concept_ids = self._desc_index.get(name_lower, [])
        # Deduplicate and return active concepts only
        seen = set()
        results = []
        for cid in concept_ids:
            if cid in seen:
                continue
            seen.add(cid)
            c = self.concepts.get(cid)
            if c and c.is_active:
                results.append(c)
        return results

    def fuzzy_match(self, query: str, top_k: int = 5,
                    min_score: float = 0.5) -> list[tuple[SNOMEDConcept, float]]:
        """Fuzzy match by description.

        Two-stage: token-overlap prefilter → SequenceMatcher scoring.
        Returns [(SNOMEDConcept, score), ...] sorted by score desc.
        """
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        # Stage 1: Token overlap to find candidates
        candidate_scores: dict[str, int] = {}
        for token in query_tokens:
            for cid in self._token_index.get(token, set()):
                candidate_scores[cid] = candidate_scores.get(cid, 0) + 1

        # Require at least 2 token overlaps for SNOMED (large dictionary)
        min_overlap = min(2, len(query_tokens))
        candidates = sorted(
            [(c, s) for c, s in candidate_scores.items() if s >= min_overlap],
            key=lambda x: x[1], reverse=True,
        )[:500]  # Cap for performance

        # Stage 2: SequenceMatcher scoring against preferred term
        scored = []
        query_lower = query.lower().strip()
        for cid, _ in candidates:
            concept = self.concepts.get(cid)
            if not concept or not concept.is_active or not concept.preferred_term:
                continue
            ratio = SequenceMatcher(
                None, query_lower, concept.preferred_term.lower()
            ).ratio()
            if ratio >= min_score:
                scored.append((concept, ratio))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def icd10_to_snomed(self, icd10_code: str) -> list[SNOMEDConcept]:
        """Reverse map: ICD-10 code → list of SNOMED concepts."""
        if not icd10_code:
            return []
        concept_ids = self._icd10_to_snomed.get(icd10_code.strip(), [])
        # Deduplicate and return active concepts only
        seen = set()
        results = []
        for cid in concept_ids:
            if cid in seen:
                continue
            seen.add(cid)
            c = self.concepts.get(cid)
            if c and c.is_active:
                results.append(c)
        return results


# ---------------------------------------------------------------------------
# Concept-model relationships (disease-disease relatedness; Phase B, §17.5)
# ---------------------------------------------------------------------------

@dataclass
class SNOMEDRelationships:
    """Active SNOMED concept-model edges relevant to disease-disease relatedness.

    out_edges[src]  = [(edge_type_name, dest), ...]   relevant typed edges (src→dest)
    in_edges[dest]  = [(edge_type_name, src), ...]    reverse index
    isa_parents[c]  = {direct is-a parent concept ids}

    Disease-disease relatedness in SNOMED is typically NOT a direct edge between
    two diagnoses; it is mediated by combination/complication concepts (e.g.
    "Chronic kidney disease due to type 2 diabetes mellitus" has a Due-to edge to
    T2DM and an Is-a edge to CKD). Callers derive relatedness by walking these
    mediating concepts (see the Phase B builder).
    """
    out_edges: dict[str, list[tuple[str, str]]]
    in_edges: dict[str, list[tuple[str, str]]]
    isa_parents: dict[str, set[str]]

    def related_direct(self, concept_id: str) -> set[str]:
        """Concepts directly linked to concept_id via a relevant typed edge."""
        out = {d for _, d in self.out_edges.get(concept_id, ())}
        inc = {s for _, s in self.in_edges.get(concept_id, ())}
        return out | inc

    def mediators_of(self, concept_id: str) -> set[str]:
        """Concepts that point TO concept_id via a relevant edge (e.g. the set of
        'X due to <concept>' combination concepts)."""
        return {s for _, s in self.in_edges.get(concept_id, ())}


def load_relationships(rel_file: Path = RELATIONSHIP_FILE) -> SNOMEDRelationships:
    """Parse the SNOMED Relationship Snapshot into relatedness + is-a indices."""
    out_edges: dict[str, list[tuple[str, str]]] = {}
    in_edges: dict[str, list[tuple[str, str]]] = {}
    isa_parents: dict[str, set[str]] = {}

    with open(rel_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["active"] != "1":
                continue
            type_id = row["typeId"]
            src = row["sourceId"]
            dest = row["destinationId"]
            if type_id == ISA_TYPE:
                isa_parents.setdefault(src, set()).add(dest)
            elif type_id in RELEVANT_REL_TYPES:
                name = RELEVANT_REL_TYPES[type_id]
                out_edges.setdefault(src, []).append((name, dest))
                in_edges.setdefault(dest, []).append((name, src))

    log.info(
        "Loaded SNOMED relationships: %d concepts with relevant out-edges, "
        "%d with in-edges, %d with is-a parents",
        len(out_edges), len(in_edges), len(isa_parents),
    )
    return SNOMEDRelationships(out_edges, in_edges, isa_parents)
