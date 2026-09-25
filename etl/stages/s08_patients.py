"""Stage 8: Patient Profiling & Clustering — group board questions into patients.

Step 8a: Extract demographics, build compatibility graph, greedy set-cover clustering
         (deterministic, no LLM, ~30 seconds)
Step 8b: Generate patient profiles via LLM for each cluster
         (~1,500 calls, resume-safe via llm_call_log cache)

Usage:
    python -m etl.stages.s08_patients --step 8a [--pilot N] [--db PATH]
    python -m etl.stages.s08_patients --step 8b [--pilot N] [--workers N] [--db PATH]
    python -m etl.stages.s08_patients --all [--pilot N] [--workers N] [--db PATH]
    python -m etl.stages.s08_patients --verify-only [--db PATH]
    python -m etl.stages.s08_patients --export-csv [--db PATH]
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from etl.config import DB_PATH, DATA_DIR
from etl.db import get_connection
from etl.stages.s05_ontology import (
    MODEL,
    _call_with_retry,
)
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s08_patients")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_8A = "s08a_clustering"
STAGE_8B = "s08b_profiles"
DEFAULT_WORKERS = 5

# Age bucket boundaries and tolerances (spec §7.1 line 1511-1512)
AGE_BUCKETS = [
    ("pediatric", 0, 17),
    ("young_adult", 18, 35),
    ("mid_adult", 36, 60),
    ("older_adult", 61, 200),
]
AGE_TOLERANCE = {
    "pediatric": 2,
    "young_adult": 5,
    "mid_adult": 7,
    "older_adult": 10,
}

# Target cluster sizes by acuity type (spec §7.1 lines 1504-1507)
TARGET_SIZE = {
    "acute_only": (2, 3),
    "chronic_single": (3, 5),
    "chronic_multi": (5, 8),
}

# ---------------------------------------------------------------------------
# Regex patterns for demographics extraction
# ---------------------------------------------------------------------------

RE_AGE_YEARS = re.compile(
    r"(\d{1,3})\s*[-\s]?\s*year[\s-]*old", re.IGNORECASE
)
RE_AGE_MONTHS = re.compile(
    r"(\d{1,2})\s*[-\s]?\s*month[\s-]*old", re.IGNORECASE
)
RE_AGE_WEEKS = re.compile(
    r"(\d{1,2})\s*[-\s]?\s*week[\s-]*old", re.IGNORECASE
)
RE_AGE_DAYS = re.compile(
    r"(\d{1,2})\s*[-\s]?\s*day[\s-]*old", re.IGNORECASE
)
RE_AGE_HOURS = re.compile(
    r"(\d{1,2})\s*[-\s]?\s*hour[\s-]*old", re.IGNORECASE
)
RE_NEWBORN = re.compile(
    r"\b(?:newborn|neonate|neonatal|full[\s-]?term)\b", re.IGNORECASE
)

RE_SEX_FEMALE = re.compile(
    r"\b(?:woman|female|girl)\b", re.IGNORECASE
)
RE_SEX_MALE = re.compile(
    r"\b(?:(?<!\w)man\b|(?<![wo])male\b|boy)\b", re.IGNORECASE
)
RE_OBSTETRIC = re.compile(
    r"\b(?:gravida|G\d+P\d+|gestation|pregnant)\b", re.IGNORECASE
)
RE_PRONOUN_F = re.compile(r"\b(?:she|her)\b", re.IGNORECASE)
RE_PRONOUN_M = re.compile(r"\b(?:he|his)\b", re.IGNORECASE)

RE_RACE = re.compile(
    r"\b(African[\s-]?American|Caucasian|Hispanic|Latino|Latina|Asian|"
    r"White|Black|Native[\s-]?American|Pacific[\s-]?Islander)\b",
    re.IGNORECASE,
)

# Smoking status patterns
RE_SMOKING_NEVER = re.compile(
    r"\b(?:never\s+smok|non[\s-]?smok|nonsmoker|does\s+not\s+smoke|"
    r"denies\s+(?:any\s+)?(?:tobacco|smok))",
    re.IGNORECASE,
)
RE_SMOKING_CURRENT = re.compile(
    r"\b(?:smokes?\s+\d|pack[\s-]?year|packs?\s+(?:per|a)\s+day|"
    r"current(?:ly)?\s+smok|active\s+smok)",
    re.IGNORECASE,
)
RE_SMOKING_FORMER = re.compile(
    r"\b(?:quit\s+smok|former\s+smok|stopped\s+smok|"
    r"ex[\s-]?smoker)",
    re.IGNORECASE,
)

# Alcohol status patterns
RE_ALCOHOL_NONE = re.compile(
    r"\b(?:does\s+not\s+(?:drink|consume)\s+alcohol|denies\s+(?:any\s+)?alcohol|"
    r"no\s+alcohol|teetotal|abstinent)",
    re.IGNORECASE,
)
RE_ALCOHOL_HEAVY = re.compile(
    r"\b(?:heavy\s+(?:drink|alcohol)|alcohol(?:ic|ism)?|"
    r"drinks?\s+\d+\s+(?:beers?|glasses?|bottles?|drinks?)\s+(?:per|a|every)\s+day|"
    r"alcohol\s+(?:abuse|depend))",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Organ system normalization (54 raw values → ~20 canonical)
# ---------------------------------------------------------------------------

ORGAN_SYSTEM_CANONICAL: dict[str, str] = {
    # Cardiovascular
    "Cardiovascular System": "cardiovascular",
    "cardiovascular system": "cardiovascular",
    # Nervous System
    "Nervous System": "nervous_system",
    "neurology": "nervous_system",
    # Psychiatry
    "PsychiatricBehavioral & Substance Abuse": "psychiatry",
    "psychiatry": "psychiatry",
    # GI
    "Gastrointestinal & Nutrition": "gi",
    "gastroenterology": "gi",
    "gastrointestinal system": "gi",
    # Infectious Disease
    "Infectious Diseases": "infectious",
    "infectious diseases": "infectious",
    "infectious disease": "infectious",
    # Pulmonary
    "Pulmonary & Critical Care": "pulmonary",
    "pulmonary and critical care": "pulmonary",
    "respiratory system": "pulmonary",
    # Obstetrics
    "Pregnancy Childbirth & Puerperium": "obstetrics",
    # Female Reproductive
    "Female Reproductive System & Breast": "reproductive_female",
    "reproductive female": "reproductive_female",
    # Male Reproductive
    "Male Reproductive System": "reproductive_male",
    "reproductive male": "reproductive_male",
    # Reproductive (general)
    "reproductive system": "reproductive_general",
    # MSK / Rheumatology
    "RheumatologyOrthopedics & Sports": "msk",
    "musculoskeletal": "msk",
    "rheumatology": "msk",
    # Renal
    "Renal Urinary Systems & Electrolytes": "renal",
    "renal urinary systems and electrolytes": "renal",
    # Heme/Onc
    "Hematology & Oncology": "heme_onc",
    "hematology and oncology": "heme_onc",
    "hematology": "heme_onc",
    # Endocrine
    "Endocrine Diabetes & Metabolism": "endocrine",
    "endocrine diabetes and metabolism": "endocrine",
    # Dermatology
    "Dermatology": "derm",
    "dermatology": "derm",
    # ENT
    "Ear Nose & Throat ENT": "ent",
    "ear nose and throat ent": "ent",
    # Ophthalmology
    "Ophthalmology": "ophthalmology",
    "ophthalmology": "ophthalmology",
    # Toxicology
    "Poisoning & Environmental Exposure": "toxicology",
    "poisoning and environmental exposure": "toxicology",
    # Immunology
    "Allergy & Immunology": "immunology",
    "immunology": "immunology",
    # General Principles (non-organ-system disciplines)
    "pharmacology general principles": "pharmacology",
    "pharmacology": "pharmacology",
    "biochemistry general principles": "biochemistry",
    "genetics general principles": "genetics",
    "microbiology general principles": "microbiology",
    "microbiology": "microbiology",
    "pathology general principles": "pathology",
    # Biostats / Social Sciences
    "biostatistics and epidemiology": "biostatistics",
    "Biostatistics & Epidemiology": "biostatistics",
    "social sciences ethics legal professional": "social_sciences",
    "Social Sciences EthicsLegalProfessional": "social_sciences",
    # Multi-system / General
    "Miscellaneous Multisystem": "multisystem",
    "General Principles": "general",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QuestionDemographics:
    question_id: int
    age_years: float | None = None
    sex: str | None = None          # 'M' | 'F'
    age_bucket: str | None = None
    race_ethnicity: str | None = None
    extraction_source: str = "none"


@dataclass
class QuestionNode:
    question_id: int
    demographics: QuestionDemographics
    organ_systems: set[str] = field(default_factory=set)
    correct_dx_ids: set[int] = field(default_factory=set)
    secondary_dx_ids: set[int] = field(default_factory=set)
    all_dx_ids: set[int] = field(default_factory=set)
    acuity: str = "acute"
    smoking_status: str | None = None
    alcohol_status: str | None = None
    clustered: bool = False


# ---------------------------------------------------------------------------
# Demographics extraction helpers
# ---------------------------------------------------------------------------


def _extract_age(text: str) -> float | None:
    """Extract age in years from demographics/vignette text."""
    m = RE_AGE_YEARS.search(text)
    if m:
        return float(m.group(1))
    m = RE_AGE_MONTHS.search(text)
    if m:
        return float(m.group(1)) / 12.0
    m = RE_AGE_WEEKS.search(text)
    if m:
        return float(m.group(1)) / 52.0
    m = RE_AGE_DAYS.search(text)
    if m:
        return float(m.group(1)) / 365.0
    m = RE_AGE_HOURS.search(text)
    if m:
        return 0.0
    if RE_NEWBORN.search(text):
        return 0.0
    return None


def _extract_sex(text: str) -> str | None:
    """Extract sex from demographics/vignette text."""
    if RE_SEX_FEMALE.search(text):
        return "F"
    if RE_SEX_MALE.search(text):
        return "M"
    if RE_OBSTETRIC.search(text):
        return "F"
    # Pronoun fallback
    if RE_PRONOUN_F.search(text):
        return "F"
    if RE_PRONOUN_M.search(text):
        return "M"
    return None


def _extract_race(text: str) -> str | None:
    """Extract race/ethnicity mention if present."""
    m = RE_RACE.search(text)
    return m.group(1) if m else None


def _age_to_bucket(age_years: float | None) -> str | None:
    """Map age to age bucket per spec §7.1."""
    if age_years is None:
        return None
    for name, lo, hi in AGE_BUCKETS:
        if lo <= age_years <= hi:
            return name
    return None


def _extract_smoking(text: str) -> str | None:
    """Extract smoking status from social_history text."""
    if RE_SMOKING_NEVER.search(text):
        return "never"
    if RE_SMOKING_FORMER.search(text):
        return "former"
    if RE_SMOKING_CURRENT.search(text):
        return "current"
    return None


def _extract_alcohol(text: str) -> str | None:
    """Extract alcohol status from social_history text."""
    if RE_ALCOHOL_NONE.search(text):
        return "none"
    if RE_ALCOHOL_HEAVY.search(text):
        return "heavy"
    return None


def extract_demographics(
    conn: sqlite3.Connection,
) -> dict[int, QuestionDemographics]:
    """Extract demographics for all board questions.

    Strategy:
    1. Parse demographics EHR section (6,936 questions)
    2. Fallback to vignette_text for remaining 67 questions
    3. For sex: also try pronoun detection in vignette if still None
    """
    # Get all question IDs
    all_qids = {
        r[0] for r in
        conn.execute("SELECT question_id FROM board_questions").fetchall()
    }

    demographics: dict[int, QuestionDemographics] = {}

    # Phase 1: Extract from demographics EHR sections
    rows = conn.execute(
        "SELECT es.question_id, es.section_text "
        "FROM ehr_sections es WHERE es.section_type = 'demographics'"
    ).fetchall()

    for qid, text in rows:
        demo = QuestionDemographics(question_id=qid)
        demo.age_years = _extract_age(text)
        demo.sex = _extract_sex(text)
        demo.race_ethnicity = _extract_race(text)
        demo.age_bucket = _age_to_bucket(demo.age_years)
        demo.extraction_source = "demographics"
        demographics[qid] = demo

    phase1_count = len(demographics)
    log.info(f"Demographics phase 1: {phase1_count} from EHR sections")

    # Phase 2: Fallback to vignette_text for questions without demographics section
    missing_qids = all_qids - set(demographics.keys())
    if missing_qids:
        placeholders = ",".join("?" * len(missing_qids))
        rows = conn.execute(
            f"SELECT question_id, vignette_text FROM board_questions "
            f"WHERE question_id IN ({placeholders}) AND vignette_text IS NOT NULL",
            list(missing_qids),
        ).fetchall()
        for qid, vtext in rows:
            if not vtext:
                continue
            demo = QuestionDemographics(question_id=qid)
            demo.age_years = _extract_age(vtext)
            demo.sex = _extract_sex(vtext)
            demo.race_ethnicity = _extract_race(vtext)
            demo.age_bucket = _age_to_bucket(demo.age_years)
            demo.extraction_source = "vignette_fallback"
            demographics[qid] = demo

    log.info(
        f"Demographics phase 2: {len(demographics) - phase1_count} from vignette fallback"
    )

    # Phase 3: For questions with sex=None, try vignette pronoun detection
    sex_none_count = 0
    for qid, demo in demographics.items():
        if demo.sex is not None:
            continue
        vrow = conn.execute(
            "SELECT vignette_text FROM board_questions WHERE question_id = ?",
            (qid,),
        ).fetchone()
        if vrow and vrow[0]:
            sex = _extract_sex(vrow[0])
            if sex:
                demo.sex = sex
                sex_none_count += 1

    log.info(f"Demographics phase 3: {sex_none_count} sex resolved via vignette pronouns")

    # Remaining questions with no data at all
    still_missing = all_qids - set(demographics.keys())
    for qid in still_missing:
        demographics[qid] = QuestionDemographics(question_id=qid)

    # Stats
    has_age = sum(1 for d in demographics.values() if d.age_years is not None)
    has_sex = sum(1 for d in demographics.values() if d.sex is not None)
    has_both = sum(
        1 for d in demographics.values()
        if d.age_years is not None and d.sex is not None
    )
    log.info(
        f"Demographics extracted: {has_age} with age, {has_sex} with sex, "
        f"{has_both} with both (of {len(demographics)} total)"
    )

    return demographics


# ---------------------------------------------------------------------------
# QuestionNode builder
# ---------------------------------------------------------------------------


def _build_question_nodes(
    conn: sqlite3.Connection,
    demographics: dict[int, QuestionDemographics],
    include_distractors: bool = False,
) -> dict[int, QuestionNode]:
    """Build QuestionNode objects with organ_system, dx, acuity, smoking, alcohol.

    Args:
        include_distractors: If True, include distractor diagnoses in all_dx_ids
            for clustering. Creates noisier but larger clusters.
    """

    # Organ system lookup
    organ_rows = conn.execute(
        "SELECT question_id, organ_system FROM board_questions"
    ).fetchall()
    organ_map: dict[int, set[str]] = {}
    for qid, os_raw in organ_rows:
        if os_raw:
            canonical = ORGAN_SYSTEM_CANONICAL.get(os_raw, os_raw.lower())
            organ_map.setdefault(qid, set()).add(canonical)

    # Diagnosis IDs by role
    dx_rows = conn.execute(
        "SELECT qd.question_id, qd.diagnosis_id, qd.role "
        "FROM question_diagnoses qd"
    ).fetchall()
    correct_dx: dict[int, set[int]] = {}
    secondary_dx: dict[int, set[int]] = {}
    distractor_dx: dict[int, set[int]] = {}
    for qid, dx_id, role in dx_rows:
        if role == "correct":
            correct_dx.setdefault(qid, set()).add(dx_id)
        elif role == "secondary":
            secondary_dx.setdefault(qid, set()).add(dx_id)
        elif role == "distractor" and include_distractors:
            distractor_dx.setdefault(qid, set()).add(dx_id)

    # Acuity lookup from diagnoses table
    acuity_map: dict[int, str] = {}
    for dx_id, acuity in conn.execute(
        "SELECT diagnosis_id, acuity FROM diagnoses WHERE acuity IS NOT NULL"
    ).fetchall():
        acuity_map[dx_id] = acuity

    # Social history sections for smoking/alcohol
    sh_rows = conn.execute(
        "SELECT question_id, section_text FROM ehr_sections "
        "WHERE section_type = 'social_history'"
    ).fetchall()
    smoking_map: dict[int, str | None] = {}
    alcohol_map: dict[int, str | None] = {}
    for qid, text in sh_rows:
        if qid not in smoking_map:
            smoking_map[qid] = _extract_smoking(text)
        if qid not in alcohol_map:
            alcohol_map[qid] = _extract_alcohol(text)

    # Build nodes
    nodes: dict[int, QuestionNode] = {}
    for qid, demo in demographics.items():
        c_dx = correct_dx.get(qid, set())
        s_dx = secondary_dx.get(qid, set())

        # Determine acuity from correct diagnosis
        acuities = {acuity_map.get(dx_id, "unspecified") for dx_id in c_dx}
        if "chronic" in acuities or "acute_on_chronic" in acuities:
            node_acuity = "chronic" if "acute" not in acuities else "mixed"
        else:
            node_acuity = "acute"

        d_dx = distractor_dx.get(qid, set()) if include_distractors else set()
        nodes[qid] = QuestionNode(
            question_id=qid,
            demographics=demo,
            organ_systems=organ_map.get(qid, set()),
            correct_dx_ids=c_dx,
            secondary_dx_ids=s_dx,
            all_dx_ids=c_dx | s_dx | d_dx,
            acuity=node_acuity,
            smoking_status=smoking_map.get(qid),
            alcohol_status=alcohol_map.get(qid),
        )

    log.info(
        f"Built {len(nodes)} QuestionNodes "
        f"({sum(1 for n in nodes.values() if n.acuity == 'chronic')} chronic, "
        f"{sum(1 for n in nodes.values() if n.smoking_status is not None)} with smoking, "
        f"{sum(1 for n in nodes.values() if n.alcohol_status is not None)} with alcohol)"
    )
    return nodes


# ---------------------------------------------------------------------------
# Compatibility check + contradiction detection
# ---------------------------------------------------------------------------


def _has_contradiction(a: QuestionNode, b: QuestionNode) -> bool:
    """Check for contradictory findings between two questions."""
    # Smoking: never ↔ current, never ↔ former
    if a.smoking_status and b.smoking_status:
        pair = tuple(sorted([a.smoking_status, b.smoking_status]))
        if pair in {("current", "never"), ("former", "never")}:
            return True

    # Alcohol: none ↔ heavy
    if a.alcohol_status and b.alcohol_status:
        pair = tuple(sorted([a.alcohol_status, b.alcohol_status]))
        if pair == ("heavy", "none"):
            return True

    return False


def _demographics_compatible(a: QuestionNode, b: QuestionNode) -> bool:
    """Check demographic compatibility (rules 1,2,3,5). Must hold for ALL pairs."""
    d_a, d_b = a.demographics, b.demographics

    # Rule 1: Same sex (both non-None)
    if d_a.sex is None or d_b.sex is None:
        return False
    if d_a.sex != d_b.sex:
        return False

    # Rule 2: Same age bucket
    if d_a.age_bucket is None or d_b.age_bucket is None:
        return False
    if d_a.age_bucket != d_b.age_bucket:
        return False

    # Rule 3: Within-bucket age tolerance
    if d_a.age_years is not None and d_b.age_years is not None:
        tolerance = AGE_TOLERANCE[d_a.age_bucket]
        if abs(d_a.age_years - d_b.age_years) > tolerance:
            return False

    # Rule 5: No contradictory findings
    if _has_contradiction(a, b):
        return False

    return True


def _shares_diagnosis(a: QuestionNode, b: QuestionNode) -> bool:
    """Check if two questions share at least one diagnosis (correct or secondary)."""
    return bool(a.all_dx_ids & b.all_dx_ids)


# ---------------------------------------------------------------------------
# Greedy set-cover clustering
# ---------------------------------------------------------------------------


def _classify_cluster_acuity(nodes_list: list[QuestionNode]) -> str:
    """Classify cluster type from diagnosis acuities."""
    has_chronic = any(n.acuity in ("chronic", "mixed") for n in nodes_list)
    organ_systems: set[str] = set()
    for n in nodes_list:
        organ_systems.update(n.organ_systems)

    if not has_chronic:
        return "acute_only"
    # Filter out non-clinical organ systems for multi-system check
    clinical_systems = {
        s for s in organ_systems
        if s not in {
            "pharmacology", "biochemistry", "genetics", "microbiology",
            "pathology", "biostatistics", "social_sciences", "general",
        }
    }
    if len(clinical_systems) <= 1:
        return "chronic_single"
    return "chronic_multi"


def _build_partitions(
    nodes: dict[int, QuestionNode],
) -> dict[tuple[str, str], list[int]]:
    """Group question_ids by (sex, age_bucket). Only compatible within groups."""
    partitions: dict[tuple[str, str], list[int]] = {}
    excluded = 0
    for qid, node in nodes.items():
        d = node.demographics
        if d.sex and d.age_bucket:
            key = (d.sex, d.age_bucket)
            partitions.setdefault(key, []).append(qid)
        else:
            excluded += 1

    log.info(
        f"Partitions: {len(partitions)} groups, "
        f"{sum(len(v) for v in partitions.values())} questions partitioned, "
        f"{excluded} excluded (no sex or age_bucket)"
    )
    for key, qids in sorted(partitions.items(), key=lambda x: -len(x[1])):
        log.info(f"  {key}: {len(qids)} questions")

    return partitions


def _greedy_cluster(
    partition_qids: list[int],
    nodes: dict[int, QuestionNode],
) -> list[list[int]]:
    """Greedy set-cover clustering within a (sex, age_bucket) partition.

    Two-level compatibility:
    - Demographics (rules 1,2,3,5): must hold for ALL pairs (clique)
    - Shared diagnosis (rule 4): must hold for at least ONE existing member (connected chain)
    """
    clusters: list[list[int]] = []
    unclustered = set(partition_qids)

    # Precompute neighbor counts: must satisfy BOTH demographic + shared dx
    neighbor_counts: dict[int, int] = {}
    for qid_a in partition_qids:
        count = 0
        node_a = nodes[qid_a]
        for qid_b in partition_qids:
            if qid_a != qid_b:
                node_b = nodes[qid_b]
                if _demographics_compatible(node_a, node_b) and _shares_diagnosis(node_a, node_b):
                    count += 1
        neighbor_counts[qid_a] = count

    # Sort by descending neighbor count
    seed_order = sorted(partition_qids, key=lambda q: -neighbor_counts.get(q, 0))

    for seed_qid in seed_order:
        if seed_qid not in unclustered:
            continue

        seed_node = nodes[seed_qid]
        cluster = [seed_qid]

        # Initial target size from seed's acuity
        acuity_type = _classify_cluster_acuity([seed_node])
        _, max_size = TARGET_SIZE[acuity_type]

        # Find candidates: demographically compatible with seed AND share a diagnosis with seed
        candidates = []
        for cand_qid in unclustered:
            if cand_qid == seed_qid:
                continue
            cand_node = nodes[cand_qid]
            if _demographics_compatible(seed_node, cand_node) and _shares_diagnosis(seed_node, cand_node):
                candidates.append(cand_qid)

        # Sort by shared diagnosis count desc
        candidates.sort(key=lambda q: (
            -len(nodes[q].all_dx_ids & seed_node.all_dx_ids),
        ))

        # Greedily add candidates:
        #   - demographics_compatible with ALL existing members (clique)
        #   - shares_diagnosis with at least ONE existing member (connected chain)
        for cand_qid in candidates:
            if len(cluster) >= max_size:
                break
            cand_node = nodes[cand_qid]
            # Clique check: demographics compatible with every member
            if not all(_demographics_compatible(nodes[c], cand_node) for c in cluster):
                continue
            # Connected check: shares diagnosis with at least one member
            if not any(_shares_diagnosis(nodes[c], cand_node) for c in cluster):
                continue
            cluster.append(cand_qid)
            # Re-classify acuity with expanded cluster
            acuity_type = _classify_cluster_acuity(
                [nodes[q] for q in cluster]
            )
            _, max_size = TARGET_SIZE[acuity_type]

        # Only keep clusters with >= 2 members
        if len(cluster) >= 2:
            clusters.append(cluster)
            for qid in cluster:
                unclustered.discard(qid)
                nodes[qid].clustered = True

    return clusters


# ---------------------------------------------------------------------------
# Stage 8a orchestrator
# ---------------------------------------------------------------------------


def run_8a(
    conn: sqlite3.Connection,
    pilot: int | None = None,
) -> dict:
    """Stage 8a: Extract demographics, build compatibility graph, cluster.

    Phase 1: Extract demographics
    Phase 2: Build QuestionNode objects
    Phase 3: Partition by (sex, age_bucket)
    Phase 4: Greedy cluster within each partition
    Phase 5: Insert into longitudinal_patients + longitudinal_encounters
    """
    t0 = time.time()

    # Phase 1: Extract demographics
    log.info("Phase 1: Extracting demographics...")
    demographics = extract_demographics(conn)

    if pilot:
        # Limit to first N questions
        limited = dict(list(demographics.items())[:pilot])
        demographics = limited
        log.info(f"Pilot mode: limited to {len(demographics)} questions")

    # Phase 2: Build nodes
    log.info("Phase 2: Building QuestionNode objects...")
    nodes = _build_question_nodes(conn, demographics)

    # Phase 3: Partition
    log.info("Phase 3: Partitioning by (sex, age_bucket)...")
    partitions = _build_partitions(nodes)

    # Phase 4: Cluster
    log.info("Phase 4: Greedy set-cover clustering...")
    all_clusters: list[list[int]] = []
    for (sex, bucket), qids in sorted(partitions.items(), key=lambda x: -len(x[1])):
        log.info(f"  Clustering ({sex}, {bucket}): {len(qids)} questions...")
        clusters = _greedy_cluster(qids, nodes)
        all_clusters.extend(clusters)
        clustered_count = sum(len(c) for c in clusters)
        log.info(
            f"  → {len(clusters)} clusters, {clustered_count} questions clustered, "
            f"{len(qids) - clustered_count} singletons"
        )

    total_clustered = sum(len(c) for c in all_clusters)
    total_singletons = len(demographics) - total_clustered
    log.info(
        f"Clustering complete: {len(all_clusters)} patients, "
        f"{total_clustered} questions clustered, {total_singletons} singletons"
    )

    # Cluster size distribution
    size_dist: dict[int, int] = {}
    for c in all_clusters:
        size_dist[len(c)] = size_dist.get(len(c), 0) + 1
    for size in sorted(size_dist):
        log.info(f"  Size {size}: {size_dist[size]} clusters")

    # Phase 5: Insert into DB
    log.info("Phase 5: Inserting into longitudinal_patients + longitudinal_encounters...")

    # Delete-before-insert (FK order: encounters first, then patients)
    old_enc = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters"
    ).fetchone()[0]
    old_pat = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_patients"
    ).fetchone()[0]
    if old_enc or old_pat:
        conn.execute("DELETE FROM encounter_ehr_sections")
        conn.execute("DELETE FROM longitudinal_encounters")
        conn.execute("DELETE FROM longitudinal_patients")
        conn.commit()
        log.info(f"Cleared {old_pat} patients, {old_enc} encounters")

    # Build diagnosis name lookup for primary_diagnoses field
    dx_name_map: dict[int, str] = {}
    for dx_id, name in conn.execute(
        "SELECT diagnosis_id, display_name FROM diagnoses"
    ).fetchall():
        dx_name_map[dx_id] = name

    patients_inserted = 0
    encounters_inserted = 0

    for cluster in all_clusters:
        seed_qid = cluster[0]
        seed_node = nodes[seed_qid]

        # Compute patient-level fields
        ages = [
            nodes[q].demographics.age_years
            for q in cluster
            if nodes[q].demographics.age_years is not None
        ]
        patient_age = int(median(ages)) if ages else None
        patient_sex = seed_node.demographics.sex

        # Collect correct dx names
        all_correct_dx: list[str] = []
        for qid in cluster:
            for dx_id in nodes[qid].correct_dx_ids:
                name = dx_name_map.get(dx_id)
                if name and name not in all_correct_dx:
                    all_correct_dx.append(name)

        # Collect shared secondary dx as comorbidities
        shared_secondary: set[int] = set()
        if len(cluster) >= 2:
            for i, qid_a in enumerate(cluster):
                for qid_b in cluster[i + 1:]:
                    shared = nodes[qid_a].secondary_dx_ids & nodes[qid_b].secondary_dx_ids
                    shared_secondary.update(shared)
        comorbidity_names = [
            dx_name_map[dx_id]
            for dx_id in shared_secondary
            if dx_id in dx_name_map
        ]

        cursor = conn.execute(
            """INSERT INTO longitudinal_patients
               (profile, age, sex, num_encounters, primary_diagnoses,
                comorbidities, generation_seed)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "{}",  # Empty profile, filled by 8b
                patient_age,
                patient_sex,
                len(cluster),
                json.dumps(all_correct_dx),
                json.dumps(comorbidity_names),
                seed_qid,  # generation_seed = seed question_id
            ),
        )
        patient_id = cursor.lastrowid
        patients_inserted += 1

        for order, qid in enumerate(cluster):
            conn.execute(
                """INSERT INTO longitudinal_encounters
                   (patient_id, encounter_date, encounter_type,
                    source_question_ids, encounter_order,
                    generation_method)
                   VALUES (?, '1970-01-01', 'outpatient', ?, ?, 'template')""",
                (
                    patient_id,
                    json.dumps([qid]),
                    order,
                ),
            )
            encounters_inserted += 1

    conn.commit()

    duration = time.time() - t0
    summary = {
        "step": "8a",
        "questions_total": len(demographics),
        "patients": patients_inserted,
        "encounters": encounters_inserted,
        "singletons": total_singletons,
        "cluster_size_distribution": {
            str(k): v for k, v in sorted(size_dist.items())
        },
        "duration_sec": round(duration, 1),
    }
    log.info(f"Stage 8a complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# Stage 8a-distractor: Clustering with distractor diagnoses
# ---------------------------------------------------------------------------

_DISTRACTOR_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS longitudinal_patients_distractor (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    profile         TEXT NOT NULL,
    age             INTEGER,
    sex             TEXT CHECK (sex IN ('M', 'F')),
    race_ethnicity  TEXT,
    insurance       TEXT,
    pcp_name        TEXT,
    num_encounters  INTEGER DEFAULT 0,
    primary_diagnoses TEXT,
    comorbidities   TEXT,
    generation_seed INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS longitudinal_encounters_distractor (
    encounter_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES longitudinal_patients_distractor(patient_id),
    encounter_date  TEXT NOT NULL,
    encounter_type  TEXT NOT NULL CHECK (encounter_type IN (
        'outpatient', 'ed', 'inpatient', 'icu', 'telehealth', 'procedure', 'follow_up'
    )),
    chief_complaint TEXT,
    attending_name  TEXT,
    department      TEXT,
    source_question_ids TEXT,
    encounter_order INTEGER NOT NULL,
    note_text       TEXT,
    generation_method TEXT CHECK (generation_method IN ('template', 'llm', 'hybrid')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


def run_8a_distractor(
    conn: sqlite3.Connection,
    pilot: int | None = None,
) -> dict:
    """Stage 8a-distractor: Cluster with distractor diagnoses in the dx pool.

    Same algorithm as run_8a but includes distractor diagnoses in all_dx_ids,
    creating larger/noisier clusters for harder benchmark evaluation.
    Results stored in separate tables: longitudinal_patients_distractor,
    longitudinal_encounters_distractor.
    """
    t0 = time.time()

    # Create tables if needed
    conn.executescript(_DISTRACTOR_TABLES_DDL)
    conn.commit()

    # Phase 1: Extract demographics
    log.info("[distractor] Phase 1: Extracting demographics...")
    demographics = extract_demographics(conn)

    if pilot:
        limited = dict(list(demographics.items())[:pilot])
        demographics = limited
        log.info(f"[distractor] Pilot mode: limited to {len(demographics)} questions")

    # Phase 2: Build nodes WITH distractors
    log.info("[distractor] Phase 2: Building QuestionNode objects (with distractors)...")
    nodes = _build_question_nodes(conn, demographics, include_distractors=True)

    # Phase 3: Partition
    log.info("[distractor] Phase 3: Partitioning by (sex, age_bucket)...")
    partitions = _build_partitions(nodes)

    # Phase 4: Cluster
    log.info("[distractor] Phase 4: Greedy set-cover clustering...")
    all_clusters: list[list[int]] = []
    for (sex, bucket), qids in sorted(partitions.items(), key=lambda x: -len(x[1])):
        log.info(f"  [distractor] Clustering ({sex}, {bucket}): {len(qids)} questions...")
        clusters = _greedy_cluster(qids, nodes)
        all_clusters.extend(clusters)
        clustered_count = sum(len(c) for c in clusters)
        log.info(
            f"  → {len(clusters)} clusters, {clustered_count} questions clustered, "
            f"{len(qids) - clustered_count} singletons"
        )

    total_clustered = sum(len(c) for c in all_clusters)
    total_singletons = len(demographics) - total_clustered
    log.info(
        f"[distractor] Clustering complete: {len(all_clusters)} patients, "
        f"{total_clustered} questions clustered, {total_singletons} singletons"
    )

    size_dist: dict[int, int] = {}
    for c in all_clusters:
        size_dist[len(c)] = size_dist.get(len(c), 0) + 1
    for size in sorted(size_dist):
        log.info(f"  Size {size}: {size_dist[size]} clusters")

    # Phase 5: Insert into distractor tables
    log.info("[distractor] Phase 5: Inserting into distractor tables...")

    old_enc = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters_distractor"
    ).fetchone()[0]
    old_pat = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_patients_distractor"
    ).fetchone()[0]
    if old_enc or old_pat:
        conn.execute("DELETE FROM longitudinal_encounters_distractor")
        conn.execute("DELETE FROM longitudinal_patients_distractor")
        conn.commit()
        log.info(f"[distractor] Cleared {old_pat} patients, {old_enc} encounters")

    dx_name_map: dict[int, str] = {}
    for dx_id, name in conn.execute(
        "SELECT diagnosis_id, display_name FROM diagnoses"
    ).fetchall():
        dx_name_map[dx_id] = name

    patients_inserted = 0
    encounters_inserted = 0

    for cluster in all_clusters:
        seed_qid = cluster[0]
        seed_node = nodes[seed_qid]

        ages = [
            nodes[q].demographics.age_years
            for q in cluster
            if nodes[q].demographics.age_years is not None
        ]
        patient_age = int(median(ages)) if ages else None
        patient_sex = seed_node.demographics.sex

        all_correct_dx: list[str] = []
        for qid in cluster:
            for dx_id in nodes[qid].correct_dx_ids:
                name = dx_name_map.get(dx_id)
                if name and name not in all_correct_dx:
                    all_correct_dx.append(name)

        shared_secondary: set[int] = set()
        if len(cluster) >= 2:
            for i, qid_a in enumerate(cluster):
                for qid_b in cluster[i + 1:]:
                    shared = nodes[qid_a].secondary_dx_ids & nodes[qid_b].secondary_dx_ids
                    shared_secondary.update(shared)
        comorbidity_names = [
            dx_name_map[dx_id]
            for dx_id in shared_secondary
            if dx_id in dx_name_map
        ]

        cursor = conn.execute(
            """INSERT INTO longitudinal_patients_distractor
               (profile, age, sex, num_encounters, primary_diagnoses,
                comorbidities, generation_seed)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "{}",
                patient_age,
                patient_sex,
                len(cluster),
                json.dumps(all_correct_dx),
                json.dumps(comorbidity_names),
                seed_qid,
            ),
        )
        patient_id = cursor.lastrowid
        patients_inserted += 1

        for order, qid in enumerate(cluster):
            conn.execute(
                """INSERT INTO longitudinal_encounters_distractor
                   (patient_id, encounter_date, encounter_type,
                    source_question_ids, encounter_order,
                    generation_method)
                   VALUES (?, '1970-01-01', 'outpatient', ?, ?, 'template')""",
                (
                    patient_id,
                    json.dumps([qid]),
                    order,
                ),
            )
            encounters_inserted += 1

    conn.commit()

    duration = time.time() - t0
    summary = {
        "step": "8a-distractor",
        "questions_total": len(demographics),
        "patients": patients_inserted,
        "encounters": encounters_inserted,
        "singletons": total_singletons,
        "cluster_size_distribution": {
            str(k): v for k, v in sorted(size_dist.items())
        },
        "duration_sec": round(duration, 1),
    }
    log.info(f"Stage 8a-distractor complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# Stage 8b: LLM Patient Profile Generation
# ---------------------------------------------------------------------------

PROFILE_PROMPT = """You are a clinical documentation specialist creating a synthetic patient profile for a medical education simulation.

Given the following clinical encounters (from USMLE-style board questions), create a SINGLE coherent patient profile that could plausibly be the same person across all encounters.

PATIENT CONSTRAINTS:
- Age: {age} years old
- Sex: {sex}

ENCOUNTERS:
{encounters_block}

CORRECT DIAGNOSES across encounters:
{diagnoses_list}

SOCIAL HISTORY EXCERPTS:
{social_history_block}

REQUIREMENTS:
1. The profile must be consistent with ALL encounters — do not contradict any source question
2. age_at_first_encounter should be {age}
3. sex must be "{sex}"
4. chronic_conditions should include background comorbidities appearing across encounters, NOT the specific acute diagnoses being tested
5. home_medications should be clinically appropriate for the chronic conditions
6. family_history should be plausible given the diagnoses
7. smoking_status and alcohol_use must not contradict any social history excerpts provided
8. Generate realistic but fictional details for occupation, insurance, race_ethnicity

Return ONLY valid JSON (no markdown, no explanation):
{{
  "age_at_first_encounter": {age},
  "sex": "{sex}",
  "race_ethnicity": "...",
  "occupation": "...",
  "insurance": "...",
  "smoking_status": "...",
  "alcohol_use": "...",
  "chronic_conditions": ["..."],
  "surgical_history": ["..."],
  "family_history": ["..."],
  "allergies": ["..."],
  "home_medications": [
    {{"name": "...", "dose": "..."}}
  ]
}}"""


def _format_encounters_block(
    conn: sqlite3.Connection,
    cluster_qids: list[int],
) -> tuple[str, str, str]:
    """Build encounter block, diagnoses list, and social history block for prompt.

    Returns: (encounters_block, diagnoses_list, social_history_block)
    """
    encounters_parts = []
    diagnoses_parts = []
    social_parts = []

    for i, qid in enumerate(cluster_qids):
        # Get EHR sections for this question
        sections = conn.execute(
            "SELECT section_type, section_text FROM ehr_sections "
            "WHERE question_id = ? ORDER BY section_order",
            (qid,),
        ).fetchall()

        section_map: dict[str, str] = {}
        for stype, stext in sections:
            section_map[stype] = stext

        parts = [f"Encounter {i + 1} (Question ID: {qid}):"]
        if "chief_complaint" in section_map:
            parts.append(f"  Chief Complaint: {section_map['chief_complaint'][:200]}")
        if "hpi" in section_map:
            parts.append(f"  HPI: {section_map['hpi'][:300]}")
        if "pmh" in section_map:
            parts.append(f"  PMH: {section_map['pmh'][:200]}")
        if "medications" in section_map:
            parts.append(f"  Medications: {section_map['medications'][:200]}")
        encounters_parts.append("\n".join(parts))

        # Social history
        if "social_history" in section_map:
            social_parts.append(
                f"  Q{qid}: {section_map['social_history'][:200]}"
            )

        # Correct diagnosis
        dx_rows = conn.execute(
            "SELECT d.display_name FROM question_diagnoses qd "
            "JOIN diagnoses d ON qd.diagnosis_id = d.diagnosis_id "
            "WHERE qd.question_id = ? AND qd.role = 'correct'",
            (qid,),
        ).fetchall()
        for (dx_name,) in dx_rows:
            diagnoses_parts.append(f"  Q{qid}: {dx_name}")

    encounters_block = "\n\n".join(encounters_parts)
    diagnoses_list = "\n".join(diagnoses_parts) if diagnoses_parts else "  (none extracted)"
    social_history_block = "\n".join(social_parts) if social_parts else "  (none available)"

    return encounters_block, diagnoses_list, social_history_block


def _compute_input_hash_8b(cluster_qids: list[int]) -> str:
    """SHA-256 hash for caching an 8b LLM call."""
    qids_str = ",".join(str(q) for q in sorted(cluster_qids))
    content = f"{STAGE_8B}:{qids_str}"
    return hashlib.sha256(content.encode()).hexdigest()


def _check_cache_8b(conn: sqlite3.Connection, input_hash: str) -> dict | None:
    """Check llm_call_log for a cached Stage 8b result."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log "
        "WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (STAGE_8B, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _generate_profile_single(
    patient_id: int,
    cluster_qids: list[int],
    prompt: str,
) -> dict:
    """Thread-safe LLM worker: generate a patient profile. No DB access."""
    call_t0 = time.time()
    try:
        parsed, in_tok, out_tok, raw_text = _call_with_retry(prompt)
        latency_ms = int((time.time() - call_t0) * 1000)

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

        return {
            "patient_id": patient_id,
            "cluster_qids": cluster_qids,
            "profile": parsed,
            "in_tok": in_tok,
            "out_tok": out_tok,
            "raw_text": raw_text,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        latency_ms = int((time.time() - call_t0) * 1000)
        return {
            "patient_id": patient_id,
            "cluster_qids": cluster_qids,
            "profile": None,
            "in_tok": 0,
            "out_tok": 0,
            "raw_text": None,
            "latency_ms": latency_ms,
            "error": str(e),
        }


REQUIRED_PROFILE_KEYS = {
    "age_at_first_encounter", "sex", "race_ethnicity", "occupation",
    "insurance", "smoking_status", "alcohol_use", "chronic_conditions",
    "surgical_history", "family_history", "allergies", "home_medications",
}


def _validate_profile(
    profile: dict,
    expected_age: int | None,
    expected_sex: str | None,
) -> list[str]:
    """Validate a generated profile. Returns list of warnings."""
    warnings = []

    # Check required keys
    missing = REQUIRED_PROFILE_KEYS - set(profile.keys())
    if missing:
        warnings.append(f"Missing keys: {missing}")

    # Check age
    if expected_age is not None:
        profile_age = profile.get("age_at_first_encounter")
        try:
            profile_age = float(profile_age) if profile_age is not None else None
        except (ValueError, TypeError):
            profile_age = None
        if profile_age is not None and abs(profile_age - expected_age) > 5:
            warnings.append(
                f"Age mismatch: profile={profile_age}, expected≈{expected_age}"
            )

    # Check sex
    if expected_sex is not None:
        profile_sex = profile.get("sex")
        if profile_sex and profile_sex != expected_sex:
            warnings.append(
                f"Sex mismatch: profile={profile_sex}, expected={expected_sex}"
            )

    return warnings


def run_8b(
    conn: sqlite3.Connection,
    pilot: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Stage 8b: Generate patient profiles via LLM.

    Phase 1: Load patients from longitudinal_patients (profile = '{}')
    Phase 2: Check cache, build work queue
    Phase 3: Concurrent LLM generation
    Phase 4: Validate + update profiles
    """
    t0 = time.time()

    # Phase 1: Load patients needing profiles
    rows = conn.execute(
        "SELECT lp.patient_id, lp.age, lp.sex, lp.primary_diagnoses, lp.comorbidities "
        "FROM longitudinal_patients lp "
        "WHERE lp.profile = '{}' "
        "ORDER BY lp.patient_id"
    ).fetchall()

    if not rows:
        # Check if all patients already have profiles
        total = conn.execute(
            "SELECT COUNT(*) FROM longitudinal_patients"
        ).fetchone()[0]
        total_with = conn.execute(
            "SELECT COUNT(*) FROM longitudinal_patients WHERE profile != '{}'"
        ).fetchone()[0]
        log.info(
            f"No patients need profiles ({total_with}/{total} already have profiles)"
        )
        return {"step": "8b", "patients_total": total, "generated": 0, "cached": total_with}

    if pilot:
        rows = rows[:pilot]

    # Get cluster qids for each patient
    patient_data: list[tuple[int, int | None, str | None, list[int]]] = []
    for patient_id, age, sex, _, _ in rows:
        enc_rows = conn.execute(
            "SELECT source_question_ids FROM longitudinal_encounters "
            "WHERE patient_id = ? ORDER BY encounter_order",
            (patient_id,),
        ).fetchall()
        cluster_qids = []
        for (sqids_json,) in enc_rows:
            cluster_qids.extend(json.loads(sqids_json))
        patient_data.append((patient_id, age, sex, cluster_qids))

    log.info(f"Stage 8b: {len(patient_data)} patients need profiles")

    # Phase 2: Check cache, build work queue
    work_queue: list[tuple[int, int | None, str | None, list[int], str, str]] = []
    cached_profiles: dict[int, dict] = {}  # patient_id → profile

    for patient_id, age, sex, cluster_qids in patient_data:
        input_hash = _compute_input_hash_8b(cluster_qids)
        cached = _check_cache_8b(conn, input_hash)
        if cached is not None:
            cached_profiles[patient_id] = cached
            continue

        # Build prompt
        encounters_block, diagnoses_list, social_history_block = (
            _format_encounters_block(conn, cluster_qids)
        )
        prompt = PROFILE_PROMPT.format(
            age=age or "unknown",
            sex=sex or "unknown",
            encounters_block=encounters_block,
            diagnoses_list=diagnoses_list,
            social_history_block=social_history_block,
        )
        work_queue.append((patient_id, age, sex, cluster_qids, input_hash, prompt))

    log.info(
        f"Cache check: {len(cached_profiles)} cached, {len(work_queue)} to generate "
        f"({workers} workers)"
    )

    # Phase 3: Concurrent LLM generation
    generated = 0
    errors = 0

    if work_queue:
        completed_count = 0
        pending = len(work_queue)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for patient_id, age, sex, cluster_qids, input_hash, prompt in work_queue:
                future = pool.submit(
                    _generate_profile_single, patient_id, cluster_qids, prompt
                )
                future_map[future] = (patient_id, age, sex, cluster_qids, input_hash)

            for future in as_completed(future_map):
                patient_id, age, sex, cluster_qids, input_hash = future_map[future]
                result = future.result()
                completed_count += 1

                if result["error"] is None:
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'patient_profile', ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            STAGE_8B, MODEL, result["in_tok"], result["out_tok"],
                            result["latency_ms"], input_hash,
                            json.dumps(result["profile"]),
                            result["raw_text"], None,
                        ),
                    )
                    cached_profiles[patient_id] = result["profile"]
                    generated += 1
                else:
                    log.error(f"patient_id={patient_id} failed: {result['error']}")
                    conn.execute(
                        """INSERT INTO llm_call_log
                           (stage, model, prompt_template, input_tokens,
                            output_tokens, latency_ms, input_hash,
                            output_json, raw_response, error)
                           VALUES (?, ?, 'patient_profile', 0, 0, ?, ?, NULL, NULL, ?)""",
                        (
                            STAGE_8B, MODEL, result["latency_ms"],
                            input_hash, result["error"],
                        ),
                    )
                    errors += 1

                if completed_count % 10 == 0:
                    conn.commit()

                if completed_count % 50 == 0 or completed_count == pending:
                    elapsed = time.time() - t0
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = pending - completed_count
                    eta = remaining / rate if rate > 0 else 0
                    log.info(
                        f"Progress {completed_count}/{pending} | "
                        f"generated={generated} errors={errors} | "
                        f"{elapsed:.0f}s elapsed | ETA={eta:.0f}s"
                    )

        conn.commit()

    # Phase 4: Validate + update profiles
    log.info("Phase 4: Validating and updating patient profiles...")
    updated = 0
    validation_warnings = 0

    for patient_id, age, sex, cluster_qids in patient_data:
        profile = cached_profiles.get(patient_id)
        if profile is None:
            continue

        warnings = _validate_profile(profile, age, sex)
        if warnings:
            for w in warnings:
                log.warning(f"patient_id={patient_id}: {w}")
            validation_warnings += len(warnings)

        # Generate PCP name from profile or default
        pcp_name = profile.get("pcp_name", f"Dr. Smith-{patient_id}")

        conn.execute(
            """UPDATE longitudinal_patients
               SET profile = ?, race_ethnicity = ?, insurance = ?, pcp_name = ?
               WHERE patient_id = ?""",
            (
                json.dumps(profile),
                profile.get("race_ethnicity"),
                profile.get("insurance"),
                pcp_name,
                patient_id,
            ),
        )
        updated += 1

    conn.commit()

    duration = time.time() - t0
    summary = {
        "step": "8b",
        "patients_total": len(patient_data),
        "cached": len(cached_profiles) - generated,
        "generated": generated,
        "errors": errors,
        "updated": updated,
        "validation_warnings": validation_warnings,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Stage 8b complete: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(conn: sqlite3.Connection):
    """Run verification queries for Stage 8 output."""
    log.info("--- Stage 8 Verification ---")

    # Patient counts
    total_patients = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_patients"
    ).fetchone()[0]
    log.info(f"longitudinal_patients: {total_patients} rows")

    if total_patients == 0:
        log.warning("No patients — run Stage 8a first")
        return

    total_encounters = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_encounters"
    ).fetchone()[0]
    log.info(f"longitudinal_encounters: {total_encounters} rows")

    # Encounters per patient distribution
    log.info("Encounters per patient:")
    rows = conn.execute(
        "SELECT num_encounters, COUNT(*) FROM longitudinal_patients "
        "GROUP BY num_encounters ORDER BY num_encounters"
    ).fetchall()
    for n_enc, n_pat in rows:
        log.info(f"  {n_enc} encounters: {n_pat} patients")

    # Sex distribution
    log.info("Sex distribution:")
    rows = conn.execute(
        "SELECT sex, COUNT(*) FROM longitudinal_patients "
        "GROUP BY sex ORDER BY COUNT(*) DESC"
    ).fetchall()
    for sex, cnt in rows:
        log.info(f"  {sex}: {cnt}")

    # Question coverage
    coverage = conn.execute(
        "SELECT COUNT(DISTINCT je.value) "
        "FROM longitudinal_encounters, json_each(source_question_ids) je"
    ).fetchone()[0]
    total_questions = conn.execute(
        "SELECT COUNT(*) FROM board_questions"
    ).fetchone()[0]
    singletons = total_questions - coverage
    log.info(
        f"Question coverage: {coverage}/{total_questions} "
        f"({coverage / total_questions * 100:.1f}%), "
        f"{singletons} singletons"
    )

    # Profile completeness
    with_profile = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_patients WHERE profile != '{}'"
    ).fetchone()[0]
    log.info(
        f"Profile completeness: {with_profile}/{total_patients} "
        f"({with_profile / total_patients * 100:.1f}%)"
    )

    # Average encounters
    avg_enc = conn.execute(
        "SELECT AVG(num_encounters) FROM longitudinal_patients"
    ).fetchone()[0]
    log.info(f"Average encounters per patient: {avg_enc:.1f}")

    # Success criteria check
    gte2 = conn.execute(
        "SELECT COUNT(*) FROM longitudinal_patients WHERE num_encounters >= 2"
    ).fetchone()[0]
    log.info(
        f"Success criteria: {gte2} patients with >=2 encounters "
        f"(target: >=1,500) — {'PASS' if gte2 >= 1500 else 'BELOW TARGET'}"
    )

    # LLM call log stats
    ok_8b = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NULL",
        (STAGE_8B,),
    ).fetchone()[0]
    err_8b = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE stage = ? AND error IS NOT NULL",
        (STAGE_8B,),
    ).fetchone()[0]
    if ok_8b or err_8b:
        log.info(f"LLM calls (8b): {ok_8b} OK / {err_8b} errors")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


def export_csv(conn: sqlite3.Connection, output_dir: Path):
    """Export longitudinal_patients and longitudinal_encounters to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Patients
    path_p = output_dir / "s08_longitudinal_patients.csv"
    rows = conn.execute(
        """SELECT patient_id, age, sex, race_ethnicity, insurance, pcp_name,
                  num_encounters, primary_diagnoses, comorbidities,
                  generation_seed, profile, created_at
           FROM longitudinal_patients ORDER BY patient_id"""
    ).fetchall()
    with open(path_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id", "age", "sex", "race_ethnicity", "insurance", "pcp_name",
            "num_encounters", "primary_diagnoses", "comorbidities",
            "generation_seed", "profile", "created_at",
        ])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} patients to {path_p}")

    # Encounters
    path_e = output_dir / "s08_longitudinal_encounters.csv"
    rows = conn.execute(
        """SELECT encounter_id, patient_id, encounter_date, encounter_type,
                  chief_complaint, attending_name, department,
                  source_question_ids, encounter_order,
                  generation_method, created_at
           FROM longitudinal_encounters ORDER BY patient_id, encounter_order"""
    ).fetchall()
    with open(path_e, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "encounter_id", "patient_id", "encounter_date", "encounter_type",
            "chief_complaint", "attending_name", "department",
            "source_question_ids", "encounter_order",
            "generation_method", "created_at",
        ])
        writer.writerows(rows)
    log.info(f"Exported {len(rows)} encounters to {path_e}")

    # Distractor tables (if they exist)
    try:
        rows_dp = conn.execute(
            """SELECT patient_id, age, sex, race_ethnicity, insurance, pcp_name,
                      num_encounters, primary_diagnoses, comorbidities,
                      generation_seed, profile, created_at
               FROM longitudinal_patients_distractor ORDER BY patient_id"""
        ).fetchall()
        if rows_dp:
            path_dp = output_dir / "s08_longitudinal_patients_distractor.csv"
            with open(path_dp, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "patient_id", "age", "sex", "race_ethnicity", "insurance", "pcp_name",
                    "num_encounters", "primary_diagnoses", "comorbidities",
                    "generation_seed", "profile", "created_at",
                ])
                writer.writerows(rows_dp)
            log.info(f"Exported {len(rows_dp)} distractor patients to {path_dp}")

            path_de = output_dir / "s08_longitudinal_encounters_distractor.csv"
            rows_de = conn.execute(
                """SELECT encounter_id, patient_id, encounter_date, encounter_type,
                          chief_complaint, attending_name, department,
                          source_question_ids, encounter_order,
                          generation_method, created_at
                   FROM longitudinal_encounters_distractor ORDER BY patient_id, encounter_order"""
            ).fetchall()
            with open(path_de, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "encounter_id", "patient_id", "encounter_date", "encounter_type",
                    "chief_complaint", "attending_name", "department",
                    "source_question_ids", "encounter_order",
                    "generation_method", "created_at",
                ])
                writer.writerows(rows_de)
            log.info(f"Exported {len(rows_de)} distractor encounters to {path_de}")
    except sqlite3.OperationalError:
        pass  # Distractor tables don't exist yet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 8: Patient Profiling & Clustering"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step", choices=["8a", "8b", "8a-distractor"],
                       help="Which step to run")
    group.add_argument("--all", action="store_true",
                       help="Run 8a then 8b")
    group.add_argument("--verify-only", action="store_true",
                       help="Run verification queries only")
    group.add_argument("--export-csv", action="store_true",
                       help="Export tables to CSV")

    parser.add_argument("--pilot", type=int, default=0,
                        help="Process only first N questions/patients")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of concurrent LLM workers (for 8b)")
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()
    conn = get_connection(args.db)

    if args.verify_only:
        verify(conn)
        conn.close()
        return

    if args.export_csv:
        export_csv(conn, DATA_DIR)
        conn.close()
        return

    pilot = args.pilot if args.pilot > 0 else None

    if args.step == "8a" or args.all:
        summary = run_8a(conn, pilot=pilot)
        print(json.dumps(summary, indent=2))

    if args.step == "8a-distractor":
        summary = run_8a_distractor(conn, pilot=pilot)
        print(json.dumps(summary, indent=2))

    if args.step == "8b" or args.all:
        summary = run_8b(conn, pilot=pilot, workers=args.workers)
        print(json.dumps(summary, indent=2))

    if args.step or args.all:
        verify(conn)

    conn.close()


if __name__ == "__main__":
    main()
