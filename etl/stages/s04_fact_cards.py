"""Stage 4: Extract Fact Cards — populate fact_cards from fact raw_cards.

  Method: Rule-based + LLM for classification. Fact decks vary in field layout;
  each supported layout has a small extractor, keyed by note model / deck below.

  Rule-based extraction (examples of supported fact-card layouts):
    - Resolve cloze deletions: {{c1::answer}} → resolved plain text + original markup
    - QID-tagged cloze: QID → tags, Objective → fact_text, Subject/System/Topic → metadata
    - Hierarchy-tagged cloze: Text/Extra → cloze_fact; deck hierarchy Root::Specialty → specialty
    - Resource-tagged cloze: Text/Extra → cloze_fact; resource fields → resource_refs JSON;
      tag hierarchy → subject/organ_system
    - Specialty-per-deck cloze: Text/Extra → cloze_fact; deck hierarchy → specialty;
      filename fallback map
    - Multi-model deck: field priority Text → Front/Back → Question/Answer → Original

  LLM enrichment (batched, deferred to Stage 4b):
    - Classify subject/organ_system for ~4,737 untagged cards
"""

import json
import re
import sqlite3
import time

from etl.deck_registry import REGISTRY
from etl.parsers.cloze import resolve_all_cloze, has_cloze
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s04_fact_cards")


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

def _tag_cfg(filename: str) -> dict:
    """Source-specific tag conventions for a card, read from its deck profile.

    Returns the profile's `tag_config` (or {}). This keeps all source/vendor
    tokens (field names, tag namespaces) in configuration, not in code.
    """
    profile = REGISTRY.resolve(filename=filename)
    return dict(profile.tag_config) if profile else {}


# Per-deck specialty fallback is declared on each deck profile
# (specialty_fallback: in etl/deck_profiles/*.yaml) and read via the registry.
# See _deck_specialty_fallback below.

# Chapter code → (subject, organ_system) mapping
_CHAPTER_ORGAN_MAP: dict[str, tuple[str, str]] = {
    "01_Biochemistry": ("Biochemistry", "Biochemistry"),
    "01_Biochem": ("Biochemistry", "Biochemistry"),
    "02_Immunology": ("Immunology", "Immune System"),
    "03_Micro": ("Microbiology", "Infectious Disease"),
    "04_Pathology": ("Pathology", "Pathology"),
    "05_Pharmacology": ("Pharmacology", "Pharmacology"),
    "06_Repro": ("Reproductive", "Reproductive System"),
    "07_Cardio": ("Cardiology", "Cardiovascular"),
    "08_Endo": ("Endocrinology", "Endocrine System"),
    "09_GI": ("Gastroenterology", "Gastrointestinal"),
    "10_Heme": ("Hematology", "Hematologic System"),
    "11_MSK": ("Musculoskeletal", "Musculoskeletal System"),
    "12_Neuro": ("Neurology", "Nervous System"),
    "13_Psych": ("Psychiatry", "Psychiatry"),
    "14_Renal": ("Nephrology", "Renal System"),
    "15_Respiratory": ("Pulmonology", "Respiratory System"),
    "16_Derm": ("Dermatology", "Integumentary System"),
}


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def _normalize_tags(tags_str: str) -> list[str]:
    """Split Anki tags into a JSON-ready list."""
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split() if t.strip()]


# Deck-hierarchy tokens that are structural boilerplate, not a specialty.
_HIERARCHY_BOILERPLATE = re.compile(
    r"^(step\s*\d+|update|default|medicine|deck|root|cards?|new)$", re.IGNORECASE
)


def _deck_specialty_fallback(filename: str) -> str | None:
    """Fixed specialty declared on the deck's profile (specialty_fallback)."""
    profile = REGISTRY.resolve(filename=filename)
    return (profile.specialty_fallback or None) if profile else None


def _specialty_from_deck(deck_name: str, filename: str = "") -> str | None:
    """Derive specialty generically from the '::'-delimited deck-name hierarchy.

    Walks the path and returns the first token that isn't structural boilerplate
    (namespace labels, 'Step N', etc.). Examples:
        Root::Neurology::surgery                  → "Neurology"
        Step 3::Cardiology                        → "Cardiology"
        Medicine::Step 3::InternalMedicine::GI    → "InternalMedicine"
    Decks whose hierarchy doesn't encode specialty should instead set
    `specialty_fallback` in their deck profile. Returns None if nothing matches.
    """
    if not deck_name:
        return None
    parts = [p.strip() for p in deck_name.split("::") if p.strip()]
    # Skip the leading namespace token, then return the first non-boilerplate part.
    for p in parts[1:] if len(parts) > 1 else parts:
        if _HIERARCHY_BOILERPLATE.match(p):
            continue
        return p
    return None


def _parse_resource_refs(field_data: dict, field_data_text: dict,
                         cfg: dict | None = None) -> dict | None:
    """Extract resource-reference fields into a dict.

    The field names to look for come from the deck profile's
    tag_config["resource_fields"] (list). Checks field_data (HTML) for non-empty
    since resource fields often contain <img> tags that become empty after HTML
    stripping. Falls back to text version.
    """
    refs = {}
    for field_name in (cfg or {}).get("resource_fields", []):
        # Check HTML version first (resource fields often have only images)
        html_value = field_data.get(field_name, "").strip()
        text_value = field_data_text.get(field_name, "").strip()
        if text_value:
            key = field_name.lower().replace(" ", "_").replace("&", "and")
            refs[key] = text_value
        elif html_value:
            key = field_name.lower().replace(" ", "_").replace("&", "and")
            refs[key] = html_value
    return refs if refs else None


# ---------------------------------------------------------------------------
# Tag parsing functions
# ---------------------------------------------------------------------------

def _parse_tags_qid(tags_str: str, cfg: dict | None = None) -> dict:
    """Parse QID-namespaced tags for subject/organ_system/topic.

    Namespace token comes from tag_config["qid_namespace"]. The tag shape is:
        <namespace>::<positional>::<Subject>::<System>::<Topic>
    """
    result = {"subject": None, "organ_system": None, "topic": None}
    namespace = (cfg or {}).get("qid_namespace")
    if not tags_str or not namespace:
        return result
    for tag in tags_str.split():
        parts = tag.lstrip("#").split("::")
        if len(parts) < 2 or parts[0] != namespace:
            continue
        # parts: <namespace>, <positional>, Subject, System, Topic
        if len(parts) >= 3:
            result["subject"] = parts[2].replace("_", " ")
        if len(parts) >= 4:
            result["organ_system"] = parts[3].replace("_", " ")
        if len(parts) >= 5:
            result["topic"] = parts[4].replace("_", " ")
    return result


def _parse_tags_review_hier(tags_str: str, cfg: dict | None = None) -> dict:
    """Parse hierarchy-namespaced tags for specialty/subject.

    Tokens come from tag_config:
      hier_root + hier_level2  -> <root>::<level2>::<Specialty>::<SubSpecialty>
      subject_namespace        -> <subject_namespace>::<Subject>
    plus a keyword fallback for unstructured tags.
    """
    cfg = cfg or {}
    hier_root = cfg.get("hier_root")
    hier_level2 = cfg.get("hier_level2")
    subject_namespace = cfg.get("subject_namespace")
    result = {"specialty": None, "subject": None}
    if not tags_str:
        return result
    for tag in tags_str.split():
        parts = tag.split("::")
        if (hier_root and len(parts) >= 3
                and parts[0] == hier_root and parts[1] == hier_level2):
            result["specialty"] = parts[2]
            if len(parts) >= 4:
                result["subject"] = parts[3].replace("/", " / ")
            break
        if subject_namespace and len(parts) >= 2 and parts[0] == subject_namespace:
            result["subject"] = parts[1].replace("_", " ")
            break

    # Fallback: keyword matching for unstructured tags
    if not result["subject"] and tags_str:
        tags_lower = tags_str.lower()
        for keyword, subj in _TAG_KEYWORD_SUBJECT.items():
            if keyword in tags_lower:
                result["subject"] = subj
                break

    return result


_SHELF_SUBJECT_MAP: dict[str, str] = {
    "IM": "Internal Medicine",
    "FM": "Family Medicine",
    "ObGyn": "OB/GYN",
    "Peds": "Pediatrics",
    "Surgery": "Surgery",
    "EM": "Emergency Medicine",
    "Psych": "Psychiatry",
    "Neuro": "Neurology",
}

# Secondary chapter code → subject mapping
_CHAPTER_SUBJECT_MAP: dict[str, str] = {
    "01_Biochemistry": "Biochemistry",
    "01_Biochem": "Biochemistry",
    "02_Immunology": "Immunology",
    "03_Biochem": "Biochemistry",
    "03_Microbiology": "Microbiology",
    "04_Pathology": "Pathology",
    "05_Pharmacology": "Pharmacology",
    "06_Repro": "Reproductive",
    "07_Cardio": "Cardiology",
    "08_Endo": "Endocrinology",
    "09_GI": "Gastroenterology",
    "09_Dermatology": "Dermatology",
    "10_Heme": "Hematology",
    "10_Neurology": "Neurology",
    "10_Pulmonary": "Pulmonology",
    "11_MSK": "Musculoskeletal",
    "11_Pediatrics": "Pediatrics",
    "11_Renal": "Nephrology",
    "12_Neuro": "Neurology",
    "13_Infectious_Disease": "Infectious Disease",
    "13_Obstetrics": "Obstetrics",
    "14_Gynecology": "Gynecology",
    "14_Renal": "Nephrology",
    "15_General_Surgery": "Surgery",
    "15_Respiratory": "Pulmonology",
    "16_Derm": "Dermatology",
    "16_Surgical_Subspecialties": "Surgery",
}

# Lowercase keyword → subject for fallback pattern matching in tag parts
_TAG_KEYWORD_SUBJECT: dict[str, str] = {
    "cardio": "Cardiology",
    "cardiac": "Cardiology",
    "neuro": "Neurology",
    "derm": "Dermatology",
    "gi": "Gastroenterology",
    "endo": "Endocrinology",
    "renal": "Nephrology",
    "pulm": "Pulmonology",
    "respiratory": "Pulmonology",
    "heme": "Hematology",
    "onc": "Oncology",
    "psych": "Psychiatry",
    "repro": "Reproductive",
    "msk": "Musculoskeletal",
    "ophtho": "Ophthalmology",
    "ob": "Obstetrics",
    "gyn": "Gynecology",
    "obgyn": "OB/GYN",
    "immuno": "Immunology",
    "micro": "Microbiology",
    "pharm": "Pharmacology",
    "biostats": "Biostatistics",
    "ethics": "Ethics",
    "ortho": "Orthopedics",
    "rheum": "Rheumatology",
    "infect": "Infectious Disease",
    "ent": "ENT",
    "surgery": "Surgery",
    "surg": "Surgery",
    "fam": "Family Medicine",
    "uro": "Urology",
    "urology": "Urology",
    "peds": "Pediatrics",
    "im": "Internal Medicine",
    "em": "Emergency Medicine",
}


def _parse_tags_resource(tags_str: str, cfg: dict | None = None) -> dict:
    """Parse resource/chapter hierarchical tags for subject and organ_system.

    Strategies, in priority order:
    1. Primary chapter map (chapter code -> subject + organ_system)
    2. Secondary chapter map (chapter code -> subject)
    3. Shelf markers (tag_config["shelf_markers"]) -> next part is a subject code
    4. Keyword matching in tag parts (cardio, neuro, ...)
    Topic-candidate prefixes to ignore come from tag_config["topic_exclude_prefixes"].
    """
    cfg = cfg or {}
    shelf_markers = tuple(cfg.get("shelf_markers", ()))
    topic_excludes = tuple(cfg.get("topic_exclude_prefixes", ()))
    result = {"subject": None, "organ_system": None, "topic": None}
    if not tags_str:
        return result

    for tag in tags_str.split():
        parts = tag.lstrip("#").split("::")

        # Strategy 1: primary chapter mapping (subject + organ_system)
        for part in parts:
            for chapter_key, (subj, organ) in _CHAPTER_ORGAN_MAP.items():
                if part == chapter_key or part.startswith(chapter_key):
                    result["subject"] = subj
                    result["organ_system"] = organ
                    # Topic from deeper parts
                    if len(parts) >= 4:
                        candidate = parts[-1].replace("_", " ").strip()
                        if (len(candidate) > 2
                                and not any(candidate.startswith(pre) for pre in topic_excludes)):
                            result["topic"] = candidate
                    return result

        # Strategy 2: secondary chapter mapping (subject only)
        for part in parts:
            for chapter_key, subj in _CHAPTER_SUBJECT_MAP.items():
                if part == chapter_key or part.startswith(chapter_key):
                    result["subject"] = subj
                    return result

        # Strategy 3: shelf marker -> subject code
        for i, part in enumerate(parts):
            if part in shelf_markers:
                if i + 1 < len(parts) and parts[i + 1] in _SHELF_SUBJECT_MAP:
                    result["subject"] = _SHELF_SUBJECT_MAP[parts[i + 1]]
                    return result

    # Strategy 4: Keyword fallback across all tags
    tags_lower = tags_str.lower()
    for keyword, subj in _TAG_KEYWORD_SUBJECT.items():
        # Match as an isolated segment (between :: or spaces or start/end)
        if f"::{keyword}::" in tags_lower or f"::{keyword} " in tags_lower:
            result["subject"] = subj
            return result

    return result


# Podcast-deck category → subject mapping
_PODCAST_CATEGORY_SUBJECT: dict[str, str] = {
    "IM": "Internal Medicine",
    "Cardio": "Cardiology",
    "Neurology": "Neurology",
    "NeuroShelf": "Neurology",
    "OBGYN": "OB/GYN",
    "Peds": "Pediatrics",
    "Surgery": "Surgery",
    "EmergencyMed": "Emergency Medicine",
    "FamilyMed": "Family Medicine",
    "Endocrine": "Endocrinology",
    "Biostats&Ethics": "Biostatistics",
    "Biostats": "Biostatistics",
    "2020Changes": "General Principles",
}

# Question-bank subcategory → subject mapping (for <root>::Optional::<source>::<subcat>)
_RESOURCE_SUBCAT_SUBJECT: dict[str, str] = {
    "cards": "Cardiology",
    "cardiovascular": "Cardiology",
    "gi": "Gastroenterology",
    "gastrointestinal": "Gastroenterology",
    "pulm": "Pulmonology",
    "respiratory": "Pulmonology",
    "pregnancy": "Obstetrics",
    "heme": "Hematology",
    "endocrine": "Endocrinology",
    "nervous": "Neurology",
    "neuro": "Neurology",
    "renal": "Nephrology",
    "rheumatology": "Rheumatology",
    "dermatology": "Dermatology",
    "psychiatric": "Psychiatry",
    "infectious": "Infectious Disease",
    "allergy": "Immunology",
    "biostatistics": "Biostatistics",
    "social": "Ethics",
    "ent": "ENT",
    "ear": "ENT",
    "male": "Urology",
    "general": "Pathology",
}


def _parse_dip_tags(tags_str: str, cfg: dict | None = None) -> dict:
    """Parse podcast-deck tags for subject.

    Tokens come from tag_config:
      podcast_root      -> leading tag token
      podcast_branch    -> <root>::<podcast_branch>::<Category>::<Episode>
      optional_branch   -> <root>::<optional_branch>::<Source>::<SubCategory>
      fixed_source_subjects -> {source_token: subject} (exact-match sources)
      subcat_source     -> source whose <SubCategory> maps via the subcategory map
      keyword_source_prefix -> source prefix triggering a keyword match on the tag
    """
    cfg = cfg or {}
    podcast_root = cfg.get("podcast_root")
    podcast_branch = cfg.get("podcast_branch")
    optional_branch = cfg.get("optional_branch")
    fixed_sources = cfg.get("fixed_source_subjects", {})
    subcat_source = cfg.get("subcat_source")
    keyword_source_prefix = cfg.get("keyword_source_prefix")
    result = {"subject": None}
    if not tags_str or not podcast_root:
        return result

    for tag in tags_str.split():
        parts = tag.split("::")
        if len(parts) < 3 or parts[0] != podcast_root:
            continue

        if podcast_branch and parts[1] == podcast_branch and len(parts) >= 3:
            subject = _PODCAST_CATEGORY_SUBJECT.get(parts[2])
            if subject:
                result["subject"] = subject
                return result
            # Otherwise: keyword match on episode name
            if len(parts) >= 4:
                episode_lower = parts[3].lower()
                for keyword, subj in _TAG_KEYWORD_SUBJECT.items():
                    if keyword in episode_lower:
                        result["subject"] = subj
                        return result

        if optional_branch and parts[1] == optional_branch and len(parts) >= 3:
            source = parts[2]
            if source in fixed_sources:
                result["subject"] = fixed_sources[source]
                return result
            if subcat_source and source == subcat_source and len(parts) >= 4:
                sub_lower = parts[3].lower()
                for keyword, subj in _RESOURCE_SUBCAT_SUBJECT.items():
                    if keyword in sub_lower:
                        result["subject"] = subj
                        return result
            if keyword_source_prefix and source.lower().startswith(keyword_source_prefix):
                tag_lower = tag.lower()
                for keyword, subj in _TAG_KEYWORD_SUBJECT.items():
                    if keyword in tag_lower:
                        result["subject"] = subj
                        return result

    # Keyword fallback across all tag text
    tags_lower = tags_str.lower()
    for keyword, subj in _TAG_KEYWORD_SUBJECT.items():
        if f"::{keyword}::" in tags_lower or f"::{keyword} " in tags_lower:
            result["subject"] = subj
            return result

    return result


# ---------------------------------------------------------------------------
# Source-specific extractors
# ---------------------------------------------------------------------------

def _extract_cloze_qid(field_data, field_data_text, tags_str, deck_name,
                 card_format, filename):
    """Extract fact card from a QID-namespaced cloze card.

    Fields: QID, Objective, Extra, Subject, System, Topic, T#, Q#
    """
    objective = field_data_text.get("Objective", "").strip()
    if not objective or len(objective) < 5:
        return None

    fact_text_cloze = objective if has_cloze(objective) else None
    fact_text = resolve_all_cloze(objective) if fact_text_cloze else objective

    subject = field_data_text.get("Subject", "").strip() or None
    organ_system = field_data_text.get("System", "").strip() or None
    topic = field_data_text.get("Topic", "").strip() or None
    extra = field_data_text.get("Extra", "").strip() or None

    # QID → store in tags_normalized (no source_qid column in schema)
    qid = field_data_text.get("QID", "").strip()
    tags = _normalize_tags(tags_str)
    if qid:
        tags.insert(0, f"QID:{qid}")

    # Tag-based fallback
    if not subject or not organ_system or not topic:
        tag_info = _parse_tags_qid(tags_str, _tag_cfg(filename))
        if not subject:
            subject = tag_info.get("subject")
        if not organ_system:
            organ_system = tag_info.get("organ_system")
        if not topic:
            topic = tag_info.get("topic")

    return {
        "fact_format": "cloze_fact",
        "fact_text": fact_text,
        "fact_text_cloze": fact_text_cloze,
        "fact_context": extra,
        "fact_summary": None,
        "subject": subject,
        "organ_system": organ_system,
        "topic": topic,
        "specialty": None,
        "resource_refs": None,
        "tags_normalized": json.dumps(tags, ensure_ascii=False),
        "extraction_method": "rule_based",
    }


def _extract_cloze_review_hier(field_data, field_data_text, tags_str, deck_name,
                  card_format, filename):
    """Extract fact card from a hierarchy-namespaced review card.

    16+ model types. Field priority: Text → Front/Back → Question/Answer → Original
    """
    # Try content fields in priority order
    text_field = field_data_text.get("Text", "").strip()
    front_field = field_data_text.get("Front", "").strip()
    back_field = field_data_text.get("Back", "").strip()
    question_field = field_data_text.get("Question", "").strip()
    answer_field = field_data_text.get("Answer", "").strip()
    original_field = field_data_text.get("Original", "").strip()
    extra = field_data_text.get("Extra", "").strip() or None
    summary = field_data_text.get("Summary", "").strip() or None

    fact_text = None
    fact_text_cloze = None
    fact_format = card_format

    if text_field:
        # Standard cloze models (6,757 cards)
        fact_text_cloze = text_field if has_cloze(text_field) else None
        fact_text = resolve_all_cloze(text_field) if fact_text_cloze else text_field
    elif front_field or back_field:
        # Basic models (2,019 cards)
        if front_field and back_field:
            fact_text = f"Q: {front_field}\nA: {back_field}"
        else:
            fact_text = front_field or back_field
        fact_format = "basic_fact"
    elif question_field:
        # Q/A note model — Q/A format, no cloze
        if answer_field:
            fact_text = f"Q: {question_field}\nA: {answer_field}"
        else:
            fact_text = question_field
        fact_format = "basic_fact"
    elif original_field:
        # Overlapping cloze model (168 cards) — [[oc...]] syntax
        fact_text_cloze = original_field if has_cloze(original_field) else None
        fact_text = resolve_all_cloze(original_field) if fact_text_cloze else original_field
    else:
        return None

    if not fact_text or len(fact_text.strip()) < 3:
        return None

    tag_info = _parse_tags_review_hier(tags_str, _tag_cfg(filename))
    specialty = tag_info.get("specialty") or _specialty_from_deck(deck_name, filename)
    subject = tag_info.get("subject")
    # Use specialty as subject fallback (e.g., "Pediatrics", "Surgery")
    if not subject and specialty:
        subject = specialty

    return {
        "fact_format": fact_format,
        "fact_text": fact_text,
        "fact_text_cloze": fact_text_cloze,
        "fact_context": extra,
        "fact_summary": summary,
        "subject": subject,
        "organ_system": None,
        "topic": None,
        "specialty": specialty,
        "resource_refs": None,
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
    }


def _extract_cloze_review_spec(field_data, field_data_text, tags_str, deck_name,
                    card_format, filename):
    """Extract fact card from a specialty-hierarchy review card.

    Model J with Text/Extra fields. 97.7% tagless — deck hierarchy is main source.
    """
    text_field = field_data_text.get("Text", "").strip()
    if not text_field or len(text_field) < 5:
        return None

    fact_text_cloze = text_field if has_cloze(text_field) else None
    fact_text = resolve_all_cloze(text_field) if fact_text_cloze else text_field
    extra = field_data_text.get("Extra", "").strip() or None

    specialty = _specialty_from_deck(deck_name, filename)
    # Use specialty as subject fallback for specialty-hierarchy decks
    subject = specialty

    return {
        "fact_format": card_format,
        "fact_text": fact_text,
        "fact_text_cloze": fact_text_cloze,
        "fact_context": extra,
        "fact_summary": None,
        "subject": subject,
        "organ_system": None,
        "topic": None,
        "specialty": specialty,
        "resource_refs": None,
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
    }


def _extract_cloze_resource(field_data, field_data_text, tags_str, deck_name,
                    card_format, filename):
    """Extract fact card from a resource-tagged cloze card."""
    if card_format == "image_occlusion":
        return _extract_image_occlusion(
            field_data, field_data_text, tags_str, deck_name, card_format, filename
        )

    text_field = field_data_text.get("Text", "").strip()
    if not text_field or len(text_field) < 5:
        return None

    fact_text_cloze = text_field if has_cloze(text_field) else None
    fact_text = resolve_all_cloze(text_field) if fact_text_cloze else text_field
    extra = field_data_text.get("Extra", "").strip() or None

    resource_refs = _parse_resource_refs(field_data, field_data_text, _tag_cfg(filename))
    tag_info = _parse_tags_resource(tags_str, _tag_cfg(filename))

    return {
        "fact_format": "cloze_fact",
        "fact_text": fact_text,
        "fact_text_cloze": fact_text_cloze,
        "fact_context": extra,
        "fact_summary": None,
        "subject": tag_info.get("subject"),
        "organ_system": tag_info.get("organ_system"),
        "topic": tag_info.get("topic"),
        "specialty": None,
        "resource_refs": json.dumps(resource_refs, ensure_ascii=False) if resource_refs else None,
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
    }


def _extract_cloze_specialty_deck(field_data, field_data_text, tags_str, deck_name,
                   card_format, filename):
    """Extract fact card from a single-specialty deck card.

    Text/Extra + resource fields (mostly empty). Specialty from deck/filename.
    """
    text_field = field_data_text.get("Text", "").strip()
    if not text_field or len(text_field) < 5:
        return None

    fact_text_cloze = text_field if has_cloze(text_field) else None
    fact_text = resolve_all_cloze(text_field) if fact_text_cloze else text_field
    extra = field_data_text.get("Extra", "").strip() or None

    resource_refs = _parse_resource_refs(field_data, field_data_text, _tag_cfg(filename))

    specialty = _specialty_from_deck(deck_name, filename)
    if not specialty:
        specialty = _deck_specialty_fallback(filename)

    # Try resource-tagged parsing (some specialty decks share resource tags)
    tag_info = _parse_tags_resource(tags_str, _tag_cfg(filename))
    subject = tag_info.get("subject")
    # Use specialty as subject fallback for single-specialty decks
    if not subject and specialty:
        subject = specialty

    return {
        "fact_format": card_format,
        "fact_text": fact_text,
        "fact_text_cloze": fact_text_cloze,
        "fact_context": extra,
        "fact_summary": None,
        "subject": subject,
        "organ_system": tag_info.get("organ_system"),
        "topic": tag_info.get("topic"),
        "specialty": specialty,
        "resource_refs": json.dumps(resource_refs, ensure_ascii=False) if resource_refs else None,
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
    }


def _extract_image_occlusion(field_data, field_data_text, tags_str, deck_name,
                             card_format, filename):
    """Extract minimal fact card from Image Occlusion Enhanced card (3 cards)."""
    header = field_data_text.get("Header", "").strip()
    footer = field_data_text.get("Footer", "").strip()
    remarks = field_data_text.get("Remarks", "").strip()
    extra = field_data_text.get("Extra", "").strip()

    parts = [p for p in [header, footer, remarks] if p]
    fact_text = " | ".join(parts) if parts else "[Image Occlusion Card]"

    tag_info = _parse_tags_resource(tags_str, _tag_cfg(filename))

    return {
        "fact_format": "image_occlusion",
        "fact_text": fact_text,
        "fact_text_cloze": None,
        "fact_context": extra or None,
        "fact_summary": None,
        "subject": tag_info.get("subject"),
        "organ_system": tag_info.get("organ_system"),
        "topic": tag_info.get("topic"),
        "specialty": None,
        "resource_refs": None,
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
    }


def _extract_cloze_podcast(field_data, field_data_text, tags_str, deck_name,
                 card_format, filename):
    """Extract fact card from a podcast/lecture-notes deck.

    Handles multi-model podcast/lecture decks (single- or multi-specialty).
    Field priority: Text → FrontText/Front/Back → Original (overlapping cloze).
    """
    if card_format == "image_occlusion":
        return _extract_image_occlusion(
            field_data, field_data_text, tags_str, deck_name, card_format, filename
        )

    # Try content fields in priority order
    text_field = field_data_text.get("Text", "").strip()
    front_field = (
        field_data_text.get("FrontText", "").strip()
        or field_data_text.get("Front", "").strip()
    )
    back_field = (
        field_data_text.get("BackText", "").strip()
        or field_data_text.get("Back", "").strip()
    )
    original_field = field_data_text.get("Original", "").strip()
    extra = field_data_text.get("Extra", "").strip() or None

    fact_text = None
    fact_text_cloze = None
    fact_format = card_format

    if text_field:
        fact_text_cloze = text_field if has_cloze(text_field) else None
        fact_text = resolve_all_cloze(text_field) if fact_text_cloze else text_field
    elif front_field or back_field:
        if front_field and back_field:
            fact_text = f"Q: {front_field}\nA: {back_field}"
        else:
            fact_text = front_field or back_field
        fact_format = "basic_fact"
    elif original_field:
        fact_text_cloze = original_field if has_cloze(original_field) else None
        fact_text = resolve_all_cloze(original_field) if fact_text_cloze else original_field
    else:
        return None

    if not fact_text or len(fact_text.strip()) < 3:
        return None

    # Metadata: try resource-tagged parsing first (some podcast decks carry rich tags)
    tag_info = _parse_tags_resource(tags_str, _tag_cfg(filename))
    subject = tag_info.get("subject")

    # Try podcast-deck tag parsing
    if not subject:
        dip_info = _parse_dip_tags(tags_str, _tag_cfg(filename))
        subject = dip_info.get("subject")

    # Specialty from filename for dip_psych
    specialty = None
    if filename == "dip_psych.apkg":
        specialty = "Psychiatry"

    # Use specialty as subject fallback
    if not subject and specialty:
        subject = specialty

    return {
        "fact_format": fact_format,
        "fact_text": fact_text,
        "fact_text_cloze": fact_text_cloze,
        "fact_context": extra,
        "fact_summary": None,
        "subject": subject,
        "organ_system": tag_info.get("organ_system"),
        "topic": tag_info.get("topic"),
        "specialty": specialty,
        "resource_refs": None,
        "tags_normalized": json.dumps(_normalize_tags(tags_str), ensure_ascii=False),
        "extraction_method": "rule_based",
    }


# ---------------------------------------------------------------------------
# Parser registry — maps a deck profile's `parser` key to an extractor.
# Add a fact format by writing a parser here and a profile in etl/deck_profiles/
# that references its key.
# ---------------------------------------------------------------------------

PARSERS = {
    "cloze_qid": _extract_cloze_qid,
    "cloze_review_hierarchy": _extract_cloze_review_hier,
    "cloze_review_specialty": _extract_cloze_review_spec,
    "cloze_resource": _extract_cloze_resource,
    "cloze_specialty_deck": _extract_cloze_specialty_deck,
    "cloze_podcast": _extract_cloze_podcast,
}


def _route_extractor(filename: str, note_model: str = ""):
    """Return the extraction function for a source, via its deck profile."""
    profile = REGISTRY.resolve(filename=filename, note_model=note_model)
    return PARSERS.get(profile.parser) if profile else None


# ---------------------------------------------------------------------------
# INSERT SQL
# ---------------------------------------------------------------------------

_INSERT_SQL = """
    INSERT INTO fact_cards
        (raw_card_id, fact_format,
         fact_text, fact_text_cloze, fact_context, fact_summary,
         subject, organ_system, topic, specialty,
         resource_refs, tags_normalized,
         extraction_method)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run(conn: sqlite3.Connection) -> dict:
    """Stage 4: Extract fact cards from raw_cards with card_type='fact'.

    Returns summary dict with counts and duration.
    """
    t0 = time.time()

    # Idempotency: skip raw_cards that already have fact_cards (incremental)
    existing = conn.execute("SELECT COUNT(*) FROM fact_cards").fetchone()[0]
    if existing > 0:
        log.info("fact_cards has %d existing rows; running incrementally "
                 "(skipping already-extracted cards)", existing)

    # Log start
    conn.execute(
        "INSERT INTO processing_log (stage, status) VALUES ('s04_fact_cards', 'started')"
    )
    conn.commit()

    # Fetch fact raw_cards not yet extracted (incremental)
    rows = conn.execute(
        """
        SELECT rc.raw_card_id, rc.field_data, rc.field_data_text,
               rc.anki_tags, rc.anki_deck_name, rc.card_format,
               sd.filename
        FROM raw_cards rc
        JOIN source_decks sd ON rc.deck_id = sd.deck_id
        WHERE rc.card_type = 'fact'
          AND NOT EXISTS (
              SELECT 1 FROM fact_cards fc WHERE fc.raw_card_id = rc.raw_card_id
          )
        """,
    ).fetchall()

    total_in = len(rows)
    log.info("Stage 4: Processing %d fact raw_cards", total_in)

    inserted = 0
    skipped = 0
    errors = 0
    source_counts: dict[str, int] = {}
    no_extractor = 0

    for (raw_card_id, field_data_json, field_data_text_json,
         tags_str, deck_name, card_format, filename) in rows:
        try:
            extractor = _route_extractor(filename)
            if extractor is None:
                log.warning("No extractor for filename=%s (raw_card_id=%d)",
                            filename, raw_card_id)
                no_extractor += 1
                skipped += 1
                continue

            field_data = json.loads(field_data_json) if field_data_json else {}
            field_data_text = json.loads(field_data_text_json) if field_data_text_json else {}

            result = extractor(
                field_data, field_data_text, tags_str or "",
                deck_name or "", card_format or "", filename,
            )

            if result is None:
                skipped += 1
                continue

            # Guard: skip if resolved fact_text is empty
            if not result["fact_text"] or not result["fact_text"].strip():
                skipped += 1
                continue

            conn.execute(
                _INSERT_SQL,
                (
                    raw_card_id,
                    result["fact_format"],
                    result["fact_text"],
                    result["fact_text_cloze"],
                    result["fact_context"],
                    result["fact_summary"],
                    result["subject"],
                    result["organ_system"],
                    result["topic"],
                    result["specialty"],
                    result["resource_refs"],
                    result["tags_normalized"],
                    result["extraction_method"],
                ),
            )
            inserted += 1
            source_counts[filename] = source_counts.get(filename, 0) + 1

        except Exception as exc:
            errors += 1
            log.error("Error extracting raw_card_id=%d (%s): %s",
                      raw_card_id, filename, exc)

    conn.commit()

    duration = time.time() - t0

    # Verification queries
    total_fc = conn.execute("SELECT COUNT(*) FROM fact_cards").fetchone()[0]
    format_dist = dict(conn.execute(
        "SELECT fact_format, COUNT(*) FROM fact_cards GROUP BY fact_format"
    ).fetchall())
    null_subject = conn.execute(
        "SELECT COUNT(*) FROM fact_cards WHERE subject IS NULL"
    ).fetchone()[0]
    null_specialty = conn.execute(
        "SELECT COUNT(*) FROM fact_cards WHERE specialty IS NULL"
    ).fetchone()[0]
    has_resources = conn.execute(
        "SELECT COUNT(*) FROM fact_cards WHERE resource_refs IS NOT NULL"
    ).fetchone()[0]
    unresolved_cloze = conn.execute(
        "SELECT COUNT(*) FROM fact_cards WHERE fact_text LIKE '%{{c%' OR fact_text LIKE '%[[oc%'"
    ).fetchone()[0]

    log.info("Stage 4 complete: %d in, %d inserted, %d skipped, %d errors (%.1fs)",
             total_in, inserted, skipped, errors, duration)
    log.info("Format distribution: %s", format_dist)
    log.info("Subject NULL: %d, Specialty NULL: %d, Has resources: %d",
             null_subject, null_specialty, has_resources)
    if unresolved_cloze > 0:
        log.warning("Unresolved cloze in fact_text: %d", unresolved_cloze)

    # Update processing_log
    config = {
        "source_counts": source_counts,
        "skipped": skipped,
        "no_extractor": no_extractor,
        "format_distribution": format_dist,
        "null_subject": null_subject,
        "null_specialty": null_specialty,
        "has_resource_refs": has_resources,
        "unresolved_cloze": unresolved_cloze,
    }
    conn.execute(
        """
        UPDATE processing_log
        SET status = 'completed',
            records_in = ?,
            records_out = ?,
            records_error = ?,
            duration_sec = ?,
            completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
            config = ?
        WHERE stage = 's04_fact_cards' AND status = 'started'
        """,
        (total_in, inserted, errors, round(duration, 2), json.dumps(config)),
    )
    conn.commit()

    return {
        "records_in": total_in,
        "records_out": inserted,
        "records_error": errors,
        "skipped": skipped,
        "source_counts": source_counts,
        "format_distribution": format_dist,
        "null_subject": null_subject,
        "null_specialty": null_specialty,
        "has_resource_refs": has_resources,
        "unresolved_cloze": unresolved_cloze,
        "duration_sec": round(duration, 2),
    }
