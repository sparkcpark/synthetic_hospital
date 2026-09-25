"""Pipeline configuration. All paths, deck type mappings, and specificity rankings are defined here."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APKG_DIR = PROJECT_ROOT / "apkg"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "benchmark.db"

# --- Deck configuration ---
# Sources are now described declaratively by deck profiles in etl/deck_profiles/
# (see etl/deck_profiles/README.md). Each profile tags a source with a role
# ("board_exam" or "fact"), a parser, and match rules. To add your own deck,
# drop in a profile — no code change needed.
#
# The maps below are derived from those profiles for backward compatibility.
# DECK_TYPE_MAP / DECK_SPECIFICITY only list profiles that name explicit files;
# wildcard-matched sources are resolved at runtime via etl.deck_registry.REGISTRY.
from etl.deck_registry import REGISTRY  # noqa: E402

DECK_TYPE_MAP: dict[str, str] = REGISTRY.deck_type_map()
DECK_SPECIFICITY: dict[str, int] = REGISTRY.specificity_map()

# --- Files excluded from ingestion ---
# Filenames to skip even if present in APKG_DIR (e.g. non-.apkg exports).
EXCLUDED_FILES: set[str] = set()
