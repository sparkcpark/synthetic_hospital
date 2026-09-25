"""Declarative deck-profile registry.

Replaces per-filename branching in the ETL with data-driven source configuration.
Each source deck/format is described by a `DeckProfile` loaded from a YAML file in
`etl/deck_profiles/`. To support a new deck, a user drops in a profile — no code
changes required.

A profile answers four questions the pipeline used to hardcode:
  * which cards belong to this source   -> `filename_globs` / `note_model_substrings`
  * what role the source plays          -> `role` (board_exam | fact)
  * how a card should be classified     -> `card_format` (used by Stage 2)
  * which parser extracts its fields    -> `parser` (a key resolved by Stage 3/4)

See `etl/stages/EXTENDING.md` for the full design.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROFILES_DIR = Path(__file__).resolve().parent / "deck_profiles"


@dataclass(frozen=True)
class DeckProfile:
    name: str
    role: str  # "board_exam" | "fact"
    parser: str  # parser key used by Stage 3 / Stage 4 dispatch
    card_format: str = ""  # classification hint for Stage 2
    filename_globs: tuple[str, ...] = ()
    note_model_substrings: tuple[str, ...] = ()
    dedup_priority: int = 999  # lower = more specific = wins dedup
    specialty_fallback: str = ""  # for single-specialty decks
    tag_root: str = ""  # optional namespace token for the source's tag hierarchy
    # Source-specific tag conventions consumed by the Stage 4 parsers. Keeping
    # these here (rather than hardcoded in code) means the pipeline ships no
    # source/vendor tokens. See etl/deck_profiles/README.md for the keys.
    tag_config: dict = field(default_factory=dict)

    def matches_filename(self, filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, g) for g in self.filename_globs)

    def matches_model(self, note_model: str) -> bool:
        m = (note_model or "").lower()
        return any(sub.lower() in m for sub in self.note_model_substrings)

    def concrete_filenames(self) -> tuple[str, ...]:
        """Filename globs that are plain filenames (no wildcard)."""
        return tuple(g for g in self.filename_globs if not any(c in g for c in "*?["))


class DeckRegistry:
    def __init__(self, profiles: list[DeckProfile]):
        # Sort by dedup_priority so filename resolution is deterministic when
        # multiple profiles could match: the more specific one wins.
        self.profiles = sorted(profiles, key=lambda p: p.dedup_priority)

    # -- resolution ---------------------------------------------------------
    def resolve(self, filename: str = "", note_model: str = "") -> DeckProfile | None:
        """Return the best-matching profile, preferring a filename match."""
        if filename:
            for p in self.profiles:
                if p.matches_filename(filename):
                    return p
        if note_model:
            for p in self.profiles:
                if p.matches_model(note_model):
                    return p
        return None

    def role_for(self, filename: str) -> str | None:
        p = self.resolve(filename=filename)
        return p.role if p else None

    def priority_for(self, filename: str) -> int:
        p = self.resolve(filename=filename)
        return p.dedup_priority if p else 999

    # -- backward-compatible views (used by etl.config) ---------------------
    def deck_type_map(self) -> dict[str, str]:
        """{concrete filename -> role} for the explicit filenames profiles name."""
        out: dict[str, str] = {}
        for p in self.profiles:
            for g in p.concrete_filenames():
                out[g] = p.role
        return out

    def specificity_map(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.profiles:
            for g in p.concrete_filenames():
                out[g] = p.dedup_priority
        return out


def load_profiles(profiles_dir: Path = PROFILES_DIR) -> DeckRegistry:
    profiles: list[DeckProfile] = []
    if profiles_dir.is_dir():
        for path in sorted(profiles_dir.glob("*.yaml")) + sorted(profiles_dir.glob("*.yml")):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            profiles.append(
                DeckProfile(
                    name=data.get("name", path.stem),
                    role=data["role"],
                    parser=data["parser"],
                    card_format=data.get("card_format", ""),
                    filename_globs=tuple(data.get("filename_globs", []) or []),
                    note_model_substrings=tuple(data.get("note_model_substrings", []) or []),
                    dedup_priority=int(data.get("dedup_priority", 999)),
                    specialty_fallback=data.get("specialty_fallback", "") or "",
                    tag_root=data.get("tag_root", "") or "",
                    tag_config=dict(data.get("tag_config", {}) or {}),
                )
            )
    return DeckRegistry(profiles)


# Module-level singleton loaded from the example profiles.
REGISTRY = load_profiles()


# ===========================================================================
# PDF sources
# ===========================================================================

PDF_PROFILES_DIR = Path(__file__).resolve().parent / "pdf_profiles"


@dataclass(frozen=True)
class PdfProfile:
    name: str
    layout: str  # "toc" | "bullets" | "font_headings" — selects the parser
    display_name: str = ""
    filename_globs: tuple[str, ...] = ()
    # How Stage 1d derives per-section metadata:
    topic_from: str = "section_title"  # "section_title" | "llm"
    context_style: str = "hierarchy"  # "hierarchy" | "episode"

    def matches_filename(self, filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, g) for g in self.filename_globs)

    @property
    def needs_topic_enrichment(self) -> bool:
        return self.topic_from == "llm"


class PdfRegistry:
    def __init__(self, profiles: list[PdfProfile]):
        self.profiles = list(profiles)

    def resolve(self, filename: str) -> PdfProfile | None:
        for p in self.profiles:
            if p.matches_filename(filename):
                return p
        return None

    def display_name(self, filename: str) -> str:
        p = self.resolve(filename)
        return p.display_name if (p and p.display_name) else filename

    def enrichment_globs(self) -> list[str]:
        """Filename globs of PDF profiles that need LLM topic enrichment."""
        globs: list[str] = []
        for p in self.profiles:
            if p.needs_topic_enrichment:
                globs.extend(p.filename_globs)
        return globs


def load_pdf_profiles(profiles_dir: Path = PDF_PROFILES_DIR) -> PdfRegistry:
    profiles: list[PdfProfile] = []
    if profiles_dir.is_dir():
        for path in sorted(profiles_dir.glob("*.yaml")) + sorted(profiles_dir.glob("*.yml")):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            profiles.append(
                PdfProfile(
                    name=data.get("name", path.stem),
                    layout=data["layout"],
                    display_name=data.get("display_name", "") or "",
                    filename_globs=tuple(data.get("filename_globs", []) or []),
                    topic_from=data.get("topic_from", "section_title") or "section_title",
                    context_style=data.get("context_style", "hierarchy") or "hierarchy",
                )
            )
    return PdfRegistry(profiles)


PDF_REGISTRY = load_pdf_profiles()
