"""Tests for the declarative deck-profile registry and stage wiring.

These run without any source decks or database — they validate that profiles
load, resolve correctly, and that each stage's PARSERS map covers the parser
keys the example profiles reference.
"""
import pytest

from etl.deck_registry import (
    DeckProfile, DeckRegistry, REGISTRY, load_profiles,
    PdfProfile, PdfRegistry, PDF_REGISTRY,
)


def _reg():
    return DeckRegistry([
        DeckProfile(
            name="board_a", role="board_exam", parser="mcq_fielded",
            card_format="mcq_vignette",
            filename_globs=("board_a.apkg", "board_fielded_*.apkg"),
            note_model_substrings=("nbme mcq",), dedup_priority=19,
        ),
        DeckProfile(
            name="fact_spec", role="fact", parser="cloze_specialty_deck",
            card_format="cloze_fact",
            filename_globs=("fact_specialty_*.apkg",),
            specialty_fallback="Cardiology", dedup_priority=1,
        ),
        DeckProfile(
            name="fact_general", role="fact", parser="cloze_resource",
            card_format="cloze_fact",
            filename_globs=("fact_general.apkg",), dedup_priority=30,
        ),
    ])


# -- resolution ------------------------------------------------------------

def test_resolve_by_exact_filename():
    r = _reg()
    p = r.resolve(filename="board_a.apkg")
    assert p and p.name == "board_a" and p.parser == "mcq_fielded"


def test_resolve_by_glob():
    r = _reg()
    assert r.resolve(filename="fact_specialty_neuro.apkg").name == "fact_spec"


def test_resolve_by_note_model():
    r = _reg()
    p = r.resolve(note_model="Some NBME MCQ model")
    assert p and p.role == "board_exam"


def test_filename_beats_model():
    r = _reg()
    # filename should win even if a model substring also matches something
    assert r.resolve(filename="fact_general.apkg", note_model="nbme mcq").name == "fact_general"


def test_unknown_returns_none():
    assert _reg().resolve(filename="mystery.apkg") is None


def test_role_and_priority_helpers():
    r = _reg()
    assert r.role_for("fact_general.apkg") == "fact"
    assert r.priority_for("fact_specialty_x.apkg") == 1
    assert r.role_for("nope.apkg") is None


# -- config-compat views ---------------------------------------------------

def test_deck_type_map_lists_concrete_only():
    r = _reg()
    m = r.deck_type_map()
    assert m["board_a.apkg"] == "board_exam"
    assert m["fact_general.apkg"] == "fact"
    # wildcard-only profile contributes no concrete filename
    assert not any("*" in k for k in m)


# -- example profiles on disk ---------------------------------------------

def test_example_profiles_load():
    assert len(REGISTRY.profiles) >= 1
    for p in REGISTRY.profiles:
        assert p.role in {"board_exam", "fact"}
        assert p.parser


def test_stage_parsers_cover_profile_keys():
    """Every parser key referenced by a profile exists in the owning stage."""
    from etl.stages import s03_extract_board as s3
    from etl.stages import s04_fact_cards as s4
    known = set(s3.PARSERS) | set(s4.PARSERS)
    for p in REGISTRY.profiles:
        assert p.parser in known, f"profile {p.name} references unknown parser {p.parser}"


def test_board_profiles_use_board_parsers():
    from etl.stages import s03_extract_board as s3
    for p in REGISTRY.profiles:
        if p.role == "board_exam":
            assert p.parser in s3.PARSERS


def test_fact_profiles_use_fact_parsers():
    from etl.stages import s04_fact_cards as s4
    for p in REGISTRY.profiles:
        if p.role == "fact":
            assert p.parser in s4.PARSERS


# -- PDF registry ----------------------------------------------------------

def _pdf_reg():
    return PdfRegistry([
        PdfProfile(name="toc_doc", layout="toc", display_name="TOC Doc",
                   filename_globs=("doc_toc_*.pdf", "example_im.pdf")),
        PdfProfile(name="bullet_doc", layout="bullets", display_name="Bullet Doc",
                   filename_globs=("doc_bullets_*.pdf",),
                   topic_from="llm", context_style="episode"),
    ])


def test_pdf_resolve_and_layout():
    r = _pdf_reg()
    assert r.resolve("example_im.pdf").layout == "toc"
    assert r.resolve("doc_bullets_1.pdf").layout == "bullets"
    assert r.resolve("unknown.pdf") is None


def test_pdf_display_name_fallback():
    r = _pdf_reg()
    assert r.display_name("doc_toc_x.pdf") == "TOC Doc"
    assert r.display_name("unknown.pdf") == "unknown.pdf"


def test_pdf_enrichment_globs():
    r = _pdf_reg()
    globs = r.enrichment_globs()
    assert "doc_bullets_*.pdf" in globs
    assert all("doc_toc" not in g for g in globs)  # toc doc not flagged


def test_example_pdf_profiles_load():
    assert len(PDF_REGISTRY.profiles) >= 1
    for p in PDF_REGISTRY.profiles:
        assert p.layout in {"toc", "bullets", "font_headings"}


def test_pdf_layouts_covered_by_parser():
    """Every example PDF profile's layout has a parser in pdf_sections."""
    pytest.importorskip("pdfplumber", reason="PDF ingestion needs the full requirements.txt")
    from etl.parsers import pdf_sections as ps
    for p in PDF_REGISTRY.profiles:
        assert hasattr(ps, {"toc": "parse_toc", "bullets": "parse_bullets",
                             "font_headings": "parse_font_headings"}[p.layout])


# -- tag_config-driven parsers --------------------------------------------

def test_qid_parser_uses_config_namespace():
    from etl.stages.s04_fact_cards import _parse_tags_qid
    cfg = {"qid_namespace": "QSRC"}
    out = _parse_tags_qid("QSRC::1::Cardiology::Cardiovascular::Angina", cfg)
    assert out["subject"] == "Cardiology"
    assert out["organ_system"] == "Cardiovascular"
    assert out["topic"] == "Angina"
    # wrong namespace -> no match
    assert _parse_tags_qid("OTHER::1::Cardiology", cfg)["subject"] is None
    # no config -> no parsing (no hardcoded vendor token)
    assert _parse_tags_qid("QSRC::1::Cardiology", {})["subject"] is None


def test_review_hier_parser_uses_config():
    from etl.stages.s04_fact_cards import _parse_tags_review_hier
    cfg = {"hier_root": "REVIEW", "hier_level2": "L2", "subject_namespace": "SUBJ"}
    out = _parse_tags_review_hier("REVIEW::L2::Neurology::Stroke", cfg)
    assert out["specialty"] == "Neurology"
    out2 = _parse_tags_review_hier("SUBJ::Pharmacology", cfg)
    assert out2["subject"] == "Pharmacology"


def test_podcast_parser_uses_config():
    from etl.stages.s04_fact_cards import _parse_dip_tags
    cfg = {"podcast_root": "PodDeck", "optional_branch": "Optional",
           "fixed_source_subjects": {"SourceP": "Pathology"}}
    out = _parse_dip_tags("PodDeck::Optional::SourceP::x", cfg)
    assert out["subject"] == "Pathology"
    # no config -> no parsing
    assert _parse_dip_tags("PodDeck::Optional::SourceP::x", {})["subject"] is None


def test_resource_refs_uses_config_fields():
    from etl.stages.s04_fact_cards import _parse_resource_refs
    cfg = {"resource_fields": ["Resource A"]}
    refs = _parse_resource_refs({"Resource A": "<img>"}, {"Resource A": "see page 5"}, cfg)
    assert refs == {"resource_a": "see page 5"}
    # no config -> nothing extracted
    assert _parse_resource_refs({"Resource A": "x"}, {"Resource A": "x"}, {}) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
