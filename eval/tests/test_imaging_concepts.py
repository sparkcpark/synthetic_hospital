"""Unit tests for the imaging concept extractor (no database needed)."""

from eval.imaging_concepts import ConceptExtractor, canon, concept_f1_batch, stems, tokens

INVENTORY = [
    ("S1", "Acute appendicitis"), ("S1", "Appendicitis"),
    ("S2", "Nephrolithiasis"), ("S2", "Kidney stone"),
    ("S3", "Small bowel obstruction"),
    ("S4", "Pulmonary embolism"),
    ("S5", "Pneumonia"),
    ("S6", "Right lower quadrant pain"),
]


def test_tokens_expand_abbreviations_and_slashes():
    assert tokens("r/o SBO, s/p appy") [:4] == ["rule", "out", "small", "bowel"]
    assert tokens("IUP") == ["intrauterine", "pregnancy"]


def test_stems_and_canon_are_deterministic():
    assert stems("stones") >= {"stone", "stones"}
    assert "nephrolithiasis" in stems("nephrolithiases")
    # canon() is the shortest candidate stem (the paper's rule): it folds inflections
    # consistently between inventory and query rather than producing dictionary forms.
    assert canon("nephrolithiases") == "nephrolithias"
    assert canon("babies") == "babi"           # tie broken by (len, str), not hash order


def test_extractor_matches_synonyms_and_abbreviations():
    ex = ConceptExtractor(INVENTORY)
    assert ex.concepts("evaluate for kidney stone") == ex.concepts("nephrolithiasis?")
    assert ex.concepts("nephrolithiases?") == ex.concepts("nephrolithiasis?")
    assert "S3" in ex.concepts("r/o SBO")
    assert "S3" in ex.concepts("rule out small bowel obstruction")


def test_extractor_is_order_free_but_phrase_bound():
    ex = ConceptExtractor(INVENTORY)
    assert "S3" in ex.concepts("obstruction of the small bowel")
    # the words of a multi-word concept must appear near each other
    assert "S3" not in ex.concepts("small pneumothorax; later a bowel study; then an obstruction elsewhere far away")


def test_typo_snapping():
    ex = ConceptExtractor(INVENTORY)
    assert ex.concepts("evaluate for appendicitiss") == ex.concepts("evaluate for appendicitis")


def test_prf_and_batch():
    ex = ConceptExtractor(INVENTORY)
    p, r, f1, n_pred, n_ref = ex.prf("kidney stone or appendicitis", "nephrolithiasis")
    assert r == 1.0 and 0 < p < 1 and 0 < f1 < 1 and n_pred == 2 and n_ref == 1
    assert ex.prf("", "nephrolithiasis")[2] == 0.0
    m = concept_f1_batch(ex, ["pneumonia", "pulmonary embolism"], ["pneumonia", "PE"])
    assert m["clinical_question_concept_f1"] >= 0.5
    assert concept_f1_batch(ex, [], [])["clinical_question_concept_f1"] == 0.0


def test_curated_layer_absorbs_generic_ontology_names():
    ex = ConceptExtractor(INVENTORY + [("S9", "Pneumothorax")])
    c = ex.concepts("evaluate for pneumothorax")
    assert c == {"C:pneumothorax"}       # curated concept, not the raw ontology id
