"""Deterministic, abbreviation-aware clinical string matching.

Faithful implementation of the Appendix-A matcher used for physician scoring
(spec_v1.33 §17.9): a medical-abbreviation-aware similarity scorer combining
token-level Jaccard, min-set coverage, and character 3-gram Jaccard, with a
substring bonus. Used so model and human summarization scoring are consistent
(SapBERT can be swapped in later).

Two entry points:
  - phrase_in_text(phrase, text): recall-oriented — is a finding mentioned in a
    free-text summary? (abbreviation-expanded substring + token coverage)
  - similarity(a, b) / matches(a, b, T): pairwise — used for leakage/precision
    and for deduping/aligning finding lists in conditioned scoring.

No external dependencies; pure-Python and deterministic.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Weights for the pairwise similarity (Appendix-A): token-Jaccard 0.55,
# min-set coverage 0.30, char-3gram Jaccard 0.15, +0.10 substring bonus.
_W_TOKEN_JACCARD = 0.55
_W_MIN_SET_COVERAGE = 0.30
_W_CHAR_3GRAM = 0.15
_SUBSTRING_BONUS = 0.10

# Calibrated threshold (Appendix-A: T_F1* = 0.300 on the ICD-10<->SNOMED set).
DEFAULT_THRESHOLD = 0.300

# Token-coverage threshold for phrase_in_text recall matching.
_PHRASE_COVERAGE = 0.70

# Clinical stopwords (generic connective/structural tokens that should not
# carry matching weight).
_STOPWORDS = frozenset({
    "of", "the", "a", "an", "and", "or", "with", "without", "due", "to", "in",
    "on", "for", "by", "at", "as", "is", "was", "are", "patient", "history",
    "status", "post", "type", "unspecified", "other", "nos", "disease",
    "disorder", "syndrome", "chronic", "acute", "left", "right", "bilateral",
})

# ~90 common clinical shorthand expansions (Appendix-A: "80 common clinical
# shorthand expansions"). Expanded one extra notch for coverage.
_ABBREV_MAP: dict[str, str] = {
    "t1dm": "type 1 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "dm": "diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "htn": "hypertension",
    "ckd": "chronic kidney disease",
    "esrd": "end stage renal disease",
    "aki": "acute kidney injury",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "hfref": "heart failure with reduced ejection fraction",
    "hfpef": "heart failure with preserved ejection fraction",
    "chb": "complete heart block",
    "mi": "myocardial infarction",
    "stemi": "st elevation myocardial infarction",
    "nstemi": "non st elevation myocardial infarction",
    "afib": "atrial fibrillation",
    "af": "atrial fibrillation",
    "aflutter": "atrial flutter",
    "svt": "supraventricular tachycardia",
    "vt": "ventricular tachycardia",
    "vf": "ventricular fibrillation",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "vte": "venous thromboembolism",
    "copd": "chronic obstructive pulmonary disease",
    "ild": "interstitial lung disease",
    "osa": "obstructive sleep apnea",
    "ards": "acute respiratory distress syndrome",
    "uri": "upper respiratory infection",
    "uti": "urinary tract infection",
    "cap": "community acquired pneumonia",
    "hap": "hospital acquired pneumonia",
    "gerd": "gastroesophageal reflux disease",
    "ibd": "inflammatory bowel disease",
    "ibs": "irritable bowel syndrome",
    "uc": "ulcerative colitis",
    "gib": "gastrointestinal bleed",
    "ugib": "upper gastrointestinal bleed",
    "lgib": "lower gastrointestinal bleed",
    "cirrhosis": "cirrhosis",
    "nash": "nonalcoholic steatohepatitis",
    "nafld": "nonalcoholic fatty liver disease",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "sah": "subarachnoid hemorrhage",
    "ich": "intracerebral hemorrhage",
    "ms": "multiple sclerosis",
    "sz": "seizure",
    "ra": "rheumatoid arthritis",
    "sle": "systemic lupus erythematosus",
    "oa": "osteoarthritis",
    "gout": "gout",
    "hld": "hyperlipidemia",
    "dlp": "dyslipidemia",
    "hypothyroid": "hypothyroidism",
    "dka": "diabetic ketoacidosis",
    "hhs": "hyperosmolar hyperglycemic state",
    "dki": "diabetic ketoacidosis",
    "ods": "osmotic demyelination syndrome",
    "af ": "atrial fibrillation",
    "bph": "benign prostatic hyperplasia",
    "aaa": "abdominal aortic aneurysm",
    "pad": "peripheral artery disease",
    "pvd": "peripheral vascular disease",
    "anemia": "anemia",
    "ida": "iron deficiency anemia",
    "ckd": "chronic kidney disease",
    "sob": "shortness of breath",
    "doe": "dyspnea on exertion",
    "cp": "chest pain",
    "abd": "abdominal",
    "n/v": "nausea and vomiting",
    "loc": "loss of consciousness",
    "ams": "altered mental status",
    "fx": "fracture",
    "ca": "cancer",
    "mets": "metastases",
    "ln": "lymph node",
    "wbc": "white blood cell count",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet count",
    "cr": "creatinine",
    "bun": "blood urea nitrogen",
    "na": "sodium",
    "k": "potassium",
    "hco3": "bicarbonate",
    "ldl": "low density lipoprotein",
    "hdl": "high density lipoprotein",
    "tg": "triglycerides",
    "a1c": "hemoglobin a1c",
    "hba1c": "hemoglobin a1c",
    "bnp": "b type natriuretic peptide",
    "trop": "troponin",
    "lfts": "liver function tests",
    "tsh": "thyroid stimulating hormone",
    "egfr": "estimated glomerular filtration rate",
    "gfr": "glomerular filtration rate",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "ef": "ejection fraction",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _expand_abbreviations(text: str) -> str:
    """Lowercase and expand known abbreviations token-by-token."""
    text = text.lower()
    # Handle a few slash/punctuation forms before tokenizing.
    text = text.replace("n/v", " nausea and vomiting ")
    tokens = _WORD_RE.findall(text)
    out: list[str] = []
    for tok in tokens:
        out.append(_ABBREV_MAP.get(tok, tok))
    return " ".join(out)


@lru_cache(maxsize=8192)
def _content_tokens(text: str) -> frozenset[str]:
    """Abbreviation-expanded, stop-filtered content tokens."""
    expanded = _expand_abbreviations(text)
    toks = {t for t in _WORD_RE.findall(expanded) if t not in _STOPWORDS}
    return frozenset(toks)


@lru_cache(maxsize=8192)
def _normalized_phrase(text: str) -> str:
    """Abbreviation-expanded, whitespace-normalized phrase (keeps order)."""
    return " ".join(_expand_abbreviations(text).split())


def _char_3grams(text: str) -> frozenset[str]:
    s = "".join(_normalized_phrase(text).split())
    if len(s) < 3:
        return frozenset({s}) if s else frozenset()
    return frozenset(s[i:i + 3] for i in range(len(s) - 2))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def similarity(a: str, b: str) -> float:
    """Pairwise abbreviation-aware similarity in [0, 1] (Appendix-A formula)."""
    if not a or not b:
        return 0.0
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        # Fall back to char-gram only when content tokens vanish (e.g. all stop).
        return _W_CHAR_3GRAM * _jaccard(_char_3grams(a), _char_3grams(b))

    token_jacc = _jaccard(ta, tb)
    min_set_cov = len(ta & tb) / min(len(ta), len(tb))
    char_jacc = _jaccard(_char_3grams(a), _char_3grams(b))

    score = (
        _W_TOKEN_JACCARD * token_jacc
        + _W_MIN_SET_COVERAGE * min_set_cov
        + _W_CHAR_3GRAM * char_jacc
    )
    na, nb = _normalized_phrase(a), _normalized_phrase(b)
    if na and nb and (na in nb or nb in na):
        score += _SUBSTRING_BONUS
    return min(1.0, score)


def matches(a: str, b: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True if two clinical strings refer to the same concept (pairwise)."""
    return similarity(a, b) >= threshold


def phrase_in_text(phrase: str, text: str, coverage: float = _PHRASE_COVERAGE) -> bool:
    """Recall match: is `phrase` (a finding name) mentioned in free-text `text`?

    Hit if (a) the abbreviation-expanded phrase is a substring of the expanded
    text, or (b) at least `coverage` of the phrase's content tokens appear in
    the text's content-token set (catches abbreviation/synonym/word-order
    variation, e.g. "T2DM" vs "type 2 diabetes").
    """
    if not phrase or not text:
        return False
    np_phrase = _normalized_phrase(phrase)
    np_text = _normalized_phrase(text)
    if np_phrase and np_phrase in np_text:
        return True
    p_tokens = _content_tokens(phrase)
    if not p_tokens:
        return np_phrase in np_text
    t_tokens = _content_tokens(text)
    covered = len(p_tokens & t_tokens) / len(p_tokens)
    return covered >= coverage


def count_present(phrases: list[str], text: str, coverage: float = _PHRASE_COVERAGE) -> int:
    """How many of `phrases` are mentioned in `text` (recall numerator)."""
    return sum(1 for p in phrases if phrase_in_text(p, text, coverage))
