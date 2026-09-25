"""Value-normalizer + negation guard for the specialty-summarization matcher cascade.

Bucket A (raw value vs interpretation): a summary reports a lab as a value
("ACTH 200 pg/mL (normal 10-60)") while the GT finding is an interpretation
("Elevated ACTH level"). This credits the match ONLY when the direction derived
from the summary (an explicit flag word, or an in-summary reference range) AGREES
with the GT finding's direction — polarity-preserved, so "ACTH 30 (low)" never
credits "Elevated ACTH". Plus a negex-lite negation guard so "no orthostatic
hypotension" is not counted as present.

Deterministic and auditable. Used only on the residual findings the lexical matcher
missed (cascade), so every value-normalized credit is a listable delta.
"""
import re

DIR_HIGH = {"elevated", "increased", "raised", "high", "elevation", "elevations"}
DIR_LOW = {"decreased", "reduced", "low", "depressed", "suppressed", "deficiency",
           "deficient", "diminished"}

# Interpreted hyper-/hypo- findings -> (analyte, direction).
HYPER_HYPO = {
    "hyperkalemia": ("potassium", "high"), "hypokalemia": ("potassium", "low"),
    "hypernatremia": ("sodium", "high"), "hyponatremia": ("sodium", "low"),
    "hypercalcemia": ("calcium", "high"), "hypocalcemia": ("calcium", "low"),
    "hyperglycemia": ("glucose", "high"), "hypoglycemia": ("glucose", "low"),
    "hypermagnesemia": ("magnesium", "high"), "hypomagnesemia": ("magnesium", "low"),
    "hyperphosphatemia": ("phosphate", "high"), "hypophosphatemia": ("phosphate", "low"),
    "hyperbilirubinemia": ("bilirubin", "high"), "hyperlactatemia": ("lactate", "high"),
}

# analyte -> UNAMBIGUOUS synonyms (no bare 1-2 char tokens like 'k'/'na'/'mg' that
# collide with units/words; '+'-suffixed ion forms are safe).
ANALYTE_SYNONYMS = {
    "sodium": ["sodium", "na+"], "potassium": ["potassium", "k+"],
    "calcium": ["calcium"], "magnesium": ["magnesium"], "phosphate": ["phosphate", "phosphorus"],
    "glucose": ["glucose", "blood sugar", "blood glucose"],
    "creatinine": ["creatinine"], "bun": ["bun", "urea nitrogen"],
    "cortisol": ["cortisol"], "acth": ["acth", "adrenocorticotropic", "adrenocorticotropin"],
    "tsh": ["tsh", "thyroid stimulating hormone", "thyrotropin"],
    "t4": ["thyroxine", "free t4", "ft4"], "hemoglobin": ["hemoglobin", "hgb"],
    "hematocrit": ["hematocrit", "hct"], "wbc": ["wbc", "white blood cell", "leukocyte"],
    "platelet": ["platelet", "plt"], "troponin": ["troponin"],
    "bnp": ["bnp", "natriuretic peptide", "nt-probnp"], "lactate": ["lactate", "lactic acid"],
    "bilirubin": ["bilirubin"], "albumin": ["albumin"], "ferritin": ["ferritin"],
    "b12": ["b12", "cobalamin"], "pth": ["pth", "parathyroid hormone"],
    "ck": ["creatine kinase"], "ast": ["ast", "aspartate aminotransferase"],
    "alt": ["alt", "alanine aminotransferase"],
}

NEG_CUES = ("no ", "not ", "without ", "denies ", "denied ", "negative for ",
            "absence of ", "absent ", "ruled out", "r/o ", "free of ",
            "no evidence of ", "resolution of ", "resolved")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)")


def _gt_analyte_dir(finding):
    """(analyte, expected_direction) for an interpreted-lab GT finding, else None."""
    low = finding.lower()
    for kw, ad in HYPER_HYPO.items():
        if kw in low:
            return ad
    toks = set(re.findall(r"[a-z]+", low))
    direction = "high" if toks & DIR_HIGH else ("low" if toks & DIR_LOW else None)
    if not direction:
        return None
    for analyte, syns in ANALYTE_SYNONYMS.items():
        if any(re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", low) for s in syns):
            return analyte, direction
    return None


def _window_direction(window):
    """Direction implied by a flag word or a reference range in the post-analyte window."""
    toks = set(re.findall(r"[a-z]+", window))
    if toks & DIR_HIGH:
        return "high"
    if toks & DIR_LOW:
        return "low"
    m = _RANGE.search(window)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        for nm in _NUM.finditer(window):              # first value NOT inside the range
            if m.start() <= nm.start() < m.end():
                continue
            v = float(nm.group())
            return "low" if v < lo else ("high" if v > hi else "normal")
    return None


def value_polarity_match(finding, summary):
    """True iff `summary` reports the GT lab finding's analyte with a direction that
    matches the finding's direction (polarity-preserved)."""
    ad = _gt_analyte_dir(finding)
    if not ad:
        return False
    analyte, expected = ad
    low = (summary or "").lower()
    for s in ANALYTE_SYNONYMS[analyte]:
        for m in re.finditer(rf"(?<![a-z]){re.escape(s)}(?![a-z])", low):
            pre = low[max(0, m.start() - 25):m.start()]
            if any(c in pre for c in NEG_CUES):
                continue
            if _window_direction(low[m.start():m.end() + 60]) == expected:
                return True
    return False


def negated_mention(finding, summary):
    """True iff the finding's phrase appears in `summary` ONLY in negated contexts
    (a lexical match should then be discounted). False if not literally present or
    found at least once unnegated."""
    low = (summary or "").lower()
    p = finding.lower().strip()
    idx = low.find(p)
    if idx == -1:
        return False
    while idx != -1:
        pre = low[max(0, idx - 30):idx]
        for b in (" but ", " however ", ";", ". "):   # negation does not cross a clause boundary
            k = pre.rfind(b)
            if k != -1:
                pre = pre[k + len(b):]
        if not any(c in pre for c in NEG_CUES):
            return False
        idx = low.find(p, idx + 1)
    return True
