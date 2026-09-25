"""Tests for Phase A (ICD-10 -> specialty map) and Phase F (specialty-conditioned
summarization scoring). No DB or LLM required."""

from etl.ontology import specialty_map as sm
from eval import scoring


# ---------------------------------------------------------------------------
# Phase A: chapter -> specialty map
# ---------------------------------------------------------------------------

def test_specialty_map_core():
    cases = {
        "E11.9": "Endocrinology",
        "I10": "Cardiology",
        "I21.01": "Cardiology",
        "J45.909": "Pulmonology",
        "K70.30": "Gastroenterology",
        "N18.9": "Nephrology",       # renal
        "N40.0": "Urology",          # urinary/male genital
        "N80.0": "Obstetrics_Gynecology",
        "O80": "Obstetrics_Gynecology",
        "G40.909": "Neurology",
        "F32.9": "Psychiatry",
        "L40.0": "Dermatology",
        "M05.79": "Rheumatology",
        "C34.90": "Hematology_Oncology",
        "D64.9": "Hematology_Oncology",
        "H25.9": "Ophthalmology",
        "H66.90": "Otolaryngology",
        "P07.30": "Neonatology",
        "A41.9": "Infectious_Disease",
    }
    for code, spec in cases.items():
        assert sm.specialty_for_icd10(code) == spec, f"{code} -> {sm.specialty_for_icd10(code)} != {spec}"


def test_specialty_map_crosscutting_is_none():
    # symptoms (R), external causes (V-Y), generic Z status -> no home
    for code in ["R55", "R10.9", "Z79.4", "Z00.00", "Z88.0", "V43.52", "Y93.6"]:
        assert sm.specialty_for_icd10(code) is None, f"{code} should be None"


def test_specialty_map_multihome():
    assert sm.specialties_for_icd10("G00.9") == ("Neurology", "Infectious_Disease")
    assert sm.specialties_for_icd10("M00.05") == ("Rheumatology", "Infectious_Disease")
    assert sm.specialties_for_icd10("I01.1") == ("Cardiology", "Rheumatology")
    assert sm.specialties_for_icd10("S72.001A") == ("Emergency_Medicine", "Orthopedics", "General_Surgery")
    assert "Medical_Genetics" in sm.specialties_for_icd10("Q21.0")        # congenital heart
    assert sm.specialties_for_icd10("U07.1") == ("Infectious_Disease",)   # COVID
    assert sm.specialties_for_icd10("Z51.11") == ("Hematology_Oncology",)  # chemo encounter
    assert sm.specialties_for_icd10("Z34.90") == ("Obstetrics_Gynecology",)  # pregnancy supervision
    assert sm.specialties_for_icd10("F03.90") == ("Neurology", "Psychiatry")  # dementia (organic)
    assert sm.specialties_for_icd10("F32.9") == ("Psychiatry",)            # depression


def test_specialty_map_edges():
    assert sm.specialty_for_icd10("") is None
    assert sm.specialty_for_icd10(None) is None
    assert sm.specialty_for_icd10("E1") is None          # too short
    # H is split by sub-range
    assert sm.specialty_for_icd10("H10.9") == "Ophthalmology"
    assert sm.specialty_for_icd10("H81.10") == "Otolaryngology"


# ---------------------------------------------------------------------------
# Phase F: specialty-conditioned scoring
# ---------------------------------------------------------------------------

def _sc_gt(primary=None, relevant=None, excluded=None, involvement="high", neutral=None):
    return {
        "variant": "specialty_conditioned",
        "specialty": "nephrology",
        "involvement": involvement,
        "reference_summary": "ref",
        "tiers": {
            "primary": [{"display_name": p, "importance": "critical"} for p in (primary or [])],
            "relevant": [{"display_name": r, "importance": "critical"} for r in (relevant or [])],
            "neutral": [{"display_name": n} for n in (neutral or [])],
            "excluded_sample": [{"display_name": e} for e in (excluded or [])],
        },
    }


def test_sc_high_involvement_perfect():
    gts = [_sc_gt(primary=["acute kidney injury"], relevant=["type 2 diabetes mellitus"],
                  excluded=["atrial fibrillation"], involvement="high")]
    preds = [{"summary": "Nephrology: acute kidney injury is the active problem, "
                         "driven by the patient's type 2 diabetes mellitus."}]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert m["n_involved"] == 1 and m["n_absent"] == 0
    assert m["primary_recall_critical"] == 1.0
    assert m["primary_recall_complete"] == 1.0
    assert m["relevant_recall_critical"] == 1.0
    assert m["leakage_rate"] == 0.0          # atrial fibrillation not mentioned
    assert m["conditioned_f1"] == 1.0


def test_sc_leakage_penalized():
    gts = [_sc_gt(primary=["acute kidney injury"], relevant=["type 2 diabetes mellitus"],
                  excluded=["atrial fibrillation"], involvement="high")]
    preds = [{"summary": "Acute kidney injury and type 2 diabetes mellitus; also "
                         "notes atrial fibrillation unrelated to renal care."}]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert m["primary_recall_critical"] == 1.0   # recall still perfect
    assert m["leakage_rate"] == 1.0          # pulled in the excluded finding
    assert m["conditioned_f1"] == 0.0        # precision collapses


def test_sc_abstention():
    gts = [
        _sc_gt(excluded=["atrial fibrillation"], involvement="absent"),
        _sc_gt(excluded=["atrial fibrillation"], involvement="absent"),
    ]
    preds = [
        {"summary": "No active nephrology problems."},                       # abstains
        {"summary": "The patient has atrial fibrillation and a long cardiac "
                    "history with multiple prior admissions and procedures."},  # populated + leaks
    ]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert m["n_involved"] == 0 and m["n_absent"] == 2
    assert m["abstention_accuracy"] == 0.5   # one abstained, one didn't
    assert m["absent_leakage_rate"] == 0.5   # the populated one leaked the excluded finding


def test_sc_routing_mixed_batch():
    gts = [
        _sc_gt(primary=["acute kidney injury"], involvement="high"),
        _sc_gt(excluded=["atrial fibrillation"], involvement="absent"),
    ]
    preds = [{"summary": "Acute kidney injury active."}, {"summary": "No active nephrology problems."}]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert m["n_involved"] == 1 and m["n_absent"] == 1
    assert "primary_recall_critical" in m and "abstention_accuracy" in m


def test_specialty_tiers_typed_relevance():
    from etl.stages.s10f_specialty import _build_specialty_tiers
    present = {1, 2}  # 1 = CKD (Nephrology), 2 = DM (Endocrinology)
    dx_specs = {1: ("Nephrology",), 2: ("Endocrinology",)}
    typed_adj = {1: {2}, 2: {1}}   # CKD <-> DM Class 1 (due_to)
    residual_adj = {}
    finding_owners = {"elevated creatinine": {1}, "hyperglycemia": {2}}
    out = _build_specialty_tiers(present, finding_owners, dx_specs, typed_adj, residual_adj,
                                 all_specialties={"Nephrology", "Endocrinology", "Cardiology"})
    by = {t["specialty"]: t for t in out}
    # Nephrology involved (CKD present-home); DM finding RELEVANT via the typed edge
    assert by["Nephrology"]["involvement"] == "involved"
    assert by["Nephrology"]["primary"] == ["elevated creatinine"]
    assert by["Nephrology"]["relevant"] == ["hyperglycemia"]
    assert by["Endocrinology"]["primary"] == ["hyperglycemia"]
    assert by["Endocrinology"]["relevant"] == ["elevated creatinine"]
    # Cardiology: no present home diagnosis -> absent (abstention control)
    assert by["Cardiology"]["involvement"] == "absent"
    assert set(by["Cardiology"]["excluded_sample"]) == {"elevated creatinine", "hyperglycemia"}


def test_specialty_tiers_gold_vs_neutral():
    from etl.stages.s10f_specialty import _build_specialty_tiers
    present = {1, 2}  # 1 = CKD (Nephrology), 2 = a derm dx (Dermatology)
    dx_specs = {1: ("Nephrology",), 2: ("Dermatology",)}
    finding_owners = {"creatinine": {1}, "rash": {2}}
    SP = {"Nephrology", "Dermatology"}
    # no link -> derm finding is EXCLUDED for the nephrology view
    out = {t["specialty"]: t for t in _build_specialty_tiers(present, finding_owners, dx_specs, {}, {}, SP)}
    assert out["Nephrology"]["relevant"] == [] and out["Nephrology"]["excluded_sample"] == ["rash"]
    # GOLD link (Class-1 definitional / curated) -> credited RELEVANT
    g = {t["specialty"]: t for t in _build_specialty_tiers(present, finding_owners, dx_specs, {2: {1}, 1: {2}}, {}, SP)}
    assert g["Nephrology"]["relevant"] == ["rash"] and g["Nephrology"]["neutral"] == []
    # NEUTRAL link (Class-2/3) -> NEUTRAL buffer (not credited, not a leak)
    n = {t["specialty"]: t for t in _build_specialty_tiers(present, finding_owners, dx_specs, {}, {2: {1}, 1: {2}}, SP)}
    assert n["Nephrology"]["relevant"] == [] and n["Nephrology"]["neutral"] == ["rash"]


def test_sc_neutral_buffer():
    # a finding in the NEUTRAL tier (Class-2/3 link) is neither required for recall nor a leak;
    # leakage_rate excludes it, leakage_rate_strict folds it back in (boundary sensitivity).
    gts = [_sc_gt(primary=["acute kidney injury"], neutral=["essential hypertension"],
                  involvement="high")]
    preds = [{"summary": "Acute kidney injury is the active renal problem; essential "
                         "hypertension is noted as a contributor."}]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert m["primary_recall_critical"] == 1.0
    assert m["leakage_rate"] == 0.0          # no excluded findings -> mentioning neutral is NOT a leak
    assert m["leakage_rate_strict"] == 1.0   # graph-wide boundary folds the neutral mention back in


def test_value_normalizer_polarity():
    from eval import value_match as vm
    # raw value + reference range with the correct direction -> credited
    assert vm.value_polarity_match("Elevated ACTH level", "ACTH 200 pg/mL (normal 10-60)")
    assert vm.value_polarity_match("Decreased morning cortisol level", "8 AM cortisol 3 ug/dL (normal 7-25)")
    assert vm.value_polarity_match("Hyperkalemia", "potassium 5.5 mmol/L, elevated")
    # WRONG direction must NOT be credited (the whole point — polarity safety)
    assert not vm.value_polarity_match("Decreased morning cortisol level", "8 AM cortisol 30 ug/dL (normal 7-25)")
    assert not vm.value_polarity_match("Hyperkalemia", "potassium 3.0 mmol/L (normal 3.5-5.0)")
    # not a lab interpretation, or no direction signal -> no spurious credit
    assert not vm.value_polarity_match("Orthostatic hypotension", "BP 100/60 mmHg")
    assert not vm.value_polarity_match("Elevated ACTH level", "ACTH 200 pg/mL")  # bare value, no range/flag


def test_negation_guard():
    from eval import value_match as vm
    assert vm.negated_mention("orthostatic hypotension", "Patient denies orthostatic hypotension")
    assert vm.negated_mention("atrial fibrillation", "no atrial fibrillation on telemetry")
    assert not vm.negated_mention("atrial fibrillation", "atrial fibrillation with RVR")
    assert not vm.negated_mention("atrial fibrillation", "no afib history but new atrial fibrillation now")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} Phase A + F tests passed.")


if __name__ == "__main__":
    _run_all()
