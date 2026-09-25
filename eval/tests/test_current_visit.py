"""Smoke tests for the current-visit summarization variant (spec_v1.33 §17).

Covers three layers without requiring a database or live LLM:
  1. eval.semantic_match — abbreviation-aware matching.
  2. etl.stages.s10_ground_truth._build_current_visit_core — deterministic GT
     (index selection, new/resolved deltas, must-include).
  3. eval.scoring — variant routing + current-visit metrics + abstention safety.

The s05_ontology import is stubbed so the deterministic GT core can be imported
without pulling the LLM client.
"""

import sys
import types

# --- Stub the LLM-backed ontology module before importing the GT stage ---
if "etl.stages.s05_ontology" not in sys.modules:
    _stub = types.ModuleType("etl.stages.s05_ontology")
    _stub.MODEL = "stub-model"

    def _no_llm(*_a, **_k):  # pragma: no cover - guard
        raise RuntimeError("LLM disabled in tests")

    _stub._call_with_retry = _no_llm
    sys.modules["etl.stages.s05_ontology"] = _stub

from etl.stages import s10_ground_truth as s10  # noqa: E402
from eval import scoring, semantic_match  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Semantic matcher
# ---------------------------------------------------------------------------

def test_phrase_in_text_abbreviation():
    assert semantic_match.phrase_in_text("T2DM", "history of type 2 diabetes mellitus")
    assert semantic_match.phrase_in_text("CKD", "stage 3 chronic kidney disease noted")
    assert not semantic_match.phrase_in_text("pulmonary embolism", "patient has a sore throat")


def test_similarity_synonym_vs_unrelated():
    syn = semantic_match.similarity("T2DM", "type 2 diabetes mellitus")
    unrel = semantic_match.similarity("acute kidney injury", "atrial fibrillation")
    assert syn >= semantic_match.DEFAULT_THRESHOLD
    assert syn > unrel
    assert semantic_match.matches("HTN", "hypertension")


# ---------------------------------------------------------------------------
# 2. Deterministic GT core
# ---------------------------------------------------------------------------

def _enc(order, qids, eid=None):
    return {
        "encounter_id": eid if eid is not None else 100 + order,
        "encounter_order": order,
        "encounter_date": f"2024-0{order}-01",
        "encounter_type": "outpatient",
        "chief_complaint": f"complaint {order}",
        "department": "Internal Medicine",
        "note_text": f"note for encounter {order}",
        "source_question_ids": f"[{qids}]",
    }


_KFQ = {
    1: [{"display_name": "Finding A", "finding_type": "symptom"},
        {"display_name": "Finding B", "finding_type": "sign"}],
    2: [{"display_name": "Finding B", "finding_type": "sign"},
        {"display_name": "Finding C", "finding_type": "lab_value"}],
    3: [{"display_name": "Finding D", "finding_type": "lab_value"}],
}

# background/distractor findings per qid -> off-target (precision) set
_OFF = {
    2: [{"display_name": "Incidental skin tag", "finding_type": "sign"}],
}


def test_build_core_deltas_and_index():
    encs = [_enc(1, 1), _enc(2, 2)]
    core = s10._build_current_visit_core(encs, 1, _KFQ, _OFF)  # k=1 = terminal
    assert core is not None
    assert core["variant"] == "current_visit"
    assert core["index_encounter_order"] == 2
    assert core["index_encounter_id"] == 102
    assert core["index_position"] == 1
    assert core["has_future"] is False               # terminal anchor, no future
    assert core["future_findings"] == []
    new_names = {f["display_name"] for f in core["deltas"]["new"]}
    resolved_names = {f["display_name"] for f in core["deltas"]["resolved"]}
    must_names = {f["display_name"] for f in core["must_include_findings"]}
    off_names = {f["display_name"] for f in core["off_target_findings"]}
    assert new_names == {"Finding C"}        # C is first-seen at the index visit
    assert resolved_names == {"Finding A"}   # A present last visit, absent now
    assert must_names == {"Finding B", "Finding C"}  # active at the index visit
    assert off_names == {"Incidental skin tag"}      # background/distractor at index
    assert "complaint 2" in core["clinical_question"]


def test_build_core_requires_prior():
    # k with no prior (k=0) or out of range -> None
    encs = [_enc(1, 1), _enc(2, 2)]
    assert s10._build_current_visit_core(encs, 0, _KFQ) is None   # no prior
    assert s10._build_current_visit_core(encs, 2, _KFQ) is None   # out of range
    assert s10._build_current_visit_core([_enc(1, 1)], 1, _KFQ) is None
    assert s10._build_current_visit_core([], 1, _KFQ) is None


def test_select_index_positions():
    assert s10._select_index_positions(2) == [1]
    assert s10._select_index_positions(3) == [1, 2]
    assert s10._select_index_positions(4) == [1, 2, 3]
    assert s10._select_index_positions(7) == [1, 4, 6]   # early / mid / terminal
    assert s10._select_index_positions(10) == [1, 5, 9]


def test_interior_index_holds_out_future():
    # 3 encounters; anchor on the MIDDLE one (k=1) -> the 3rd encounter's finding
    # is future-only and must not be asserted.
    encs = [_enc(1, 1), _enc(2, 2), _enc(3, 3)]
    core = s10._build_current_visit_core(encs, 1, _KFQ, _OFF)
    assert core["index_position"] == 1
    assert core["has_future"] is True
    assert core["index_encounter_id"] == 102          # the middle encounter
    future_names = {f["display_name"] for f in core["future_findings"]}
    assert future_names == {"Finding D"}              # only appears in encounter 3
    # prompt context must exclude the future encounter
    assert all(e["encounter_order"] < 2 for e in core["_prior_encs"])


# ---------------------------------------------------------------------------
# 3. Scoring — variant routing + current-visit metrics
# ---------------------------------------------------------------------------

def _cv_gt(must, new, resolved, off=None, future=None, variant="current_visit"):
    return {
        "variant": variant,
        "reference_summary": "ref",
        "must_include_findings": [{"display_name": m} for m in must],
        "deltas": {
            "new": [{"display_name": n} for n in new],
            "resolved": [{"display_name": r} for r in resolved],
        },
        "off_target_findings": [{"display_name": o} for o in (off or [])],
        "future_findings": [{"display_name": f} for f in (future or [])],
    }


def test_current_visit_metrics_routing():
    gts = [_cv_gt(["chest pain", "elevated troponin"], ["elevated troponin"], ["fever"],
                  off=["incidental skin tag"])]
    preds = [{"summary": "At today's visit the patient developed elevated troponin; "
                         "the prior fever has resolved. Active problem: chest pain."}]
    m = scoring.compute_all_metrics("context_summarization", preds, gts,
                                    ehr_texts=["chest pain elevated troponin fever"])
    assert "delta_coverage_new" in m            # proves current-visit branch ran
    assert "delta_coverage_resolved" in m
    assert "mean_summary_words" in m
    assert m["clinical_f1"] == 1.0              # both must-includes mentioned
    assert m["delta_coverage_new"] == 1.0
    assert m["delta_coverage_resolved"] == 1.0
    assert m["off_target_rate"] == 0.0          # skin tag NOT mentioned (focused)
    assert m["selection_f1"] == 1.0             # perfect recall + perfect precision
    assert 0.0 <= m["omission_rate"] <= 1.0


def test_off_target_penalizes_verbosity():
    # Same GT, but a verbose summary dumps the off-target finding.
    gts = [_cv_gt(["chest pain", "elevated troponin"], ["elevated troponin"], ["fever"],
                  off=["incidental skin tag"])]
    preds = [{"summary": "Chest pain and elevated troponin; fever resolved. Also notes "
                         "an incidental skin tag unrelated to the presentation."}]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert m["clinical_f1"] == 1.0              # recall still perfect
    assert m["off_target_rate"] == 1.0         # pulled in the off-target finding
    assert m["selection_f1"] == 0.0            # precision collapses -> selection F1 0
    # verbosity is no longer a free win: same recall, worse selection_f1


def test_current_visit_partial_and_empty_deltas():
    # Model omits the resolved finding; one item has empty deltas (no crash).
    gts = [
        _cv_gt(["chest pain", "elevated troponin"], ["elevated troponin"], ["fever"]),
        _cv_gt(["hyperkalemia"], [], []),
    ]
    preds = [
        {"summary": "New elevated troponin noted; chest pain ongoing."},
        {"summary": "Stable interval; hyperkalemia managed."},
    ]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert m["delta_coverage_resolved"] == 0.0  # fever not mentioned (item 2 skipped: empty)
    assert m["delta_coverage_new"] == 1.0
    assert m["clinical_f1"] > 0.0


def test_future_leakage_penalized():
    # A summary that asserts a future-only finding leaks the temporal cutoff.
    gts = [_cv_gt(["chest pain"], ["chest pain"], [], future=["myocardial infarction"])]
    clean = [{"summary": "Today the patient presents with chest pain."}]
    leaky = [{"summary": "Chest pain today; will go on to have a myocardial infarction."}]
    mc = scoring.compute_all_metrics("context_summarization", clean, gts)
    ml = scoring.compute_all_metrics("context_summarization", leaky, gts)
    assert mc["future_leakage_rate"] == 0.0
    assert ml["future_leakage_rate"] == 1.0       # leaked the future finding
    assert ml["selection_f1"] < mc["selection_f1"]  # leakage costs precision


def test_unconditioned_still_routes_and_scores():
    gts = [{"variant": "unconditioned",
            "reference_summary": "ref",
            "must_include_findings": [{"display_name": "chest pain"}]}]
    preds = [{"summary": "The patient reports chest pain."}]
    m = scoring.compute_all_metrics("context_summarization", preds, gts)
    assert "delta_coverage_new" not in m        # unconditioned branch
    assert m["clinical_f1"] == 1.0
    assert "mean_summary_words" in m


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} current-visit smoke tests passed.")


if __name__ == "__main__":
    _run_all()
