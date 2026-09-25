"""Chart-neutral category sets for patient-diagnosis scoring.

A predicted diagnosis that does not match the graph-derived reference but falls in
the patient's *chart-neutral set* (the ICD-10 categories of conditions the chart
documents in its profile: chronic conditions, surgical history, smoking and
alcohol status) is neither credited nor penalized. The sets are computed
deterministically by `scripts/chart_neutral_sets.py`; this module exposes them to
the scorer with a per-process cache so that single-item scoring (the reward
endpoint) does not recompute them on every call.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "chart_neutral_sets.py"

_lock = threading.Lock()
_cache: dict[int, set[str]] | None = None


def _load_script_module():
    spec = importlib.util.spec_from_file_location("chart_neutral_sets", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def neutral_sets(conn, refresh: bool = False) -> dict[int, set[str]]:
    """Return {patient_id: {3-char ICD-10 categories}} for every patient.

    Computed once per process from the `longitudinal_patients.profile` column
    (plus the benchmark's own `diagnoses` table and, when loaded, ICD-10-CM
    titles in `terminology_codes`), then cached.
    """
    global _cache
    with _lock:
        if _cache is None or refresh:
            module = _load_script_module()
            sets, _how, _unmapped = module.neutral_sets(conn.cursor())
            _cache = {int(pid): set(cats) for pid, cats in sets.items()}
        return _cache


def neutral_categories_for(conn, patient_id: int) -> set[str]:
    """Chart-neutral categories for one patient (empty set if unknown)."""
    return set(neutral_sets(conn).get(int(patient_id), set()))
