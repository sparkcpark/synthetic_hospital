"""Terminology file discovery and CMS parsing (pure filesystem tests, no database)."""

from pathlib import Path

from epic_sim.migrate import load_terminology as lt

CMS_ORDER_LINES = [
    # order  code     hdr  short description                                            long description
    "00001 A00     0 Cholera                                                      Cholera",
    "00002 A000    1 Cholera due to Vibrio cholerae 01, biovar cholerae           Cholera due to Vibrio cholerae 01, biovar cholerae",
    "00003 I2101   1 ST elevation (STEMI) myocardial infarction involving left main coronary artery ST elevation (STEMI) myocardial infarction involving left main coronary artery",
]


def _touch(p: Path, text: str = "x\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_parse_cms_order_lines():
    rows = lt.parse_cms_order_lines(CMS_ORDER_LINES)
    assert rows[0] == ("A00", "Cholera", "0")
    assert rows[1][0] == "A00.0" and rows[1][2] == "1"
    assert rows[2][0] == "I21.01" and "myocardial infarction" in rows[2][1]


def test_build_icd10_csv_from_cms_txt(tmp_path: Path):
    src = _touch(tmp_path / "icd10cm-order-2026.txt", "\n".join(CMS_ORDER_LINES) + "\n")
    dest = lt.build_icd10_csv_from_cms(src, tmp_path / "icd10cm_2026.csv")
    text = dest.read_text()
    assert text.splitlines()[0] == "code,description,is_header"
    assert "I21.01" in text
    assert lt.find_icd10_csv(tmp_path) == dest
    assert lt.find_icd10_cms_source(tmp_path) == src


def test_discovery_is_layout_and_version_agnostic(tmp_path: Path):
    # SNOMED International 2027 release, original folder layout
    sn = tmp_path / "SnomedCT_InternationalRF2_PRODUCTION_20270131T120000Z" / "Snapshot" / "Terminology"
    concept = _touch(sn / "sct2_Concept_Snapshot_INT_20270131.txt")
    desc = _touch(sn / "sct2_Description_Snapshot-en_INT_20270131.txt")
    _touch(sn / "sct2_TextDefinition_Snapshot-en_INT_20270131.txt")
    # LOINC 2.83 in its own folder, plus a stray Loinc.csv elsewhere
    loinc = _touch(tmp_path / "Loinc_2.83" / "LoincTable" / "Loinc.csv")
    _touch(tmp_path / "other" / "Loinc.csv")
    found = lt.discover(tmp_path)
    assert found["snomedct"] == (concept, desc)
    assert found["loinc"] == loinc
    assert found["icd10cm"] is None


def test_discovery_accepts_flat_snomed_files(tmp_path: Path):
    concept = _touch(tmp_path / "sct2_Concept_Snapshot_US1000124_20250901.txt")
    desc = _touch(tmp_path / "sct2_Description_Snapshot-en_US1000124_20250901.txt")
    assert lt.find_snomed_files(tmp_path) == (concept, desc)


def test_discovery_on_missing_dir(tmp_path: Path):
    found = lt.discover(tmp_path / "does-not-exist")
    assert all(v is None for v in found.values())
