"""Load ICD-10-CM, SNOMED CT, and LOINC terminology into the terminology_codes table.

Usage:
    python -m epic_sim.migrate.load_terminology --auto [--download-icd10]   # what the container runs at boot
    python -m epic_sim.migrate.load_terminology [--icd10] [--snomed] [--loinc] [--all] [--force]
    python -m epic_sim.migrate.load_terminology --verify-only

Files are discovered under the ontology directory (EPIC_SIM_ONTOLOGY_DIR, default
data/ontology) by pattern, so any release version works and the archives may be
unpacked flat or in their original folder layout:

  ICD-10-CM   icd10cm_*.csv (parsed; produced here from the CMS "code descriptions
              in tabular order" release, icd10cm-order-*.txt / *.zip, or downloaded
              with --download-icd10 — the CMS file is public domain)
  SNOMED CT   **/sct2_Concept_Snapshot*.txt with the matching
              **/sct2_Description_Snapshot-en*.txt in the same folder
              (US Edition or International; UMLS licence required)
  LOINC       **/Loinc.csv (the LoincTable folder of the LOINC release; free registration)

--auto loads each system whose files are present and whose rows are not yet in
the table, so mounting the files and restarting the container is enough. Loading:
- ~97,000 ICD-10-CM codes, ~380,000 active SNOMED CT concepts (preferred term =
  FSN without semantic tag, else first synonym), ~97,000 active LOINC codes
- enables pg_trgm and a trigram index for fuzzy display-name search
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import psycopg

from epic_sim.app.config import settings

# SNOMED description types
FSN_TYPE = "900000000000003001"
SYNONYM_TYPE = "900000000000013009"
_SEMANTIC_TAG_RE = re.compile(r"\s*\([^)]+\)\s*$")

ICD10_CMS_URL = "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
ICD10_CSV_NAME = "icd10cm_2025.csv"

SYSTEMS = ("icd10cm", "snomedct", "loinc")

# Increase csv field size limit for SNOMED descriptions
csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def ontology_dir() -> Path:
    return Path(settings.ontology_dir)


def find_icd10_csv(root: Path | None = None) -> Path | None:
    root = root or ontology_dir()
    if not root.is_dir():
        return None
    hits = sorted(root.glob("icd10cm_*.csv")) + sorted(root.glob("**/icd10cm_*.csv"))
    return hits[0] if hits else None


def find_icd10_cms_source(root: Path | None = None) -> Path | None:
    """A CMS 'code descriptions in tabular order' release: the order .txt or the .zip."""
    root = root or ontology_dir()
    if not root.is_dir():
        return None
    for pat in ("**/icd10cm*order*.txt", "**/*order*.txt", "**/*code-descriptions*.zip", "**/icd10cm*.zip"):
        hits = sorted(p for p in root.glob(pat) if p.is_file())
        if hits:
            return hits[0]
    return None


def find_snomed_files(root: Path | None = None) -> tuple[Path, Path] | None:
    """(concept snapshot, English description snapshot) from any SNOMED CT RF2 release."""
    root = root or ontology_dir()
    if not root.is_dir():
        return None
    for concept in sorted(root.glob("**/sct2_Concept_Snapshot*.txt")):
        descs = sorted(concept.parent.glob("sct2_Description_Snapshot-en*.txt"))
        if descs:
            return concept, descs[0]
    return None


def find_loinc_csv(root: Path | None = None) -> Path | None:
    root = root or ontology_dir()
    if not root.is_dir():
        return None
    hits = sorted(root.glob("**/Loinc.csv"))
    # prefer the LoincTable copy over any other file of the same name
    hits.sort(key=lambda p: (0 if p.parent.name == "LoincTable" else 1, str(p)))
    return hits[0] if hits else None


def discover(root: Path | None = None) -> dict[str, object]:
    return {
        "icd10cm": find_icd10_csv(root),
        "icd10cm_source": find_icd10_cms_source(root),
        "snomedct": find_snomed_files(root),
        "loinc": find_loinc_csv(root),
    }


# ---------------------------------------------------------------------------
# ICD-10-CM: parse or download the CMS release
# ---------------------------------------------------------------------------

def parse_cms_order_lines(lines: list[str]) -> list[tuple[str, str, str]]:
    """Parse the fixed-width CMS 'order' file into (code, long description, is_header).

    Column layout (1-based): order number 1-5, code 7-13, header flag 15
    (0 = header/non-billable, 1 = billable), short description 17-76, long description 78-.
    """
    rows = []
    for line in lines:
        if len(line.strip()) < 17:
            continue
        code_raw = line[6:13].strip()
        is_header = line[14].strip()
        long_desc = line[77:].strip() if len(line) > 77 else line[16:77].strip()
        code = code_raw[:3] + "." + code_raw[3:] if len(code_raw) > 3 else code_raw
        rows.append((code, long_desc, is_header))
    return rows


def _order_lines_from(source: Path) -> list[str]:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            names = [n for n in zf.namelist() if "order" in n.lower() and n.endswith(".txt")]
            if not names:
                raise FileNotFoundError(f"no order .txt inside {source}")
            with zf.open(names[0]) as fin:
                return fin.read().decode("utf-8", errors="replace").splitlines()
    return source.read_text(encoding="utf-8", errors="replace").splitlines()


def write_icd10_csv(rows: list[tuple[str, str, str]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["code", "description", "is_header"])
        w.writerows(rows)
    return dest


def build_icd10_csv_from_cms(source: Path, dest: Path | None = None) -> Path:
    dest = dest or ontology_dir() / ICD10_CSV_NAME
    rows = parse_cms_order_lines(_order_lines_from(source))
    print(f"[icd10] parsed {len(rows)} codes from {source.name}")
    return write_icd10_csv(rows, dest)


def download_icd10(dest: Path | None = None, url: str = ICD10_CMS_URL, timeout: float = 120.0) -> Path:
    """Download the public-domain CMS ICD-10-CM release and write the parsed CSV."""
    import httpx  # core dependency

    dest = dest or ontology_dir() / ICD10_CSV_NAME
    print(f"[icd10] downloading {url}")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = Path(tmp.name)
    try:
        return build_icd10_csv_from_cms(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_sync_url() -> str:
    """Synchronous database URL from settings (strip SQLAlchemy dialect prefix)."""
    return settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")


def enable_pg_trgm(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    conn.commit()
    print("[pg_trgm] Extension enabled")


def create_trgm_index(conn: psycopg.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_term_display_trgm "
        "ON terminology_codes USING gin (display gin_trgm_ops)"
    )
    conn.commit()
    print("[trgm_index] GIN trigram index created on terminology_codes.display")


def clear_system(conn: psycopg.Connection, system: str) -> int:
    result = conn.execute("DELETE FROM terminology_codes WHERE system = %s", (system,))
    count = result.rowcount
    conn.commit()
    return count


def system_counts(conn: psycopg.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT system, COUNT(*) FROM terminology_codes GROUP BY system").fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_icd10(conn: psycopg.Connection, batch_size: int = 2000, path: Path | None = None) -> int:
    """Load ICD-10-CM codes from the parsed CSV into terminology_codes."""
    path = path or find_icd10_csv()
    if path is None or not path.exists():
        print(f"[icd10] ERROR: no icd10cm_*.csv under {ontology_dir()}")
        return 0

    deleted = clear_system(conn, "icd10cm")
    if deleted:
        print(f"[icd10] Cleared {deleted} existing rows")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"]
            description = row["description"]
            is_header = row.get("is_header", "0") == "0"  # 0 = header in CMS format
            billable = not is_header
            rows.append((
                "icd10cm", code, description, True,
                f'{{"billable": {str(billable).lower()}, "is_header": {str(is_header).lower()}}}',
            ))

    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO terminology_codes (system, code, display, is_active, properties) "
                "VALUES (%s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (system, code) DO UPDATE SET "
                "display = EXCLUDED.display, is_active = EXCLUDED.is_active, "
                "properties = EXCLUDED.properties",
                batch,
            )
        conn.commit()
        inserted += len(batch)
        if (i // batch_size) % 10 == 0:
            print(f"[icd10] Inserted {inserted}/{len(rows)}...")

    print(f"[icd10] Done: {inserted} codes loaded from {path.name}")
    return inserted


def load_snomed(conn: psycopg.Connection, batch_size: int = 5000,
                files: tuple[Path, Path] | None = None) -> int:
    """Load SNOMED CT concepts into terminology_codes.

    1. Active concept IDs from the Concept snapshot
    2. Descriptions: preferred term = FSN without semantic tag, else first synonym
    3. Insert with the preferred term as display
    """
    files = files or find_snomed_files()
    if files is None:
        print(f"[snomed] ERROR: no sct2_Concept_Snapshot*.txt with a matching "
              f"sct2_Description_Snapshot-en*.txt under {ontology_dir()}")
        return 0
    concept_file, desc_file = files

    deleted = clear_system(conn, "snomedct")
    if deleted:
        print(f"[snomed] Cleared {deleted} existing rows")

    print(f"[snomed] Pass 1: Loading concept IDs from {concept_file.name}...")
    active_concepts: set[str] = set()
    total_concepts = 0
    with open(concept_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_concepts += 1
            if row["active"] == "1":
                active_concepts.add(row["id"])
    print(f"[snomed] {len(active_concepts)} active / {total_concepts} total concepts")

    print(f"[snomed] Pass 2: Loading descriptions from {desc_file.name}...")
    fsn_map: dict[str, str] = {}
    synonym_map: dict[str, str] = {}
    desc_count = 0
    with open(desc_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["active"] != "1":
                continue
            cid = row["conceptId"]
            if cid not in active_concepts:
                continue
            desc_count += 1
            term = row["term"]
            type_id = row["typeId"]
            if type_id == FSN_TYPE:
                fsn_map[cid] = _SEMANTIC_TAG_RE.sub("", term).strip()
            elif type_id == SYNONYM_TYPE:
                if cid not in synonym_map:
                    synonym_map[cid] = term
    print(f"[snomed] {desc_count} active descriptions, {len(fsn_map)} FSN terms, {len(synonym_map)} synonyms")

    rows = []
    for cid in active_concepts:
        display = fsn_map.get(cid) or synonym_map.get(cid)
        if not display:
            continue
        rows.append(("snomedct", cid, display, True))
    print(f"[snomed] {len(rows)} concepts with display terms (of {len(active_concepts)} active)")

    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO terminology_codes (system, code, display, is_active) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (system, code) DO UPDATE SET "
                "display = EXCLUDED.display, is_active = EXCLUDED.is_active",
                batch,
            )
        conn.commit()
        inserted += len(batch)
        if (i // batch_size) % 10 == 0:
            print(f"[snomed] Inserted {inserted}/{len(rows)}...")

    print(f"[snomed] Done: {inserted} concepts loaded")
    return inserted


def load_loinc(conn: psycopg.Connection, batch_size: int = 2000, path: Path | None = None) -> int:
    """Load active LOINC codes from Loinc.csv with their properties as JSONB."""
    path = path or find_loinc_csv()
    if path is None or not path.exists():
        print(f"[loinc] ERROR: no Loinc.csv under {ontology_dir()}")
        return 0

    deleted = clear_system(conn, "loinc")
    if deleted:
        print(f"[loinc] Cleared {deleted} existing rows")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("STATUS") != "ACTIVE":
                continue
            loinc_num = row["LOINC_NUM"]
            display = row.get("LONG_COMMON_NAME", "")
            if not display:
                continue
            properties = json.dumps({
                "component": row.get("COMPONENT", ""),
                "property": row.get("PROPERTY", ""),
                "system": row.get("SYSTEM", ""),
                "scale": row.get("SCALE_TYP", ""),
                "class": row.get("CLASS", ""),
                "units": row.get("EXAMPLE_UCUM_UNITS", ""),
            })
            rows.append(("loinc", loinc_num, display, True, properties))
    print(f"[loinc] {len(rows)} active codes parsed from {path}")

    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO terminology_codes (system, code, display, is_active, properties) "
                "VALUES (%s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (system, code) DO UPDATE SET "
                "display = EXCLUDED.display, is_active = EXCLUDED.is_active, "
                "properties = EXCLUDED.properties",
                batch,
            )
        conn.commit()
        inserted += len(batch)
        if (i // batch_size) % 10 == 0:
            print(f"[loinc] Inserted {inserted}/{len(rows)}...")

    print(f"[loinc] Done: {inserted} codes loaded")
    return inserted


# ---------------------------------------------------------------------------
# Auto mode: load whatever is mounted and not yet in the table
# ---------------------------------------------------------------------------

def auto_load(conn: psycopg.Connection, batch_size: int = 5000, force: bool = False,
              download_icd10_if_missing: bool = False) -> dict[str, str]:
    """Load each system whose files are discoverable and whose rows are absent.

    Returns {system: 'loaded N' | 'present N' | 'no files'}.
    """
    root = ontology_dir()
    found = discover(root)
    counts = system_counts(conn)
    status: dict[str, str] = {}

    # ICD-10-CM: parsed CSV, else build from a CMS release on disk, else download.
    icd_csv = found["icd10cm"]
    if icd_csv is None and (counts.get("icd10cm", 0) == 0 or force):
        src = found["icd10cm_source"]
        try:
            if src is not None:
                icd_csv = build_icd10_csv_from_cms(src, root / ICD10_CSV_NAME)
            elif download_icd10_if_missing:
                icd_csv = download_icd10(root / ICD10_CSV_NAME)
        except Exception as exc:  # noqa: BLE001 — never fail boot over an optional download
            print(f"[icd10] could not obtain the CMS release: {exc}")
            icd_csv = None

    plan = [
        ("icd10cm", icd_csv, lambda: load_icd10(conn, batch_size, icd_csv)),
        ("snomedct", found["snomedct"], lambda: load_snomed(conn, batch_size, found["snomedct"])),
        ("loinc", found["loinc"], lambda: load_loinc(conn, batch_size, found["loinc"])),
    ]
    any_loaded = False
    for system, files, loader in plan:
        n = counts.get(system, 0)
        if files is None:
            status[system] = f"present {n}" if n else "no files"
            continue
        if n and not force:
            status[system] = f"present {n}"
            continue
        if not any_loaded:
            enable_pg_trgm(conn)
        status[system] = f"loaded {loader()}"
        any_loaded = True
    if any_loaded:
        create_trgm_index(conn)
    return status


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(conn: psycopg.Connection) -> None:
    print("\n=== Verification ===")
    result = conn.execute(
        "SELECT system, COUNT(*), SUM(CASE WHEN is_active THEN 1 ELSE 0 END) "
        "FROM terminology_codes GROUP BY system ORDER BY system"
    )
    for row in result.fetchall():
        print(f"  {row[0]}: {row[1]} total, {row[2]} active")
    total = conn.execute("SELECT COUNT(*) FROM terminology_codes").fetchone()[0]
    print(f"  TOTAL: {total}")

    print("\n=== Sample Lookups ===")
    for label, system, code in (("ICD-10 I21.01", "icd10cm", "I21.01"),
                                ("SNOMED 22298006", "snomedct", "22298006"),
                                ("LOINC 2160-0", "loinc", "2160-0")):
        row = conn.execute(
            "SELECT code, display, properties FROM terminology_codes WHERE system = %s AND code = %s LIMIT 1",
            (system, code),
        ).fetchone()
        if row:
            print(f"  {label}: {row[1]}" + (f" ({row[2]})" if row[2] else ""))

    try:
        rows = conn.execute(
            "SELECT code, display, similarity(display, %(q)s) AS sim FROM terminology_codes "
            "WHERE system = 'icd10cm' AND display %% %(q)s ORDER BY sim DESC LIMIT 3",
            {"q": "myocardial infarction"},
        ).fetchall()
        if rows:
            print("\n  Trigram search 'myocardial infarction' (ICD-10):")
            for r in rows:
                print(f"    {r[0]}: {r[1]} (similarity={r[2]:.3f})")
    except Exception as e:  # noqa: BLE001
        print(f"  Trigram search not available: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load terminology into PostgreSQL")
    parser.add_argument("--auto", action="store_true",
                        help="Load every system whose files are present and whose rows are missing")
    parser.add_argument("--download-icd10", action="store_true",
                        help="With --auto: fetch the public-domain CMS ICD-10-CM release if no file is present")
    parser.add_argument("--icd10", action="store_true", help="Load ICD-10-CM codes")
    parser.add_argument("--snomed", action="store_true", help="Load SNOMED CT concepts")
    parser.add_argument("--loinc", action="store_true", help="Load LOINC codes")
    parser.add_argument("--all", action="store_true", help="Load all terminology")
    parser.add_argument("--force", action="store_true", help="With --auto: reload systems that are already present")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch insert size")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    parser.add_argument("--list", action="store_true", help="Show which ontology files were found and exit")
    args = parser.parse_args()

    if args.list:
        for k, v in discover().items():
            print(f"  {k:15s} {v if v else '-'}")
        return

    if not any((args.auto, args.icd10, args.snomed, args.loinc, args.all, args.verify_only)):
        parser.print_help()
        return

    url = get_sync_url()
    print(f"Connecting to: {url.split('@')[1] if '@' in url else url}")
    print(f"Ontology directory: {ontology_dir()}")

    with psycopg.connect(url) as conn:
        start = time.time()

        if args.auto:
            status = auto_load(conn, args.batch_size, force=args.force,
                               download_icd10_if_missing=args.download_icd10)
            for system, st in status.items():
                print(f"[auto] {system:9s} {st}")
        elif not args.verify_only:
            enable_pg_trgm(conn)
            if args.icd10 or args.all:
                t0 = time.time()
                n = load_icd10(conn, batch_size=args.batch_size)
                print(f"[icd10] Completed in {time.time() - t0:.1f}s ({n} rows)")
            if args.snomed or args.all:
                t0 = time.time()
                n = load_snomed(conn, batch_size=args.batch_size)
                print(f"[snomed] Completed in {time.time() - t0:.1f}s ({n} rows)")
            if args.loinc or args.all:
                t0 = time.time()
                n = load_loinc(conn, batch_size=args.batch_size)
                print(f"[loinc] Completed in {time.time() - t0:.1f}s ({n} rows)")
            create_trgm_index(conn)

        verify(conn)
        print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
