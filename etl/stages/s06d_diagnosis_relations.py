"""Stage 6d: diagnosis<->diagnosis relatedness graph (Phase B, REVISED 2026-06-07).

Typed-edge relevance rule (phaseB_typed_edge_relevance_rule.md). Replaces the
scalar `base_type x 0.85_mediated x breadth_factor` weight + global tau, which
conflated SNOMED path-specificity with clinical relevance (HTN<->CKD at 0.34 sat
with broad-hub noise). Tier membership is now keyed on the relationship TYPE:

  Class 1 (definitional/etiologic): SNOMED `Due to` (direct or recovered through a
      precoordinated combination concept). [+ ICD combination codes / Code-first /
      Use-additional -> open item, needs the CMS Tabular XML.] -> relevant by
      construction (never thresholded).
  Class 2 (associative/co-localized): SNOMED `Associated with`, `After`,
      `Pathological process`; shared `Finding site` at organ granularity or finer.
      -> relevant.
  Class 3 (untyped residual): shared-finding overlap only, for pairs with NO
      Class 1/2 edge. -> relevant iff overlap >= tau_residual (applied in Phase C).

Subsumption (a relationship on a general concept applies to its subtypes) is kept,
with a discrete MAX_SUBSUME hard-drop of only the top-tier concepts. The continuous
breadth_factor is RETIRED. Deterministic; no LLM.
"""

import argparse
import csv
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

from etl.ontology.snomed import RELATIONSHIP_FILE, SNOMED_RELEASE  # noqa: E402
from etl.utils.logging import get_logger  # noqa: E402

log = get_logger("etl.stages.s06d")

TC_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "ontology"
    / "SnomedCT_ManagedServiceUSTransitiveClosure_PRODUCTION_US1000124_20250901T120000Z"
    / "Resources" / "TransitiveClosure"
    / "res2_TransitiveClosure_US1000124_20250901.txt"
)

# SNOMED attribute typeIds (verified against the loaded release).
ISA = "116680003"
DUE_TO = "42752001"
FINDING_SITE = "363698007"
# typeId -> (relation_class, relation_type)
TYPED = {
    DUE_TO: ("1_definitional", "due_to"),
    "47429007": ("2_associative", "associated_with"),
    "255234002": ("2_associative", "after"),
    "370135005": ("2_associative", "pathological_process"),
}

# Drop only top-tier concepts (Disease/Clinical-finding subsume 800-6800); keep
# clinical hubs. Discrete hard cap, NOT a continuous penalty.
MAX_SUBSUME = 100
# A shared finding-site is Class 2 only at "organ granularity or finer". The
# granularity is set by a clinician-graded allowlist (open item 2, option b:
# finding_site_review.csv, GRADE_organ_or_system=='organ'). FINDING_SITE_MAX_DX
# is the fallback count guard, used only if the graded file is absent.
FINDING_SITE_MAX_DX = 30
FINDING_SITE_GRADES = Path(__file__).resolve().parent.parent.parent / "finding_site_review.csv"
# Clinician-curated edges closing comorbidity-coverage gaps (open item 3). Rows with
# blank FILL_* (the C3/sparse "DECIDE" group) are intentionally skipped.
CURATED_EDGES = Path(__file__).resolve().parent.parent.parent / "curated_diagnosis_edges.csv"

# Class-3 residual (shared-finding overlap) — stored with overlap_score; Phase C
# applies tau_residual. Edge-strength weights and the non-specific-finding guard.
_DF_REL_WEIGHT = {"pathognomonic": 1.0, "highly_suggestive": 0.8, "commonly_seen": 0.5,
                  "risk_factor": 0.3, "protective": 0.2, "rules_out": 0.1}
SHARED_FINDING_MAX_DF = 40       # findings in >this many dx are too non-specific
SHARED_FINDING_STORE_MIN = 0.20  # don't store trivially-small overlaps

_DDL = """
DROP TABLE IF EXISTS diagnosis_relations;
CREATE TABLE IF NOT EXISTS diagnosis_relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dx_a            INTEGER NOT NULL,
    dx_b            INTEGER NOT NULL,
    relation_class  TEXT NOT NULL,   -- 1_definitional | 2_associative | 3_residual
    relation_type   TEXT NOT NULL,   -- due_to | associated_with | after |
                                     -- pathological_process | shared_finding_site | shared_findings
    direction       TEXT NOT NULL,   -- a_to_b | b_to_a | symmetric
    overlap_score   REAL,            -- class 3 only
    source          TEXT NOT NULL,   -- snomed | icd10cm | derived
    mediated_via    TEXT,
    frozen_version  TEXT,
    UNIQUE(dx_a, dx_b, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_dr_dx_a ON diagnosis_relations(dx_a);
CREATE INDEX IF NOT EXISTS idx_dr_dx_b ON diagnosis_relations(dx_b);
CREATE INDEX IF NOT EXISTS idx_dr_class ON diagnosis_relations(relation_class);
"""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_diagnoses(conn):
    rows = conn.execute("SELECT diagnosis_id, icd10_code, snomed_id FROM diagnoses").fetchall()
    snomed_to_dx = defaultdict(list)
    for dx_id, _icd, snomed in rows:
        if snomed:
            snomed_to_dx[str(snomed)].append(dx_id)
    log.info("Loaded %d diagnoses (%d with SNOMED id)", len(rows), len(snomed_to_dx))
    return rows, snomed_to_dx


def _load_snomed_graph(rel_file=RELATIONSHIP_FILE):
    """One pass over RF2 Relationship: typed out-edges, is-a parents, finding sites."""
    out_typed = defaultdict(list)        # src -> [(typeId, dest)]
    isa_parents = defaultdict(set)       # child -> {parents}
    finding_sites = defaultdict(set)     # concept -> {body structures}
    with open(rel_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["active"] != "1":
                continue
            t = row["typeId"]; src = row["sourceId"]; dest = row["destinationId"]
            if t == ISA:
                isa_parents[src].add(dest)
            elif t in TYPED:
                out_typed[src].append((t, dest))
            elif t == FINDING_SITE:
                finding_sites[src].add(dest)
    log.info("SNOMED graph: %d concepts w/ typed out-edges, %d w/ is-a, %d w/ finding-site",
             len(out_typed), len(isa_parents), len(finding_sites))
    return out_typed, isa_parents, finding_sites


def _load_ancestors(concepts, tc_file=TC_FILE):
    ancestors = defaultdict(set)
    with open(tc_file, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            sup, _, sub = line.partition("\t")
            sub = sub.strip()
            if sub in concepts:
                ancestors[sub].add(sup)
    return ancestors


def _load_finding_site_allowlist(path=FINDING_SITE_GRADES):
    """Clinician-graded organ-granularity finding-sites (open item 2). Returns the
    set of site IDs graded 'organ', or None if the graded file is absent (the
    builder then falls back to the FINDING_SITE_MAX_DX count guard)."""
    if not path.exists():
        return None
    allow = set()
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r.get("GRADE_organ_or_system") or "").strip().lower() == "organ":
                allow.add(r["finding_site_id"])
    return allow


# ---------------------------------------------------------------------------
# Edge accumulation (best class per (a,b,relation_type))
# ---------------------------------------------------------------------------

_CLASS_RANK = {"1_definitional": 1, "2_associative": 2, "3_residual": 3}


class _Edges:
    def __init__(self):
        self.e = {}                 # (a,b,rtype) -> dict row
        self.typed_pairs = set()    # {frozenset({a,b})} that have a Class 1/2 edge

    def add(self, a, b, rclass, rtype, direction, source, mediated=None, overlap=None):
        if a == b:
            return
        key = (a, b, rtype)
        prev = self.e.get(key)
        if prev is None or _CLASS_RANK[rclass] < _CLASS_RANK[prev["relation_class"]]:
            self.e[key] = {"dx_a": a, "dx_b": b, "relation_class": rclass,
                           "relation_type": rtype, "direction": direction,
                           "overlap_score": overlap, "source": source, "mediated_via": mediated}
        if rclass != "3_residual":
            self.typed_pairs.add(frozenset((a, b)))

    def has_typed(self, a, b):
        return frozenset((a, b)) in self.typed_pairs

    def rows(self):
        return list(self.e.values())


# ---------------------------------------------------------------------------
# Class 1 + Class 2: SNOMED typed edges (direct + combination-concept mediated)
# ---------------------------------------------------------------------------

def _add_snomed_typed(edges, dx_rows, snomed_to_dx, out_typed, isa_parents, finding_sites):
    present = set(snomed_to_dx)

    # subsumption: present diagnoses reachable from a (possibly general) concept
    anc_present = _load_ancestors(present)
    subsumes = defaultdict(set)
    for s in present:
        subsumes[s].update(snomed_to_dx[s])
        for a in anc_present.get(s, ()):
            subsumes[a].update(snomed_to_dx[s])
    subsumes = {c: d for c, d in subsumes.items() if len(d) <= MAX_SUBSUME}

    def resolve(concept):
        return subsumes.get(concept, set())

    # DIRECT typed edges: present dx --typed--> concept subsuming present dx
    direct = 0
    for dx_id, _icd, snomed in dx_rows:
        if not snomed:
            continue
        for t, dest in out_typed.get(str(snomed), ()):
            rclass, rtype = TYPED[t]
            for dy in resolve(dest):
                edges.add(dx_id, dy, rclass, rtype, "a_to_b", "snomed")
                direct += 1

    # MEDIATED: combination concept C with a typed out-edge resolving to a present
    # dx, whose is-a ancestor is a present dx (the base disease) -> link base<->target.
    candidates = {}
    for c, oedges in out_typed.items():
        res = [(t, dest) for (t, dest) in oedges if dest in subsumes]
        if res:
            candidates[c] = res
    anc_c = _load_ancestors(set(candidates))
    mediated = 0
    for c, res in candidates.items():
        bases = set()
        for a in anc_c.get(c, ()):
            if a in present:
                bases.update(snomed_to_dx[a])
        if not bases:
            continue
        for t, dest in res:
            rclass, rtype = TYPED[t]
            for tgt in resolve(dest):
                for b in bases:
                    edges.add(b, tgt, rclass, rtype, "a_to_b", "snomed", mediated=c)
                    mediated += 1
    log.info("SNOMED typed edges: %d direct, %d mediated", direct, mediated)

    # Class 2: shared Finding site at organ granularity or finer (clinician-graded)
    allow = _load_finding_site_allowlist()
    site_dx = defaultdict(set)
    for dx_id, _icd, snomed in dx_rows:
        if not snomed:
            continue
        for site in finding_sites.get(str(snomed), ()):
            site_dx[site].add(dx_id)
    fs = 0
    for site, dxs in site_dx.items():
        if len(dxs) < 2:
            continue
        if allow is not None:
            if site not in allow:           # not graded organ-granularity -> skip
                continue
        elif len(dxs) > FINDING_SITE_MAX_DX:  # fallback count guard
            continue
        dxs = sorted(dxs)
        for i in range(len(dxs)):
            for j in range(i + 1, len(dxs)):
                edges.add(dxs[i], dxs[j], "2_associative", "shared_finding_site",
                          "symmetric", "snomed")
                fs += 1
    log.info("Finding-site (Class 2) edges: %d (granularity=%s)",
             fs, "graded organ allowlist" if allow is not None else f"count<={FINDING_SITE_MAX_DX}")


# ---------------------------------------------------------------------------
# Class 1/2: ICD-10-CM Tabular instructional notes (codeFirst / useAdditional / codeAlso)
# ---------------------------------------------------------------------------

def _add_icd_notes(edges, dx_rows):
    from etl.ontology.icd10_notes import load_icd_note_edges
    note_edges, ex2 = load_icd_note_edges()
    norm = lambda c: c.replace(".", "").replace("-", "").strip().upper()
    codes = []          # (normalized_code, dx_id)
    by_cat = defaultdict(list)
    for dx_id, icd, _sn in dx_rows:
        if not icd:
            continue
        c = norm(icd)
        codes.append((c, dx_id))
        by_cat[c[:3]].append(dx_id)
    n = 0
    for code, ref, rclass, rtype in note_edges:
        diag_dx = [dx for c, dx in codes if c.startswith(code)]
        if not diag_dx:
            continue
        for a in diag_dx:
            for b in by_cat.get(ref, ()):
                edges.add(a, b, rclass, rtype, "a_to_b", "icd10cm")
                n += 1
    log.info("ICD-note edges: %d (from %d note refs; %d excludes2 hints not admitted)",
             n, len(note_edges), ex2)


# ---------------------------------------------------------------------------
# Class 1/2: clinician-curated edges (comorbidity-coverage gaps, open item 3)
# ---------------------------------------------------------------------------

def _add_curated_edges(edges, dx_rows, path=CURATED_EDGES):
    if not path.exists():
        return
    norm = lambda c: c.replace(".", "").replace("-", "").strip().upper()
    codes = [(norm(icd), dx) for dx, icd, _sn in dx_rows if icd]

    def match(prefix):
        p = norm(prefix)
        return [dx for c, dx in codes if c.startswith(p)]

    n = used = skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            a_icd = (r.get("FILL_a_icd") or "").strip()
            b_icd = (r.get("FILL_b_icd") or "").strip()
            rclass = (r.get("FILL_relation_class") or "").strip()
            rtype = (r.get("FILL_relation_type") or "").strip() or "curated"
            if not (a_icd and b_icd and rclass):   # DECIDE rows left blank -> skip
                skipped += 1
                continue
            if rclass not in _CLASS_RANK:
                log.warning("curated row %s: bad relation_class %r, skipped", r.get("row"), rclass)
                continue
            split = lambda f: f.replace(";", " ").replace(",", " ").split()
            A = {d for p in split(a_icd) for d in match(p)}
            B = {d for p in split(b_icd) for d in match(p)}
            if not A or not B:
                log.warning("curated row %s: %s(%d) <-> %s(%d) unresolved",
                            r.get("row"), a_icd, len(A), b_icd, len(B))
                continue
            used += 1
            for a in A:
                for b in B:
                    edges.add(a, b, rclass, rtype, "a_to_b", "curated")
                    n += 1
    log.info("Curated edges: %d (from %d filled rows; %d DECIDE rows skipped)", n, used, skipped)


# ---------------------------------------------------------------------------
# Class 3: untyped shared-finding residual (only for pairs with no Class 1/2 edge)
# ---------------------------------------------------------------------------

def _add_residual(edges, conn):
    dx_find = defaultdict(dict)
    finding_dxs = defaultdict(list)
    for dx_id, fid, rel in conn.execute(
            "SELECT diagnosis_id, finding_id, relationship FROM diagnosis_findings"):
        w = _DF_REL_WEIGHT.get(rel, 0.3)
        if w > dx_find[dx_id].get(fid, 0.0):
            dx_find[dx_id][fid] = w
        finding_dxs[fid].append(dx_id)
    dx_total = {d: sum(fs.values()) for d, fs in dx_find.items()}

    shared = defaultdict(float)
    for fid, dxs in finding_dxs.items():
        dxs = sorted(set(dxs))
        if len(dxs) < 2 or len(dxs) > SHARED_FINDING_MAX_DF:
            continue
        for i in range(len(dxs)):
            for j in range(i + 1, len(dxs)):
                a, b = dxs[i], dxs[j]
                shared[(a, b)] += min(dx_find[a].get(fid, 0.0), dx_find[b].get(fid, 0.0))
    added = 0
    for (a, b), s in shared.items():
        if edges.has_typed(a, b):
            continue  # typed edge already -> not residual
        denom = dx_total[a] + dx_total[b] - s
        jac = s / denom if denom > 0 else 0.0
        if jac >= SHARED_FINDING_STORE_MIN:
            edges.add(a, b, "3_residual", "shared_findings", "symmetric", "derived", overlap=jac)
            added += 1
    log.info("Class-3 residual (shared-finding) edges: %d", added)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_diagnosis_relations(read_conn, write_conn=None, include_residual=True):
    t0 = time.monotonic()
    write_conn = write_conn or read_conn
    write_conn.executescript(_DDL)

    dx_rows, snomed_to_dx = _load_diagnoses(read_conn)
    out_typed, isa_parents, finding_sites = _load_snomed_graph()
    edges = _Edges()
    _add_snomed_typed(edges, dx_rows, snomed_to_dx, out_typed, isa_parents, finding_sites)
    _add_icd_notes(edges, dx_rows)
    _add_curated_edges(edges, dx_rows)
    if include_residual:
        _add_residual(edges, read_conn)

    rows = edges.rows()
    write_conn.execute("DELETE FROM diagnosis_relations")
    write_conn.executemany(
        "INSERT OR REPLACE INTO diagnosis_relations "
        "(dx_a, dx_b, relation_class, relation_type, direction, overlap_score, source, mediated_via, frozen_version) "
        "VALUES (:dx_a,:dx_b,:relation_class,:relation_type,:direction,:overlap_score,:source,:mediated_via,:fv)",
        [{**r, "fv": SNOMED_RELEASE} for r in rows])
    write_conn.commit()

    by_class = defaultdict(int)
    for r in rows:
        by_class[r["relation_class"]] += 1
    log.info("diagnosis_relations: %d edges in %.1fs | by class: %s",
             len(rows), time.monotonic() - t0, dict(by_class))
    return {"edges": len(rows), "by_class": dict(by_class)}


def main():
    ap = argparse.ArgumentParser(description="Stage 6d: typed diagnosis relatedness graph")
    ap.add_argument("--read", required=True)
    ap.add_argument("--write", default=None)
    ap.add_argument("--no-residual", action="store_true")
    args = ap.parse_args()
    read_conn = sqlite3.connect(f"file:{args.read}?mode=ro", uri=True)
    write_conn = sqlite3.connect(args.write) if args.write else read_conn
    try:
        print(build_diagnosis_relations(read_conn, write_conn, include_residual=not args.no_residual))
    finally:
        read_conn.close()
        if write_conn is not read_conn:
            write_conn.close()


if __name__ == "__main__":
    main()
