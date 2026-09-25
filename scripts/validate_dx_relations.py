"""Spot-check the Phase B diagnosis_relations graph (scratch) against canonical names."""
import sqlite3
import sys

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "data/diagnosis_relations_scratch.db"
CANON = sys.argv[2] if len(sys.argv) > 2 else "data/benchmark_v1.2.db"

sc = sqlite3.connect(f"file:{SCRATCH}?mode=ro", uri=True)
ca = sqlite3.connect(f"file:{CANON}?mode=ro", uri=True)

_name_cache = {}
def name(dx):
    if dx not in _name_cache:
        r = ca.execute("SELECT display_name, icd10_code FROM diagnoses WHERE diagnosis_id=?", (dx,)).fetchone()
        _name_cache[dx] = f"{r[0]} [{r[1]}]" if r else str(dx)
    return _name_cache[dx]

def related(dx, limit=14):
    rows = sc.execute(
        "SELECT dx_b, edge_type, weight, mediated_via FROM diagnosis_relations WHERE dx_a=? "
        "UNION SELECT dx_a, edge_type, weight, mediated_via FROM diagnosis_relations WHERE dx_b=?",
        (dx, dx),
    ).fetchall()
    rows.sort(key=lambda r: -r[2])
    return rows[:limit]

print("=== edges by type / source ===")
for et, src, c in sc.execute("SELECT edge_type, source, COUNT(*) AS c FROM diagnosis_relations GROUP BY edge_type, source ORDER BY c DESC"):
    print(f"  {et:30s} {src:14s} {c}")
print("  total:", sc.execute("SELECT COUNT(*) FROM diagnosis_relations").fetchone()[0])
print("  distinct diagnoses with >=1 edge:",
      sc.execute("SELECT COUNT(*) FROM (SELECT dx_a FROM diagnosis_relations UNION SELECT dx_b FROM diagnosis_relations)").fetchone()[0])

def show(label, where, params):
    print(f"\n=== {label} ===")
    ids = [r[0] for r in ca.execute(f"SELECT diagnosis_id FROM diagnoses WHERE {where}", params)]
    for d in ids[:3]:
        rel = related(d)
        print(f"{name(d)}  ->  {len(rel)} related")
        for dx_b, et, w, med in rel:
            tag = f" via {med}" if med else " (direct)"
            print(f"     {name(dx_b):55s} {et} w={w:.2f}{tag}")

show("T2DM (SNOMED 44054006)", "snomed_id = ?", ("44054006",))
show("Hypertension (ICD-10 I10)", "icd10_code = ?", ("I10",))
