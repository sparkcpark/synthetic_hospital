"""NOTE (release): this script reproduces paper tables and reads the physician-study export
(gui/clinical-eval-platform/blob_pull_content.json) and the paper's run ids, neither of which is
part of the release. The reusable metric code lives in eval/ (eval/imaging_concepts.py, eval/scoring.py).
"""
"""Ontology-grounded concept F1 for the imaging 'inferred clinical question', usable on the full benchmark.

A question (and the reference question) is reduced to the set of clinical concepts it mentions; precision, recall and F1 are taken on those sets.
Concept inventory = every diagnosis and clinical finding in the Synthetic Hospital knowledge graph (display name + SNOMED description share one
concept id) + a small curated lay-synonym layer (so 'kidney stone', 'IUP', 'SBO' resolve). Matching is abbreviation-aware, order-free and
plural-folded; misspelt tokens are snapped to the nearest vocabulary token. The identical extractor is applied to physicians, models and references.
"""
import json, re, sys, difflib, importlib.util
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from eval.config import get_pg_connection
from eval import scoring as S

# The extractor lives in eval/imaging_concepts.py (released); this script only reproduces the paper tables.
from eval.imaging_concepts import ConceptExtractor
cur = get_pg_connection().cursor()
_EX = ConceptExtractor.from_db(cur.connection)
FORMS, ABSORB = _EX.forms, _EX.absorb
def concepts(text): return _EX.concepts(text)
def prf(q, ref): return _EX.prf(q, ref)

if __name__ == "__main__":
    IDS = [2610, 2850, 1726, 2046, 2834, 2285, 1853, 2631, 2323, 1969, 1741, 2549, 2376]
    RUNS = {"Gemini 3.1": 225, "GPT 5.3": 191, "Kimi 2.5-thinking (1T-A32B)": 238, "Opus 4.6": 228, "DeepSeek V3.2 (671B-A37B)": 198, "GLM 5 (744B-A40B)": 216,
            "Qwen 3.5 (397B-A17B)": 209, "Mistral Large (123B)": 194, "Llama 4 Scout (109B-A17B)": 206, "Gemma 3 (27B)": 203}   # the runs behind Table 3's Ques. F1 column
    print(f"inventory: {len(FORMS)} surface forms, {len(set(FORMS.values()))} concepts, {len(ABSORB)} folded into curated synonyms")
    cur.execute("select gt_id,patient_id,ground_truth->>'inferred_clinical_question' from benchmark_ground_truth where task::text='imaging_indication'")
    REF = {g: (pid, q) for g, pid, q in cur.fetchall()}
    res = {}
    for name, rid in RUNS.items():
        cur.execute("select gt_id,prediction,(select (metrics->>'clinical_question_f1')::float from evaluation_runs where run_id=%s) from evaluation_predictions where run_id=%s", (rid, rid))
        rows = cur.fetchall(); full, sub = [], []
        for gid, p, stored in rows:
            p = p if isinstance(p, dict) else json.loads(p); q = p.get("clinical_question") or ""
            x = prf(q, REF[gid][1]) + (S.clinical_question_f1([q], [REF[gid][1]]),); full.append(x)
            if REF[gid][0] in IDS: sub.append((gid, x))
        a = np.array(full); res[name] = dict(n=len(full), f1=a[:, 2].mean(), p=a[:, 0].mean(), r=a[:, 1].mean(), k=a[:, 3].mean(), tok=a[:, 5].mean(), stored=stored, sub=dict(sub))
    # physicians, 13 patients
    cur.execute("select o.order_id,o.gt_id from imaging_orders o where o.gt_id is not null"); O2G = {str(o): g for o, g in cur.fetchall()}
    phys = defaultdict(list)
    for s in [x["content"] for x in json.load(open(ROOT / "gui/clinical-eval-platform/blob_pull_content.json"))["submissions"]]:
        if int(s["physicianId"]) in (3, 17) or not str(s["patientId"]).isdigit() or int(s["patientId"]) not in IDS: continue
        for oid, a in (s["annotations"].get("imagingOrders") or {}).items():
            q = (a.get("question") or "").strip(); gid = O2G.get(str(oid))
            if q and gid in REF: phys[int(s["physicianId"])].append((gid, prf(q, REF[gid][1])))
    allp = [x for v in phys.values() for x in v]; cnt = Counter(g for g, _ in allp)
    pa = np.array([x for _, x in allp]); per = {k: float(np.mean([x[2] for _, x in v])) for k, v in phys.items()}
    print(f"\nPHYSICIANS (13 patients, {len(allp)} answers): concept F1={pa[:,2].mean():.3f}  P={pa[:,0].mean():.3f}  R={pa[:,1].mean():.3f}  concepts/question={pa[:,3].mean():.1f}  ref concepts={pa[:,4].mean():.1f}"
          f"  range across physicians {min(per.values()):.2f}-{max(per.values()):.2f}")
    print(f"\n{'model':30s}{'n':>5s}{'conceptF1':>11s}{'P':>7s}{'R':>7s}{'#conc':>7s}{'tokF1':>8s}{'stored':>8s} | {'13-pt matched concept F1':>26s}")
    for name, d in res.items():
        w = [(d["sub"][g][2], cnt[g]) for g in d["sub"] if g in cnt]; m = float(np.average([a for a, _ in w], weights=[b for _, b in w]))
        d["matched"] = m; print(f"{name:30s}{d['n']:5d}{d['f1']:11.3f}{d['p']:7.3f}{d['r']:7.3f}{d['k']:7.1f}{d['tok']:8.3f}{d['stored']:8.3f} | {m:26.3f}")
    json.dump({"physicians": {"f1": float(pa[:, 2].mean()), "p": float(pa[:, 0].mean()), "r": float(pa[:, 1].mean()), "lo": min(per.values()), "hi": max(per.values()), "n": len(allp)},
               "models": {k: {x: (float(v[x]) if x != "sub" else None) for x in v if x != "sub"} for k, v in res.items()}}, open(ROOT / "data/trial1/imaging_concept_f1_full.json", "w"), indent=1)
    # a few extractions for eyeballing
    for gid in list(REF)[:3]: print("\nREF:", REF[gid][1][:150], "\n  ->", sorted(concepts(REF[gid][1])))
