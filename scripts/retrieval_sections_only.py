"""NOTE (release): this script reproduces paper tables and reads the physician-study export
(gui/clinical-eval-platform/blob_pull_content.json) and the paper's run ids, neither of which is
part of the release. The reusable metric code lives in eval/ (eval/imaging_concepts.py, eval/scoring.py).
"""
"""Evidence-retrieval metrics recomputed over chart sections only: fact cards are removed from both the judgment pool and each ranking.
Models: the zero-shot runs behind Table 3, full set. Physicians: 7 physicians x 13 patients, ranking = flagged sections ordered by their grade
(ties broken at random, averaged over 200 shuffles), same as the earlier comparison."""
import json, random, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from eval.config import get_pg_connection
from eval import scoring as S
cur = get_pg_connection().cursor()
IDS = [2610, 2850, 1726, 2046, 2834, 2285, 1853, 2631, 2323, 1969, 1741, 2549, 2376]
RUNS = {"Gemini 3.1": 109, "GPT 5.3": 114, "Kimi 2.5-thinking (1T-A32B)": 21, "Opus 4.6": 119, "DeepSeek V3.2 (671B-A37B)": 79, "GLM 5 (744B-A40B)": 104,
        "Qwen 3.5 (397B-A17B)": 94, "Mistral Large (123B)": 99, "Llama 4 Scout (109B-A17B)": 89, "Gemma 3 (27B)": 84}
cur.execute("select gt_id,passage_id,relevance_grade,passage_source::text from relevance_judgments"); J_ALL, J_SEC = defaultdict(dict), defaultdict(dict)
FC = defaultdict(dict)
for g, p, r, s in cur.fetchall():
    J_ALL[g][p] = r
    if s == "encounter_section": J_SEC[g][p] = r
    else: FC[g][p] = s
cur.execute("select gt_id,patient_id from benchmark_ground_truth where task::text='evidence_retrieval'"); G2P = dict(cur.fetchall())
KEYS = ["precision_5", "recall_5", "recall_10", "recall_20", "ndcg_10", "map_10", "mrr"]
def metrics(rankings, judg): return S.compute_all_metrics("evidence_retrieval", [{"rankings": rankings}], [{"_judgments": judg}])
res = {}
for name, rid in RUNS.items():
    cur.execute("select gt_id,prediction from evaluation_predictions where run_id=%s", (rid,)); rows = cur.fetchall(); acc = {"all": defaultdict(list), "sec": defaultdict(list)}; per = {}
    for g, p in rows:
        p = p if isinstance(p, dict) else json.loads(p); rk = p.get("rankings", [])
        fc = {pid for pid, src in FC.get(g, {}).items()}
        rk_sec = [r for r in rk if S._normalize_passage_id(str(r.get("passage_id", "")), set(J_ALL[g])) not in fc]   # drop fact cards from the ranking too
        for k, J, R in (("all", J_ALL, rk), ("sec", J_SEC, rk_sec)):
            m = metrics(R, J[g])
            for kk in KEYS: acc[k][kk].append(m[kk])
            if k == "sec": per[g] = m
    res[name] = {k: {kk: float(np.mean(v[kk])) for kk in KEYS} for k, v in acc.items()}; res[name]["per_item_sec"] = {g: per[g] for g in per}; res[name]["n"] = len(rows)
    print(f"{name:28s} n={len(rows):4d}  ALL: P@5={res[name]['all']['precision_5']:.3f} R@10={res[name]['all']['recall_10']:.3f} NDCG={res[name]['all']['ndcg_10']:.3f} | SECTIONS: P@5={res[name]['sec']['precision_5']:.3f} R@5={res[name]['sec']['recall_5']:.3f} R@10={res[name]['sec']['recall_10']:.3f} R@20={res[name]['sec']['recall_20']:.3f} NDCG@10={res[name]['sec']['ndcg_10']:.3f} MAP={res[name]['sec']['map_10']:.3f}")
# ceiling and chance on sections
cur.execute("select avg(n) from (select gt_id,count(*) n from relevance_judgments where relevance_grade>=2 and passage_source::text='encounter_section' group by gt_id) t"); rel = float(cur.fetchone()[0])
cur.execute("select avg((relevance_grade>=2)::int) from relevance_judgments where passage_source::text='encounter_section'"); chance = float(cur.fetchone()[0])
print(f"\nsections only: mean relevant sections per patient = {rel:.1f} -> R@10 ceiling ~ {min(1,10/rel):.2f}; chance P@5 = {chance:.3f}")
# physicians (13 patients)
cur.execute("""select l.patient_id,e.encounter_id,e.section_type,e.id from encounter_ehr_sections e join longitudinal_encounters l on l.encounter_id=e.encounter_id where l.patient_id=any(%s)""", (IDS,))
SEC = {(pid, str(enc), st): f"ees_{i}" for pid, enc, st, i in cur.fetchall()}; types = sorted({k[2] for k in SEC}, key=len, reverse=True)
P2G = {p: g for g, p in G2P.items()}
rng = random.Random(20260920); phys = []
for s in [x["content"] for x in json.load(open(ROOT / "gui/clinical-eval-platform/blob_pull_content.json"))["submissions"]]:
    if int(s["physicianId"]) in (3, 17) or not str(s["patientId"]).isdigit() or int(s["patientId"]) not in IDS: continue
    pid = int(s["patientId"]); flags = []
    for k, v in (s["annotations"].get("evidenceFlags") or {}).items():
        st = next((t for t in types if k.endswith("_" + t)), None)
        if not st: continue
        enc = k[4:][: -len(st) - 1]; pidp = SEC.get((pid, enc, st)); gr = (v or {}).get("grade")
        if pidp and gr is not None and gr >= 1: flags.append((pidp, gr))
    if not flags: continue
    acc = defaultdict(list)
    for _ in range(200):
        rng.shuffle(flags); rk = [{"passage_id": p} for p, _ in sorted(flags, key=lambda x: -x[1])]
        m = metrics(rk, J_SEC[P2G[pid]])
        for kk in KEYS: acc[kk].append(m[kk])
    phys.append(dict(pid=pid, phys=int(s["physicianId"]), n_flagged=len(flags), **{kk: float(np.mean(v)) for kk, v in acc.items()}))
cnt = Counter(x["pid"] for x in phys); pm = {kk: float(np.mean([x[kk] for x in phys])) for kk in KEYS}
per_phys = {ph: {kk: float(np.mean([x[kk] for x in phys if x["phys"] == ph])) for kk in KEYS} for ph in {x["phys"] for x in phys}}
print(f"\nPHYSICIANS (sections only, {len(phys)} answers): P@5={pm['precision_5']:.3f} R@5={pm['recall_5']:.3f} R@10={pm['recall_10']:.3f} R@20={pm['recall_20']:.3f} NDCG@10={pm['ndcg_10']:.3f} MAP={pm['map_10']:.3f} | mean flagged={np.mean([x['n_flagged'] for x in phys]):.1f}")
for kk in ("precision_5", "recall_10", "ndcg_10"): v = [per_phys[p][kk] for p in per_phys]; print(f"   {kk}: range across physicians {min(v):.2f}-{max(v):.2f}")
print("\n13-patient matched (sections only), weighted by physician answers per patient:")
for name in RUNS:
    it = [(g, m) for g, m in res[name]["per_item_sec"].items() if G2P[g] in cnt]; w = [cnt[G2P[g]] for g, _ in it]
    print(f"   {name:28s} P@5={np.average([m['precision_5'] for _,m in it],weights=w):.3f} R@10={np.average([m['recall_10'] for _,m in it],weights=w):.3f} NDCG@10={np.average([m['ndcg_10'] for _,m in it],weights=w):.3f}")
json.dump({"models": {k: {kk: v[kk] for kk in ("all", "sec", "n")} for k, v in res.items()}, "physicians": pm, "phys_range": {kk: [min(per_phys[p][kk] for p in per_phys), max(per_phys[p][kk] for p in per_phys)] for kk in KEYS},
           "ceiling_r10": min(1, 10 / rel), "chance_p5": chance}, open(ROOT / "data/trial1/retrieval_sections_only.json", "w"), indent=1)
