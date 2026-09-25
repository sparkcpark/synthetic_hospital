"""Partition the 1,068 test-split patients into an RL training pool and a private held-out set.

Stratified by (dominant ICD-10 chapter of the patient's keyed diagnoses, encounter-count bucket), proportional allocation, fixed seed.
The 200 public patients are untouched (public benchmark). Default is a dry run that writes membership to data/rl_split.json and prints
the disease-overlap report; --apply relabels benchmark_ground_truth.split as 'train' / 'heldout' (the 200 public patients stay 'public').
"""
import argparse, json, random, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from eval.config import get_pg_connection
ap = argparse.ArgumentParser(); ap.add_argument("--heldout", type=int, default=268); ap.add_argument("--seed", type=int, default=20260922); ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
conn = get_pg_connection(); cur = conn.cursor()
cur.execute("select distinct patient_id,split::text from benchmark_ground_truth where patient_id is not null"); SPLIT = dict(cur.fetchall())
cur.execute("""select l.patient_id, d.diagnosis_id, d.icd10_code from longitudinal_encounters l
               join question_diagnoses qd on qd.question_id = any(array(select jsonb_array_elements_text(l.source_question_ids::jsonb)::int)) and qd.role='correct'
               join diagnoses d on d.diagnosis_id=qd.diagnosis_id""")
DX = collections.defaultdict(set); CH = collections.defaultdict(collections.Counter)
for pid, did, code in cur.fetchall():
    DX[pid].add(did); CH[pid][(code or "?")[0]] += 1
cur.execute("select patient_id,count(*) from longitudinal_encounters group by 1"); NENC = dict(cur.fetchall())
pool = sorted(p for p, s in SPLIT.items() if s in ("test", "train", "heldout"))
def bucket(n): return "2-3" if n <= 3 else "4-5" if n <= 5 else "6+"
strata = collections.defaultdict(list)
for p in pool: strata[(CH[p].most_common(1)[0][0] if CH[p] else "?", bucket(NENC[p]))].append(p)
rng = random.Random(a.seed); heldout = set(); quota = a.heldout / len(pool); carry = 0.0
for key in sorted(strata):
    ps = sorted(strata[key]); rng.shuffle(ps); want = len(ps) * quota + carry; k = int(round(want)); carry = want - k
    heldout.update(ps[:k])
train = [p for p in pool if p not in heldout]; heldout = sorted(heldout)
val = sorted(p for p, s in SPLIT.items() if s == "public")
def dxset(ps): return set().union(*(DX[p] for p in ps))
Dtr, Dho, Dva = dxset(train), dxset(heldout), dxset(val)
print(f"pool={len(pool)}  train={len(train)}  heldout={len(heldout)}  val(public)={len(val)}  seed={a.seed}")
print(f"strata: {len(strata)} (chapter x encounter bucket); largest={max(len(v) for v in strata.values())}")
print("\nencounter-count buckets   train / heldout / val:")
for b in ("2-3", "4-5", "6+"): print(f"   {b:4s} {sum(bucket(NENC[p])==b for p in train)/len(train):6.1%} {sum(bucket(NENC[p])==b for p in heldout)/len(heldout):8.1%} {sum(bucket(NENC[p])==b for p in val)/len(val):8.1%}")
print("\ndominant ICD-10 chapter      train / heldout / val:")
allch = collections.Counter(CH[p].most_common(1)[0][0] for p in pool)
for c, _ in allch.most_common(10): print(f"   {c}   {sum(CH[p].most_common(1)[0][0]==c for p in train)/len(train):6.1%} {sum(CH[p].most_common(1)[0][0]==c for p in heldout)/len(heldout):8.1%} {sum(CH[p].most_common(1)[0][0]==c for p in val)/len(val):8.1%}")
print("\nDISEASE OVERLAP (keyed diagnoses, diagnosis_id level)")
print(f"   train: {len(Dtr)} distinct diagnoses | heldout: {len(Dho)} | val: {len(Dva)}")
print(f"   heldout diagnoses also keyed in train: {len(Dho & Dtr)} / {len(Dho)} = {len(Dho & Dtr)/len(Dho):.1%}   -> unseen-disease share of heldout = {1-len(Dho & Dtr)/len(Dho):.1%}")
print(f"   val diagnoses also keyed in train:     {len(Dva & Dtr)} / {len(Dva)} = {len(Dva & Dtr)/len(Dva):.1%}   -> unseen-disease share of val     = {1-len(Dva & Dtr)/len(Dva):.1%}")
enc_seen = lambda ps: sum(1 for p in ps for d in DX[p] if d in Dtr) / sum(len(DX[p]) for p in ps)
print(f"   at the encounter level (each keyed diagnosis instance): heldout seen-in-train = {enc_seen(heldout):.1%}, val seen-in-train = {enc_seen(val):.1%}")
pat_all_seen = lambda ps: sum(1 for p in ps if DX[p] <= Dtr) / len(ps); pat_none = lambda ps: sum(1 for p in ps if not (DX[p] & Dtr)) / len(ps)
print(f"   heldout patients with ALL diagnoses seen in train: {pat_all_seen(heldout):.1%}; with NONE seen: {pat_none(heldout):.1%}")
print(f"   val     patients with ALL diagnoses seen in train: {pat_all_seen(val):.1%}; with NONE seen: {pat_none(val):.1%}")
out = {"seed": a.seed, "train": train, "heldout": heldout, "val_public": val,
       "disease_overlap": {"heldout_seen_in_train": len(Dho & Dtr) / len(Dho), "val_seen_in_train": len(Dva & Dtr) / len(Dva)}}
json.dump(out, open(ROOT / "data/rl_split.json", "w"), indent=1); print("\nwrote data/rl_split.json")
if a.apply:
    cur.execute("update benchmark_ground_truth set split='heldout' where patient_id=any(%s)", (heldout,))
    cur.execute("update benchmark_ground_truth set split='train' where patient_id=any(%s)", (train,)); conn.commit(); print("applied to benchmark_ground_truth.split")
