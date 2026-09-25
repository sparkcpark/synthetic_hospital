"""Phase B coverage check against the pre-registered comorbidity list (open item 3).

For each row of comorbidity_coverage_list.md, resolve Condition A and Condition B
to our diagnoses (clinical vocab + abbreviation expansion + display-name keywords),
look for a diagnosis_relations edge between them, and compare the edge's class to
the row's Expected class. Emits:
  - comorbidity_coverage_report.csv : full per-row result
  - curated_diagnosis_edges_TEMPLATE.csv : the MISS/WEAKER gaps, pre-filled with
    suggested code pairs + class, for the clinician to confirm/edit. The builder
    consumes the filled file (source='curated') to close the gaps.
"""
import csv
import re
import sqlite3
from collections import Counter, defaultdict

DB = "data/benchmark_v1.2_copy.db"
MD = "comorbidity_coverage_list.md"

# whole-word abbreviation expansion applied before matching
ABBREV = {
    "htn": "hypertension", "dm": "diabetes mellitus", "t2dm": "type 2 diabetes",
    "ckd": "chronic kidney disease", "mi": "myocardial infarction", "cad": "coronary artery disease",
    "hf": "heart failure", "chf": "heart failure", "aki": "acute kidney injury",
    "esrd": "end stage renal disease", "pe": "pulmonary embolism", "dvt": "deep vein thrombosis",
    "vte": "venous thromboembolism", "gn": "glomerulonephritis", "sle": "lupus",
    "ra": "rheumatoid arthritis", "copd": "chronic obstructive pulmonary",
    "osa": "obstructive sleep apnea", "gca": "giant cell arteritis",
    "gpa": "granulomatosis with polyangiitis", "avsd": "atrioventricular septal defect",
    "svc": "superior vena cava", "ten": "toxic epidermal necrolysis", "pkd": "polycystic kidney",
    "adpkd": "polycystic kidney", "ms": "multiple sclerosis", "sci": "spinal cord injury",
    "cf": "cystic fibrosis", "cfrd": "cystic fibrosis diabetes", "uti": "urinary tract infection",
    "aion": "ischemic optic neuropathy", "pah": "pulmonary hypertension", "ie": "endocarditis",
    "tia": "transient cerebral ischemic", "psa": "psoriatic arthritis", "sbp": "peritonitis",
    "hcc": "hepatocellular carcinoma", "pcp": "pneumocystis", "siadh": "inappropriate antidiuretic",
    "dka": "ketoacidosis", "hhs": "hyperosmolar", "ibd": "inflammatory bowel",
    "uc": "ulcerative colitis", "nafld": "fatty liver", "nash": "steatohepatitis",
    "cll": "lymphocytic leukemia", "pres": "posterior reversible", "hbv": "hepatitis b",
    "hcv": "hepatitis c", "tls": "tumor lysis", "aps": "antiphospholipid",
    "hiv": "human immunodeficiency virus", "nf1": "neurofibromatosis",
}

# condition substring -> (icd prefixes, name keywords). For compounds the keyword
# is usually enough; ICD prefixes anchor the cross-chapter ones.
VOCAB = {
    "hypertension": (["I10"], ["essential hypertension"]),
    "hypertensive heart disease with chronic kidney": (["I13"], []),
    "hypertensive heart disease": (["I11"], ["hypertensive heart"]),
    "chronic kidney disease": (["N18"], ["chronic kidney disease"]),
    "ischemic stroke": (["I63"], ["cerebral infarction"]),
    "intracerebral hemorrhage": (["I61"], ["intracerebral h"]),
    "heart failure": (["I50"], ["heart failure"]),
    "ischemic cardiomyopathy": (["I25.5"], ["ischemic cardiomyopathy"]),
    "aortic dissection": (["I71"], ["aortic dissection"]),
    "aortic root aneurysm": (["I71"], ["aortic aneurysm", "aortic root"]),
    "coronary artery disease": (["I25"], ["coronary"]),
    "myocardial infarction": (["I21", "I22"], ["myocardial infarction"]),
    "atrial fibrillation": (["I48"], ["atrial fibrillation"]),
    "hyperlipidemia": (["E78"], ["hyperlipidemia", "dyslipidemia"]),
    "rheumatic fever": (["I00", "I01", "I02"], ["rheumatic fever"]),
    "rheumatic valvular heart disease": (["I05", "I06", "I08"], ["rheumatic"]),
    "pulmonary hypertension": (["I27.0", "I27.2"], ["pulmonary hypertension"]),
    "cor pulmonale": (["I27.81", "I27.9"], ["cor pulmonale", "right heart failure"]),
    "endocarditis": (["I33", "I38"], ["endocarditis"]),
    "septic emboli": (["I63", "N05"], ["embolism", "septic embol"]),
    "type 2 diabetes": (["E11"], ["type 2 diabetes"]),
    "diabetes mellitus": (["E08", "E09", "E10", "E11", "E13"], ["diabetes mellitus"]),
    "diabetic chronic kidney": (["E11.2"], []),
    "anemia of chronic kidney": (["D63.1"], []),
    "anemia": (["D50", "D60", "D61", "D62", "D63", "D64"], ["anemia"]),
    "renal osteodystrophy": (["N25.0"], ["osteodystrophy", "mineral and bone"]),
    "secondary hyperparathyroidism": (["N25.81", "E21.1"], ["secondary hyperparathyroid"]),
    "metabolic acidosis": (["E87.2"], ["acidosis"]),
    "cirrhosis": (["K74", "K70.3"], ["cirrhosis"]),
    "hepatorenal": (["K76.7"], ["hepatorenal"]),
    "sepsis": (["A41", "A40", "R65.2"], ["sepsis", "septic"]),
    "acute kidney injury": (["N17"], ["acute kidney injury"]),
    "renal failure": (["N17", "N18", "N19"], ["renal failure", "kidney failure"]),
    "rhabdomyolysis": (["M62.82", "T79.6"], ["rhabdomyolysis"]),
    "myeloma": (["C90"], ["myeloma"]),
    "nephrotic": (["N04"], ["nephrotic"]),
    "venous thromboembolism": (["I82", "I80", "I26"], ["venous thrombo", "deep vein"]),
    "thrombosis": (["I82", "I80", "I26", "I74"], ["thrombosis"]),
    "lupus": (["M32"], ["lupus"]),
    "lupus nephritis": (["M32.14", "M32.15", "N08"], ["lupus nephritis"]),
    "cutaneous lupus": (["L93", "M32"], ["cutaneous lupus", "discoid lupus"]),
    "neuropsychiatric lupus": (["M32.19"], []),
    "gout": (["M10", "M1A"], ["gout"]),
    "gouty": (["M10", "N22"], ["urate", "gouty"]),
    "obstructive uropathy": (["N13"], ["obstructive uropathy", "hydronephrosis"]),
    "hydronephrosis": (["N13.3", "N13.0"], ["hydronephrosis"]),
    "prostatic hyperplasia": (["N40"], ["prostatic hyperplasia", "bladder outlet"]),
    "diabetic retinopathy": (["E11.3", "H35"], ["retinopathy"]),
    "diabetic polyneuropathy": (["E11.4", "G63"], ["neuropathy"]),
    "neuropathy": (["G60", "G62", "G63"], ["neuropathy"]),
    "diabetic foot": (["E11.62", "L97"], ["foot ulcer"]),
    "peripheral arterial": (["I70", "I73.9"], ["peripheral arterial", "peripheral vascular"]),
    "gastroparesis": (["E11.43", "K31.84"], ["gastroparesis"]),
    "ketoacidosis": (["E11.1", "E10.1", "E13.1"], ["ketoacidosis"]),
    "hyperosmolar": (["E11.0"], ["hyperosmolar"]),
    "thyrotoxicosis": (["E05"], ["thyrotoxicosis", "hyperthyroid"]),
    "pheochromocytoma": (["E27.5", "C74.1", "D35.0"], ["pheochromocytoma"]),
    "secondary hypertension": (["I15"], ["secondary hypertension"]),
    "hyperaldosteronism": (["E26.0", "E26"], ["aldosteron"]),
    "hypokalemia": (["E87.6"], ["hypokalemia"]),
    "primary hyperparathyroidism": (["E21.0"], ["primary hyperparathyroid"]),
    "hypercalcemia": (["E83.52"], ["hypercalcemia"]),
    "nephrolithiasis": (["N20"], ["calculus", "nephrolithiasis"]),
    "obesity": (["E66"], ["obesity"]),
    "obstructive sleep apnea": (["G47.3"], ["sleep apnea"]),
    "cushing": (["E24"], ["cushing"]),
    "portal hypertension": (["K76.6"], ["portal hypertension"]),
    "esophageal varices": (["I85"], ["varices"]),
    "varices": (["I85"], ["varices"]),
    "hepatic encephalopathy": (["K72"], ["hepatic encephalopathy"]),
    "ascites": (["R18", "K70.31"], ["ascites"]),
    "peritonitis": (["K65"], ["peritonitis"]),
    "hepatocellular carcinoma": (["C22.0"], ["hepatocellular"]),
    "hepatitis b": (["B18.1", "B16"], ["hepatitis b"]),
    "hepatitis c": (["B18.2", "B17.1"], ["hepatitis c"]),
    "fatty liver": (["K76.0", "K75.8"], ["fatty liver", "steatohepatitis"]),
    "pancreatitis": (["K85", "K86.0", "K86.1"], ["pancreatitis"]),
    "pancreatic insufficiency": (["K86.81", "K90.3"], ["pancreatic insufficiency"]),
    "pancreatogenic diabetes": (["E13"], ["pancreatogenic"]),
    "reflux": (["K21"], ["reflux"]),
    "barrett": (["K22.7"], ["barrett"]),
    "esophageal": (["C15", "K22"], ["esophageal"]),
    "ulcerative colitis": (["K51"], ["ulcerative colitis"]),
    "sclerosing cholangitis": (["K83.0"], ["sclerosing cholangitis"]),
    "inflammatory bowel": (["K50", "K51"], ["crohn", "colitis"]),
    "colorectal": (["C18", "C19", "C20"], ["colon", "rectal", "colorectal"]),
    "celiac": (["K90.0"], ["celiac"]),
    "iron-deficiency": (["D50"], ["iron deficiency"]),
    "pylori": (["B96.81"], ["pylori"]),
    "peptic ulcer": (["K25", "K26", "K27"], ["peptic ulcer", "gastric ulcer"]),
    "chronic obstructive pulmonary": (["J44"], ["chronic obstructive"]),
    "interstitial lung": (["J84", "J99"], ["interstitial lung", "pulmonary fibrosis"]),
    "pulmonary fibrosis": (["J84.1"], ["pulmonary fibrosis"]),
    "rheumatoid arthritis": (["M05", "M06"], ["rheumatoid"]),
    "pulmonary embolism": (["I26"], ["pulmonary embolism"]),
    "deep vein thrombosis": (["I82", "I80"], ["deep vein"]),
    "sarcoidosis": (["D86"], ["sarcoidosis"]),
    "lung cancer": (["C34"], ["lung cancer", "lung carcinoma"]),
    "pleural effusion": (["J90", "J91"], ["pleural effusion"]),
    "superior vena cava": (["I87.1"], ["vena cava"]),
    "carotid": (["I65.2"], ["carotid"]),
    "transient cerebral ischemic": (["G45"], ["transient ischemic", "transient cerebral"]),
    "b12 deficiency": (["E53.8", "D51"], ["b12 deficiency", "cobalamin"]),
    "subacute combined degeneration": (["E53.8", "G32.0"], ["subacute combined"]),
    "alcohol": (["F10", "K70"], ["alcohol"]),
    "wernicke": (["E51.2"], ["wernicke"]),
    "uremic encephalopathy": (["N18.9"], ["uremic"]),
    "end stage renal": (["N18.6"], ["end stage renal", "end-stage renal"]),
    "hypertensive emergency": (["I16"], ["hypertensive emergency", "hypertensive crisis"]),
    "posterior reversible": ([], ["posterior reversible"]),
    "malignant": (["C"], ["malignan", "carcinoma", "cancer"]),
    "cancer": (["C"], ["cancer", "carcinoma", "malignan"]),
    "inappropriate antidiuretic": (["E22.2"], ["inappropriate antidiuretic", "siadh"]),
    "neutropenia": (["D70"], ["neutropenia"]),
    "sickle cell": (["D57"], ["sickle"]),
    "vaso-occlusive": (["D57.0"], ["vaso-occlusive", "vaso occlusive"]),
    "acute chest": (["D57.01"], ["acute chest"]),
    "antiphospholipid": (["D68.61"], ["antiphospholipid"]),
    "lymphocytic leukemia": (["C91.1"], ["lymphocytic leukemia"]),
    "hemolytic anemia": (["D59"], ["hemolytic anemia"]),
    "tumor lysis": (["E88.3"], ["tumor lysis"]),
    "sclerosis": (["M34"], ["systemic sclerosis", "scleroderma"]),
    "scleroderma": (["M34"], ["scleroderma"]),
    "renal crisis": (["M34", "I15"], ["renal crisis"]),
    "vasculitis": (["M31.3", "M31.7", "M31"], ["vasculitis", "polyangiitis"]),
    "polyangiitis": (["M31.3"], ["polyangiitis"]),
    "glomerulonephritis": (["N00", "N01", "N03", "N05"], ["glomerulonephritis"]),
    "ankylosing": (["M45"], ["ankylosing"]),
    "uveitis": (["H20"], ["uveitis"]),
    "human immunodeficiency virus": (["B20", "Z21"], ["immunodeficiency virus", "hiv"]),
    "pneumocystis": (["B59"], ["pneumocystis"]),
    "streptococc": (["A40", "B95"], ["streptococc"]),
    "pyelonephritis": (["N10", "N11", "N12"], ["pyelonephritis"]),
    "pneumonia": (["J18", "J15", "J13"], ["pneumonia"]),
    "respiratory failure": (["J96"], ["respiratory failure"]),
    "alcoholic": (["K70"], ["alcoholic"]),
    "opioid": (["F11"], ["opioid"]),
    "preeclampsia": (["O11", "O14"], ["preeclampsia", "pre-eclampsia"]),
    "eclampsia": (["O15"], ["eclampsia"]),
    "hellp": (["O14.2"], ["hellp"]),
    "gestational diabetes": (["O24.4"], ["gestational diabetes"]),
    "diabetes in pregnancy": (["O24"], ["diabetes", "pregnancy"]),
    "macrosomia": (["O36.6", "P08"], ["macrosomia"]),
    "neonatal hypoglycemia": (["P70"], ["neonatal hypoglycemia"]),
    "peripartum cardiomyopathy": (["O90.3"], ["peripartum"]),
    "pregnancy": (["O", "Z34", "Z33"], ["pregnancy", "gestation"]),
    "puerperium": (["O85", "O86", "O87", "O90"], ["puerper", "postpartum"]),
    "postpartum thyroiditis": (["O90.5", "E06.3"], ["thyroiditis"]),
    "hyperemesis": (["O21"], ["hyperemesis"]),
    "congenital anomaly": (["Q"], ["congenital"]),
    "psoriasis": (["L40"], ["psoriasis"]),
    "psoriatic arthritis": (["L40.5", "M07"], ["psoriatic"]),
    "cellulitis": (["L03"], ["cellulitis"]),
    "soft-tissue infection": (["L03", "L08"], ["soft tissue", "cellulitis"]),
    "stevens": (["L51.1", "L51.3"], ["stevens"]),
    "toxic epidermal necrolysis": (["L51.2"], ["toxic epidermal"]),
    "dermatomyositis": (["M33"], ["dermatomyositis"]),
    "hidradenitis": (["L73.2"], ["hidradenitis"]),
    "prostate cancer": (["C61"], ["prostate cancer"]),
    "bone metasta": (["C79.51", "C79.5"], ["bone metasta"]),
    "neurogenic bladder": (["N31"], ["neurogenic bladder"]),
    "urinary tract infection": (["N39.0", "N30"], ["urinary tract infection", "cystitis"]),
    "renal scarring": (["N26", "Q60"], ["renal scarring", "reflux nephropathy"]),
    "graves": (["E05.0"], ["graves"]),
    "thyroid eye": (["H06.2", "E05.0"], ["thyroid eye", "exophthalmos", "orbitopathy"]),
    "giant cell arteritis": (["M31.6"], ["giant cell arteritis", "temporal arteritis"]),
    "ischemic optic neuropathy": (["H47.01"], ["optic neuropathy", "ischemic optic"]),
    "vision loss": (["H54"], ["blindness", "vision loss", "visual loss"]),
    "otitis externa": (["H60.2", "H60.3"], ["otitis externa"]),
    "sinusitis": (["J32", "J01"], ["sinusitis"]),
    "otitis media": (["H66", "H65"], ["otitis media"]),
    "zoster": (["B02"], ["zoster"]),
    "marfan": (["Q87.4"], ["marfan"]),
    "ectopia lentis": (["H27.1", "Q12.1"], ["ectopia lentis", "lens dislocation"]),
    "polycystic kidney": (["Q61"], ["polycystic kidney"]),
    "intracranial aneurysm": (["I67.1", "Q28"], ["cerebral aneurysm", "berry aneurysm", "intracranial aneurysm"]),
    "hepatic cysts": (["Q44.6", "K76.89"], ["hepatic cyst", "liver cyst"]),
    "cystic fibrosis": (["E84"], ["cystic fibrosis"]),
    "bronchiectasis": (["J47"], ["bronchiectasis"]),
    "neurofibromatosis": (["Q85.0"], ["neurofibromatosis"]),
    "hemochromatosis": (["E83.11"], ["hemochromatosis"]),
    "cardiomyopathy": (["I42", "I43"], ["cardiomyopathy"]),
    "down syndrome": (["Q90"], ["down syndrome"]),
    "atrioventricular septal defect": (["Q21.2"], ["atrioventricular septal", "endocardial cushion"]),
    "duodenal atresia": (["Q41"], ["duodenal atresia"]),
    "hypothyroidism": (["E03", "E00", "E02"], ["hypothyroidism"]),
    "scoliosis": (["M41"], ["scoliosis"]),
    "optic glioma": (["C72.3", "D33.3"], ["optic glioma", "optic nerve glioma"]),
}

_WORD = re.compile(r"[a-z0-9]+")


def _fragments(text):
    t = text.lower()
    parts = re.split(r"[/,;]| and | with | & ", t)
    for inner in re.findall(r"\(([^)]*)\)", t):
        parts += re.split(r"[/,;]| and | & ", inner)
    out = []
    for p in parts:
        words = [ABBREV.get(w, w) for w in _WORD.findall(p)]
        if words:
            out.append(" ".join(words))
    return out


def _resolve(text, by_icd, by_kw):
    dx = set()
    for f in _fragments(text):
        for key, (icds, kws) in VOCAB.items():
            if key in f:
                for p in icds:
                    dx |= by_icd(p)
                for k in kws:
                    dx |= by_kw(k)
        if len(f) >= 6:
            dx |= by_kw(f)   # direct name match of the (expanded) fragment
    return dx


def _expected(cell):
    classes = re.findall(r"C[123]", cell)
    return classes or ["?"], ("sparse" in cell.lower()), ("finding-site" in cell.lower())


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dx_rows = con.execute("SELECT diagnosis_id, REPLACE(UPPER(icd10_code),'.',''), "
                          "LOWER(display_name), icd10_code FROM diagnoses").fetchall()
    icd_cache, kw_cache = {}, {}

    def by_icd(prefix):
        p = prefix.replace(".", "").upper()
        if p not in icd_cache:
            icd_cache[p] = {d for d, code, _, _ in dx_rows if code and code.startswith(p)}
        return icd_cache[p]

    def by_kw(kw):
        if kw not in kw_cache:
            kw_cache[kw] = {d for d, _, name, _ in dx_rows if kw in name}
        return kw_cache[kw]

    code_of = {d: raw for d, _, _, raw in dx_rows}
    name_of = {d: name for d, _, name, _ in dx_rows}

    edge = {}
    fs_dx = set()
    for a, b, rc, rt in con.execute(
            "SELECT dx_a, dx_b, relation_class, relation_type FROM diagnosis_relations"):
        c = int(rc[0])
        for k in ((a, b), (b, a)):
            if k not in edge or c < edge[k]:
                edge[k] = c
        if rt == "shared_finding_site":
            fs_dx.add(a); fs_dx.add(b)

    def best_class(sets):
        best = None
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                for a in sets[i]:
                    for b in sets[j]:
                        c = edge.get((a, b))
                        if c is not None and (best is None or c < best):
                            best = c
        return best

    def modal_cat(dxset):
        cats = Counter((code_of[d] or "")[:3] for d in dxset if code_of[d])
        return cats.most_common(1)[0][0] if cats else ""

    rows, gaps = [], []
    for line in open(MD):
        if not re.match(r"\|\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        num, pair = int(cells[1]), cells[2]
        exp_cell = cells[4] if len(cells) > 4 else ""
        spec_cell = cells[5] if len(cells) > 5 else ""
        cross = "✦" in spec_cell
        segs = [s for s in re.split(r"↔|->", pair) if s.strip()]
        if len(segs) < 2:
            continue
        sets = [_resolve(s, by_icd, by_kw) for s in segs]
        exp_classes, sparse, is_fs = _expected(exp_cell)
        found = best_class(sets)
        # finding-site rows: organ target won't resolve to a dx; pass if the
        # resolved condition participates in any shared_finding_site edge.
        fs_hit = is_fs and found is None and any(s & fs_dx for s in sets if s)
        if fs_hit:
            found = 2
        found_str = "C2(fs)" if fs_hit else (f"C{found}" if found else "none")
        nonempty = [s for s in sets if s]
        if len(nonempty) < 2 and not fs_hit:
            status = "UNMAPPED"
        elif found is None:
            status = "MISS"
        elif found_str.replace("(fs)", "") in exp_classes or (found == 1 and "C2" in exp_classes):
            status = "HIT"
        elif found <= max(int(e[1]) for e in exp_classes if e != "?"):
            status = "HIT(stronger)"
        else:
            status = "WEAKER"
        rows.append({"row": num, "pair": pair[:56], "expected": "/".join(exp_classes),
                     "sparse": "yes" if sparse else "", "cross": "✦" if cross else "",
                     "found": found_str, "A_n": len(sets[0]), "B_n": len(sets[-1]),
                     "status": status})
        if status in ("MISS", "WEAKER"):
            ex = lambda s: " ; ".join(f"{code_of[d]} {name_of[d][:26]}" for d in sorted(s)[:3])
            has_typed_exp = any(e in ("C1", "C2") for e in exp_classes)
            kind = "FIX (C1/C2)" if has_typed_exp and not sparse else "DECIDE (C3/sparse)"
            gaps.append({"row": num, "kind": kind, "pair": pair[:60],
                         "expected": "/".join(exp_classes), "found": found_str,
                         "suggest_a_icd": modal_cat(sets[0]), "suggest_b_icd": modal_cat(sets[-1]),
                         "A_examples": ex(sets[0]), "B_examples": ex(sets[-1]),
                         # blank columns for the clinician to fill:
                         "FILL_a_icd": "", "FILL_b_icd": "", "FILL_relation_class": "",
                         "FILL_relation_type": "", "FILL_note": ""})

    with open("comorbidity_coverage_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open("curated_diagnosis_edges_TEMPLATE.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gaps[0].keys()))
        w.writeheader(); w.writerows(gaps)

    st = Counter(r["status"] for r in rows)
    print(f"rows: {len(rows)} | {dict(st)}")
    print(f"resolved: {len(rows) - st['UNMAPPED']} | hit: {st['HIT'] + st['HIT(stronger)']} "
          f"| miss/weaker: {st['MISS'] + st['WEAKER']} | unmapped: {st['UNMAPPED']}")
    print(f"\nstill UNMAPPED: {[r['row'] for r in rows if r['status']=='UNMAPPED']}")
    print(f"\ngaps -> curated_diagnosis_edges_TEMPLATE.csv ({len(gaps)} rows to fill)")
    for g in gaps:
        print(f"  #{g['row']:>3} [{g['expected']:6s} found {g['found']:6s}] {g['pair']}")


if __name__ == "__main__":
    main()
