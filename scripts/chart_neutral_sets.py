"""Chart-neutral sets: for every patient, the 3-character ICD-10 categories of conditions the chart documents in its
profile (chronic_conditions) that may be absent from the graph-derived reference. A predicted diagnosis that is
unmatched to the reference but falls in this set is neutral (neither correct nor an error).
Mapping = curated alias table (lay names, abbreviations) -> fuzzy match against the benchmark's own diagnoses table
-> fuzzy match against ICD-10-CM code titles. Deterministic; no LLM."""
import re, json, sys, difflib
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
DM = {"E08", "E09", "E10", "E11", "E13"}; HTN = {"I10", "I11", "I12", "I13", "I15", "I16"}
ALIAS = [  # (regex on normalized string, categories)
 (r"\btype 2 diabetes|\bt2dm\b|\bdm2\b|diabetes mellitus type 2|non.?insulin", DM), (r"\btype 1 diabetes|\bt1dm\b|\bdm1\b", DM), (r"\bdiabet", DM), (r"gestational diabetes|\bgdm\b", {"O24"}),
 (r"diabetic (peripheral )?neuropath", DM | {"G63", "G62"}), (r"diabetic retinopath", DM | {"H36"}), (r"diabetic nephropath|diabetic kidney", DM | {"N18"}),
 (r"hypertensi|\bhtn\b", HTN), (r"renal artery stenosis|renovascular", HTN | {"I70", "I77"}), (r"pulmonary hypertension", {"I27"}),
 (r"hyperlipid|dyslipid|hypercholesterol|\bhld\b", {"E78"}), (r"obes|overweight", {"E66"}), (r"metabolic syndrome", {"E88"}),
 (r"coronary artery disease|\bcad\b|ischemic heart|angina|myocardial infarction|\bmi\b", {"I25", "I20", "I21", "I24"}), (r"heart failure|\bchf\b|\bhfref\b|\bhfpef\b|cardiomyopathy", {"I50", "I42", "I11", "I13"}),
 (r"atrial fibrillation|\bafib\b|\baf\b|atrial flutter", {"I48"}), (r"valve|mitral|aortic stenosis|rheumatic heart", {"I05", "I06", "I07", "I08", "I09", "I34", "I35", "Z95"}),
 (r"peripheral (arterial|vascular) disease|\bpad\b|\bpvd\b", {"I73", "I70"}), (r"stroke|\bcva\b|cerebrovascular|\btia\b", {"I63", "I69", "G45", "Z86"}), (r"dvt|deep vein|venous thrombo|pulmonary embol|\bpe\b|\bvte\b", {"I82", "I26", "Z86"}),
 (r"copd|chronic obstructive|emphysema|chronic bronchitis", {"J44", "J43"}), (r"asthma", {"J45"}), (r"allergic rhinitis|hay fever", {"J30"}), (r"sleep apnea|\bosa\b", {"G47"}), (r"pulmonary fibrosis|\bipf\b|interstitial lung", {"J84"}),
 (r"gerd|reflux", {"K21"}), (r"peptic ulcer|gastric ulcer|duodenal ulcer", {"K25", "K26", "K27"}), (r"irritable bowel|\bibs\b", {"K58"}), (r"crohn|ulcerative colitis|inflammatory bowel", {"K50", "K51"}), (r"celiac", {"K90"}),
 (r"constipation", {"K59"}), (r"cirrhosis|fatty liver|\bnafld\b|\bnash\b|hepatitis", {"K70", "K74", "K76", "B18"}), (r"achalasia", {"K22"}), (r"diverticul", {"K57"}), (r"pancreatit", {"K85", "K86"}), (r"gallstone|cholelith", {"K80"}),
 (r"chronic kidney|\bckd\b|renal insufficiency|end.stage renal|\besrd\b", {"N18", "N19"}), (r"nephrolith|kidney stone|urolith|ureterolith", {"N20", "Z87"}), (r"benign prostatic|\bbph\b", {"N40"}), (r"urinary tract infection|\buti\b", {"N39"}), (r"incontinence", {"N39", "R32"}),
 (r"hypothyroid|hashimoto", {"E03", "E06"}), (r"hyperthyroid|graves", {"E05"}), (r"hyperparathyroid", {"E21"}), (r"polycystic ovar|\bpcos\b", {"E28"}), (r"osteoporosis|osteopenia", {"M81", "M85"}), (r"vitamin d deficien", {"E55"}), (r"gout", {"M10", "M1A"}),
 (r"osteoarthritis|degenerative joint", {"M15", "M16", "M17", "M18", "M19"}), (r"rheumatoid", {"M05", "M06"}), (r"lupus|\bsle\b", {"M32"}), (r"back pain|lumbar|disc disease|spondyl", {"M54", "M51", "M47"}), (r"fibromyalg", {"M79"}), (r"psoriasis", {"L40"}), (r"atopic dermatitis|eczema", {"L20", "L30"}),
 (r"anemia", {"D50", "D51", "D52", "D53", "D63", "D64"}), (r"iron deficien", {"D50", "E61"}), (r"sickle", {"D57"}), (r"thalass", {"D56"}), (r"lymphoma", {"C81", "C82", "C83", "C85"}), (r"leukemia", {"C91", "C92"}), (r"lung cancer|non.small cell|\bnsclc\b", {"C34"}), (r"breast cancer", {"C50", "Z85"}), (r"prostate cancer", {"C61", "Z85"}), (r"colon cancer|colorectal", {"C18", "C19", "C20", "Z85"}), (r"cancer|carcinoma|malignan", {"Z85"}),
 (r"depress|\bmdd\b", {"F32", "F33"}), (r"anxiety|\bgad\b|panic", {"F41"}), (r"bipolar", {"F31"}), (r"schizophren", {"F20", "F25"}), (r"ptsd|post.?traumatic stress", {"F43"}), (r"\badhd\b|attention.deficit", {"F90"}), (r"autism", {"F84"}), (r"dementia|alzheimer", {"F03", "G30"}), (r"bulimia|anorexia nervosa|eating disorder", {"F50"}),
 (r"alcohol", {"F10"}), (r"tobacco|smok|nicotine", {"F17", "Z72", "Z87"}), (r"opioid|opiate", {"F11"}), (r"cannabis|marijuana", {"F12"}), (r"cocaine", {"F14"}), (r"stimulant|amphetamine|methamphetamine", {"F15"}), (r"steroid|laxative|diuretic misuse|substance use", {"F55", "F19"}),
 (r"epilep|seizure", {"G40"}), (r"migraine", {"G43"}), (r"parkinson", {"G20"}), (r"multiple sclerosis", {"G35"}), (r"neuropathy", {"G62", "G63"}), (r"developmental delay", {"F88", "R62"}), (r"cerebral palsy", {"G80"}), (r"failure to thrive", {"R62"}),
 (r"down syndrome|trisomy 21", {"Q90"}), (r"atrioventricular septal|\bavsd\b|septal defect|congenital heart", {"Q21", "Q20", "Q24"}), (r"duodenal atresia|intestinal atresia", {"Q41"}), (r"hirschsprung", {"Q43"}), (r"infant of (a )?diabetic mother", {"P70"}), (r"hypertrophic cardiomyopathy of the newborn", {"P29", "I42"}), (r"prematur|preterm", {"P07", "O60"}),
 (r"preterm labor", {"O60"}), (r"rh.?(d)?.?negative", {"Z67", "O36"}), (r"chlamydia", {"A56", "A74", "Z86"}), (r"hiv", {"B20", "Z21"}), (r"hepatitis c", {"B18"}), (r"rib fracture", {"S22"}), (r"pulmonary contusion", {"S27"}), (r"traumatic brain injury|concussion", {"S06"}),
 (r"dysphagia", {"R13"}), (r"lung abscess|aspiration", {"J85", "J69"}), (r"glaucoma", {"H40"}), (r"cataract", {"H25", "H26"}), (r"hearing loss", {"H90", "H91"}), (r"endometriosis", {"N80"}), (r"fibroid|leiomyoma", {"D25"}), (r"infertility", {"N97"}), (r"menopaus", {"N95"}),
 (r"chronic pain", {"G89"}), (r"insomnia", {"G47", "F51"}), (r"vertigo|meniere", {"H81"}), (r"kidney transplant|renal transplant", {"Z94"}), (r"splenectomy", {"Z90"}), (r"appendectomy", {"Z90"}), (r"cholecystectomy", {"Z90"}), (r"hysterectomy", {"Z90"}),
]
STRIP = r"\(.*?\)|\bhistory of\b|\bhx of\b|\bstatus post\b|\bs/p\b|\bresolved\b|\bhealed\b|\btreated\b|\bwell.controlled\b|\bpoorly.controlled\b|\bmoderate\b|\bmild\b|\bsevere\b|\brecurrent\b|\bchronic\b|\bessential\b|\bprimary\b|\bsecondary to\b.*$"
def norm(s): s = s.lower(); s = re.sub(STRIP, " ", s); return re.sub(r"[^a-z0-9/ ]", " ", s).strip()
_dx = None
def load_ontology(cur):
    global _dx
    cur.execute("select icd10_code, display_name, snomed_desc, icd10_desc from diagnoses where icd10_code is not null"); _dx = []
    for code, *names in cur.fetchall():
        for n in names:
            if n: _dx.append((norm(n), code[:3]))
    cur.execute("select code, display from terminology_codes where system='icd10cm' and length(code)=3"); _dx += [(norm(d), c) for c, d in cur.fetchall()]
def map_condition(s):
    low = s.lower(); cats = set()
    for rx, c in ALIAS:
        if re.search(rx, low): cats |= c
    if cats: return cats, "alias"
    n = norm(s)
    if not n or _dx is None: return set(), "none"
    exact = {c for name, c in _dx if name == n}
    if exact: return exact, "exact"
    toks = set(n.split()); best = (0.0, None)
    for name, c in _dx:
        nt = set(name.split())
        if not nt: continue
        j = len(toks & nt) / len(toks | nt)
        if j > best[0]: best = (j, c)
    return ({best[1]}, "fuzzy") if best[0] >= 0.6 else (set(), "none")
SURG = [(r"appendectomy", {"Z90", "K35", "K36", "K37"}), (r"cholecystectomy", {"Z90", "K80"}), (r"hysterectomy", {"Z90"}), (r"tonsillectomy", {"Z90"}), (r"splenectomy", {"Z90", "D73"}),
        (r"cesarean|c.section", {"Z98", "O34"}), (r"bypass|cabg|stent|angioplasty", {"Z95", "I25"}), (r"valve", {"Z95"}), (r"gastric bypass|sleeve|bariatric", {"Z98", "K91", "E66"}), (r"thyroidectomy", {"E89", "Z90"}),
        (r"colectomy|bowel resection", {"Z90"}), (r"nephrectomy", {"Z90"}), (r"mastectomy", {"Z90"}), (r"transplant", {"Z94"}), (r"arthroplasty|joint replacement", {"Z96"}), (r"pacemaker|defibrillator|icd\b", {"Z95"})]
def neutral_sets(cur):
    load_ontology(cur); cur.execute("select patient_id, profile from longitudinal_patients"); out = {}; how = defaultdict(int); unmapped = []
    for pid, pr in cur.fetchall():
        pr = pr if isinstance(pr, dict) else json.loads(pr or "{}"); cats = set()
        for s in pr.get("chronic_conditions") or []:
            c, m = map_condition(str(s)); cats |= c; how[m] += 1
            if not c: unmapped.append((pid, s))
        for s in pr.get("surgical_history") or []:
            s = str(s if not isinstance(s, dict) else s.get("procedure") or s.get("name") or "").lower()
            for rx, c in SURG:
                if re.search(rx, s): cats |= c | {"Z98"}
        sm = str(pr.get("smoking_status") or "").lower()
        if sm and not re.search(r"never|non.?smoker|\bno\b", sm): cats |= {"F17", "Z72", "Z87"} if not re.search(r"former|quit|ex.?smoker", sm) else {"Z87"}
        al = str(pr.get("alcohol_use") or "").lower()
        if re.search(r"heavy|daily|disorder|abuse|dependence|excess|binge|[4-9]\+? ?drinks|1[0-9] drinks", al): cats |= {"F10"}
        out[pid] = sorted(cats)
    return out, dict(how), unmapped
if __name__ == "__main__":
    from eval.config import get_pg_connection
    cur = get_pg_connection().cursor(); sets, how, unmapped = neutral_sets(cur)
    json.dump(sets, open(ROOT / "data/chart_neutral_sets.json", "w"))
    print("mapping methods:", how, "| unmapped strings:", len(unmapped)); print("sample unmapped:", [s for _, s in unmapped[:25]])
    hand = {2610: [], 2631: ["S22", "S27", "S06", "Z87"], 2850: ["I15", "I10", "I70", "I05", "I08", "I09", "I01", "Z95", "J44", "M19", "M15", "M18", "C82", "C34", "I48"],
            1726: ["F50", "F41", "F55", "F19"], 2046: ["I10", "I15", "N20", "R10", "R63", "Z90", "K35", "K36", "K37"], 2834: ["I10", "E11", "E78"], 2285: ["E11", "E08", "I10", "E78", "G63", "H36"],
            1853: ["J30", "O60", "O42", "O09", "Z87"], 2323: ["J84", "I63", "I69", "R13", "J85", "J69", "I10", "K21"], 1969: ["E11", "I10", "M10", "M1A", "E66", "F10", "K22", "N20", "Z87", "D52"],
            1741: ["Z67", "O36", "A56", "A74", "Z86"], 2549: ["D62", "Z72", "F17", "F10", "V43", "V49", "V89"], 2376: ["Q90", "Q21", "Q41", "Q43", "Z98", "Z87", "I42", "P29", "P70", "P92", "R62"]}
    print("\nvalidation against the 13 hand-built sets (auto covers hand / extras in auto):")
    for pid, h in hand.items():
        a = set(sets[pid]); print(f"  {pid}: hand {len(h):2d} covered {len(set(h)&a):2d}  missing {sorted(set(h)-a)}  extra {sorted(a-set(h))}")
