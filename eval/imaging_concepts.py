"""Ontology-grounded concept F1 for the imaging-indication clinical question.

This is the paper's primary imaging metric ("Question concept F1"). A question
(prediction or reference) is reduced to the set of clinical concepts it names;
precision, recall and F1 are taken on those sets, so wording differences between
a model's question and the LLM-authored reference are not penalized.

Concept inventory = every diagnosis and clinical finding in the benchmark's own
knowledge graph (display name and SNOMED description share one concept id), read
from the `diagnoses` and `clinical_findings` tables of the loaded database, plus
a small curated layer of lay synonyms and imaging shorthand (so "kidney stone",
"IUP", "SBO" resolve). Matching is abbreviation-aware, order-free within a short
window, plural-folded, and snaps misspelt tokens to the nearest vocabulary token.
No external ontology files are needed.

Determinism: the paper's script broke two ties by Python's per-process hash order
(the canonical stem of a token and the anchor token of a multi-word phrase); this
port breaks them by (length, string), so results are reproducible run to run.
Over the paper's imaging runs the aggregate concept F1 agrees with the original to
within its own run-to-run spread (about ±0.002).

Usage:
    extractor = ConceptExtractor.from_db(conn)      # cached per process
    p, r, f1, n_pred, n_ref = extractor.prf(question, reference_question)
"""

from __future__ import annotations

import difflib
import re
import threading
from collections import Counter, defaultdict
from typing import Iterable

from eval import semantic_match as _sm

# ---------------------------------------------------------------------------
# Tokenizer: abbreviation expansion (summarization matcher's table + imaging and
# ordering shorthand), slash forms, plural folding.
# ---------------------------------------------------------------------------

EXTRA_ABBREVIATIONS: dict[str, str] = {
    "iup": "intrauterine pregnancy", "sbo": "small bowel obstruction", "lbo": "large bowel obstruction",
    "hape": "high altitude pulmonary edema", "rds": "respiratory distress syndrome",
    "ttn": "transient tachypnea newborn", "ttnb": "transient tachypnea newborn", "cxr": "chest xray",
    "axr": "abdominal xray", "ptx": "pneumothorax", "htx": "hemothorax", "pna": "pneumonia",
    "nph": "normal pressure hydrocephalus", "avsd": "atrioventricular septal defect",
    "vsd": "ventricular septal defect", "asd": "atrial septal defect", "chd": "congenital heart disease",
    "pvr": "post void residual", "mvc": "motor vehicle collision", "mva": "motor vehicle accident",
    "lle": "left lower extremity", "rle": "right lower extremity", "lue": "left upper extremity",
    "rue": "right upper extremity", "ue": "upper extremity", "le": "lower extremity",
    "ipf": "idiopathic pulmonary fibrosis", "nsclc": "non small cell lung cancer",
    "gtd": "gestational trophoblastic disease", "ttts": "twin twin transfusion syndrome",
    "iugr": "intrauterine growth restriction", "sdh": "subdural hematoma", "edh": "epidural hematoma",
    "goo": "gastric outlet obstruction", "pud": "peptic ulcer disease", "lad": "lymphadenopathy",
    "hf": "heart failure", "phtn": "pulmonary hypertension", "hx": "history", "eval": "evaluate",
    "dx": "diagnosis", "pt": "patient", "sx": "symptoms", "ctap": "ct abdomen pelvis", "postvoid": "post void",
    "intraabdominal": "intra abdominal", "sao2": "oxygen saturation", "hgb": "hemoglobin", "etoh": "alcohol",
    "cnh": "", "abd": "abdominal", "r": "right", "l": "left",
}
SLASH_FORMS: dict[str, str] = {
    "r/o": " rule out ", "s/p": " status post ", "f/u": " follow up ", "h/o": " history of ",
    "s/o": " setting of ", "w/": " with ", "n/v": " nausea vomiting ", "c/f": " concern for ",
}
ABBREVIATIONS: dict[str, str] = {**{k.strip(): v for k, v in _sm._ABBREV_MAP.items()}, **EXTRA_ABBREVIATIONS}


def tokens(text: str | None) -> list[str]:
    t = (text or "").lower().replace("’", "'")
    for k, v in SLASH_FORMS.items():
        t = t.replace(k, v)
    t = t.replace("-", " ")
    out: list[str] = []
    for w in re.findall(r"[a-z0-9]+", t):
        out += re.findall(r"[a-z0-9]+", ABBREVIATIONS.get(w, w))
    return out


def stems(w: str) -> set[str]:
    """All plausible singular forms: 'stones' -> stone, 'nephrolithiases' -> nephrolithiasis."""
    out = {w}
    if len(w) > 4:
        if w.endswith("ies"):
            out.add(w[:-3] + "y")
        if w.endswith("ses"):
            out.add(w[:-3] + "sis")
        if w.endswith("es"):
            out.add(w[:-2])
        if w.endswith("s"):
            out.add(w[:-1])
    return out


def canon(w: str) -> str:
    """Shortest singular form; ties broken alphabetically so results are deterministic."""
    return sorted(stems(w), key=lambda s: (len(s), s))[0]


# ---------------------------------------------------------------------------
# Curated concept layer: one clinical idea with its surface forms (a trailing *
# is a prefix pattern). Specific curated concepts absorb ontology concepts whose
# names trigger them; the GENERIC ones are too broad to absorb anything.
# ---------------------------------------------------------------------------

CONCEPTS: dict[str, list[str]] = {
    "viable intrauterine pregnancy": ["viab* pregnan*", "intrauterine pregnan*", "normal pregnan*", "active pregnan*", "fetal status", "fetal viab*", "fetal heart*", "fetal cardiac", "viab*", "pregnan* location"],
    "molar / trophoblastic disease": ["molar", "trophoblastic"], "multiple gestation": ["multiple gestation*", "twin*", "multifetal"],
    "amniotic fluid disorder": ["polyhydramnios", "oligohydramnios", "amniotic fluid"], "twin-twin transfusion": ["twin twin transfusion"], "growth restriction": ["growth restrict*"],
    "placental abnormality": ["placent*"], "twin discordance": ["discordan*"], "ectopic pregnancy": ["ectopic"], "pregnancy loss": ["abortion", "miscarriage", "pregnancy loss", "retained products", "subchorionic"],
    "respiratory distress syndrome": ["respiratory distress syndrome", "hyaline membrane", "surfactant"], "transient tachypnea": ["transient tachypnea"],
    "pneumonia / infection": ["pneumonia", "infiltrat*", "consolidation", "infect*"], "aspiration": ["aspiration"], "pneumothorax": ["pneumothorax", "air leak"],
    "pulmonary edema": ["pulmonary edema", "high altitude pulmonary"], "pulmonary embolism": ["pulmonary embol*"],
    "urinary stone": ["stone", "calculus", "calculi", "ureterolithiasis", "nephrolithiasis", "urolithiasis", "lithiasis"], "hydronephrosis": ["hydronephrosis"],
    "obstruction": ["obstruct*"], "pyelonephritis": ["pyelonephritis"], "renal infarction": ["renal infarct*"], "ileus": ["ileus"], "volvulus / malrotation": ["volvulus", "malrotation"],
    "dilated loops / air-fluid levels": ["air fluid", "dilated loop*"], "perforation": ["free air", "perforat*"], "adhesions": ["adhesi*"],
    "malignancy / mass": ["malignan*", "cancer", "carcinoma", "adenocarcinoma", "tumor", "mass", "neoplasm", "lymphoma", "metasta*", "staging", "nodule*", "lesion"],
    "gastric outlet / pylorus": ["gastric outlet", "pylor*"], "ulcer": ["ulcer"], "structural cause": ["structural", "anatomic*"],
    "post-void residual": ["post void residual", "residual", "bladder volume"], "neurogenic bladder": ["neurogenic bladder", "cystopathy"],
    "urinary retention / distended bladder": ["bladder disten*", "distended bladder", "bladder full", "urinary retention", "retention"],
    "abscess / cavity": ["abscess", "cavit*"], "fibrosis / scar": ["fibro*", "scar*"], "opacity": ["opacity"],
    "hemorrhage / free fluid": ["hemorrhag*", "bleed*", "hematoma", "hemoperitoneum", "free fluid", "extravasation"], "stroke / infarct": ["stroke", "infarct*", "ischemi*"],
    "atresia / stenosis": ["duodenal atresia", "atresia", "double bubble", "annular pancreas", "stenosis", "stricture"], "pneumatosis": ["pneumatosis"],
    "hirschsprung": ["hirschsprung", "aganglion*", "transition zone"], "meconium plug": ["meconium plug"], "meconium aspiration": ["meconium aspiration"],
    "septal defect / congenital heart disease": ["septal defect", "atrioventricular canal", "av canal", "endocardial cushion", "congenital heart", "congenital cardiac"],
    "shunt": ["shunt*"], "pulmonary hypertension": ["pulmonary hypertension", "pulmonary pressure*", "pulmonary artery pressure*"], "hemothorax": ["hemothorax", "hemopneumothorax"],
    "tamponade": ["tamponade"], "mediastinal / hilar": ["mediastin*", "hilar"], "fracture": ["fracture"],
    "organ injury": ["organ injur*", "solid organ", "laceration", "splen*", "bladder rupture", "traumatic injur*", "intrathoracic injur*"],
    "arterial occlusion / perfusion": ["arterial occlusion", "occlusion", "thrombo*", "embol*", "vascular compromise", "vascular flow", "perfusion", "arterial flow", "blood flow", "vascular injur*"],
    "compartment syndrome": ["compartment syndrome"], "foreign body": ["foreign body"], "soft-tissue gas": ["subcutaneous gas", "gas"], "injection injury": ["injection injur*"],
    "spinal cord injury": ["spinal cord", "cord", "myelopathy", "hemisection", "brown sequard", "compress*", "transection", "contusion", "myelitis", "spine injur*", "spinal injur*"],
    "diaphragmatic hernia": ["diaphragm*", "hernia*"], "atelectasis": ["atelectasis"], "effusion": ["effusion"], "hydrocephalus": ["hydrocephalus"],
    "atrophy / neurodegeneration": ["atrophy", "neurodegenerat*", "alzheimer*"], "subdural": ["subdural"], "lymphadenopathy": ["lymphadenopathy", "adenopathy"], "sarcoidosis": ["sarcoid*"],
    "encephalitis / encephalopathy": ["encephalitis", "encephalopathy", "wernicke"], "valve disease": ["valve thrombosis", "prosthetic valve", "valve dysfunction", "mitral", "endocarditis", "vegetation", "regurgitation"],
    "ejection fraction": ["ejection fraction"],
    # presenting problems
    "vomiting": ["vomit*", "emesis", "nausea", "hyperemesis", "bilious"], "pregnancy": ["pregnan*"], "uterine size": ["size date*", "uter* larger", "uter* size", "fundal height", "abdominal size", "gestational age"],
    "cramping": ["cramp*"], "respiratory distress / dyspnea": ["respiratory distress", "tachypnea", "grunting", "breath sounds", "respiratory issue*", "shortness breath", "dyspnea", "difficulty breathing"],
    "hypoxia": ["hypox*", "desaturat*", "oxygen saturation", "cyanosis"], "cough": ["cough"], "fever": ["fever"], "high altitude": ["high altitude"], "cardiopulmonary process": ["cardiopulmonary process"],
    "flank pain": ["flank pain", "colic"], "abdominal pain": ["abdominal pain", "epigastric", "abdominal discomfort", "acute abdomen"],
    "obstipation / distension": ["obstipation", "flatus", "pass stool", "distention", "distension"], "early satiety": ["satiety", "fullness"],
    "weight loss / constitutional": ["weight loss", "constitutional", "fatig*", "failure thrive", "poor feeding"], "incontinence": ["incontinence", "leak*", "dribbl*", "difficulty initiating", "voiding"],
    "altered mental status / cognition": ["confusion", "altered mental status", "delirium", "agitation", "aphasia", "speech", "memory", "forgetful*", "cognitive", "dementia", "ataxia", "unstead*"],
    "heart failure / murmur": ["heart failure", "murmur"], "trauma": ["trauma", "stab*", "penetrating", "motor vehicle", "crush", "injury", "accident"], "shock": ["shock", "unstable", "tenderness", "guarding", "hemoglobin"],
    "tracheal deviation": ["trachea* deviat*"], "pain / swelling": ["hand pain", "swollen", "swelling"], "weakness / deficit": ["weakness", "neurologic deficit*", "sensory", "paralysis"], "chest pain": ["chest pain"],
}

GENERIC: frozenset[str] = frozenset({
    "obstruction", "malignancy / mass", "fracture", "hemorrhage / free fluid", "pneumonia / infection", "trauma",
    "structural cause", "opacity", "organ injury", "arterial occlusion / perfusion", "atresia / stenosis",
    "stroke / infarct", "pregnancy", "pain / swelling", "shock", "spinal cord injury", "valve disease",
})

STOP: frozenset[str] = frozenset("""a an the of and or to for in on at by with without from as is are was be this that these those it its any other due not no nos unspecified acute chronic subacute
 left right bilateral disease disorder disorders finding findings patient patients evaluate evaluation assess assessment rule out presence evidence cause causing causes etiology
 possible suspected history status post level type stage primary secondary into than more less""".split())

# Single-word surface forms shared by more than this many inventory entries are
# words, not concepts ('pain', 'renal', 'kidney', ...) and are dropped.
_MAX_SINGLE_TOKEN_DF = 60


def ctoks(text: str | None) -> list[str]:
    return [canon(t) for t in tokens(text) if t not in STOP and len(t) > 1]


class ConceptExtractor:
    """Concept-set extractor over an inventory of (concept_id, surface_name) pairs."""

    def __init__(self, inventory: Iterable[tuple[str, str | None]]):
        self.forms: dict[frozenset[str], str] = {}      # token set -> concept id
        for cid, name in inventory:
            self._add(cid, name)
        # Curated concepts absorb ontology concepts whose names trigger exactly one of them.
        self.absorb: dict[str, str] = {}
        for tk, cid in list(self.forms.items()):
            hits = [c for c in self._cur_hits(list(tk)) if c[2:] not in GENERIC]
            if len(hits) == 1:
                self.absorb[cid] = hits[0]
        df = Counter(t for tk in self.forms for t in tk)
        for tk in [tk for tk in self.forms if len(tk) == 1 and df[next(iter(tk))] > _MAX_SINGLE_TOKEN_DF]:
            del self.forms[tk]
        self.vocab: set[str] = {t for tk in self.forms for t in tk}
        self.index: dict[str, list[frozenset[str]]] = defaultdict(list)
        for tk in self.forms:
            self.index[max(tk, key=len)].append(tk)
        self._snap_cache: dict[str, str] = {}
        self.n_concepts = len(set(self.forms.values()))

    @classmethod
    def from_db(cls, conn) -> "ConceptExtractor":
        """Build from the benchmark's diagnoses and clinical findings (cached per process)."""
        return _cached_from_db(conn, cls)

    # -- inventory construction ------------------------------------------------
    def _add(self, cid: str, name: str | None) -> None:
        tk = frozenset(ctoks(name or ""))
        if not tk or (len(tk) == 1 and len(next(iter(tk))) < 4) or len(tk) > 6:
            return
        self.forms.setdefault(tk, cid)

    @staticmethod
    def _cur_hits(tk_list: list[str]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for c, forms in CONCEPTS.items():
            for f in forms:
                ev: list[str] | None = []
                for p in f.split():
                    if p.endswith("*"):
                        pref = canon(p[:-1])
                        m = [t for t in tk_list if t.startswith(pref)]
                    else:
                        sp = stems(p)
                        m = [t for t in tk_list if sp & stems(t)]
                    if not m:
                        ev = None
                        break
                    ev.append(m[0])
                if ev:
                    out.setdefault("C:" + c, set()).update(ev)
                    break
        return out

    # -- extraction --------------------------------------------------------------
    def _snap(self, t: str) -> str:
        if t in self.vocab or len(t) < 6:
            return t
        if t not in self._snap_cache:
            cands = [v for v in self.vocab if abs(len(v) - len(t)) <= 2 and v[0] == t[0]]
            m = difflib.get_close_matches(t, cands, n=1, cutoff=0.86)
            self._snap_cache[t] = m[0] if m else t
        return self._snap_cache[t]

    def concepts(self, text: str | None) -> set[str]:
        tl = [self._snap(t) for t in ctoks(text)]
        T = set(tl)
        found: dict[str, set[str]] = {}
        pos: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(tl):
            pos[t].append(i)

        def near(tk: frozenset[str]) -> bool:
            # a multi-word concept must appear as a phrase: all its words within a short window
            if len(tk) == 1:
                return True
            anchor = max(tk, key=lambda w: (len(w), w))   # deterministic anchor token
            return any(all(any(abs(j - i) <= len(tk) + 1 for j in pos[o]) for o in tk) for i in pos[anchor])

        for t in T:
            for tk in self.index.get(t, ()):
                if tk <= T and near(tk):
                    cid = self.forms[tk]
                    cid = self.absorb.get(cid, cid)
                    found.setdefault(cid, set()).update(tk)
        onto = {c: e for c, e in found.items() if not c.startswith("C:")}
        keep = {c: e for c, e in onto.items() if not any(e < e2 for e2 in onto.values())}   # maximal ontology matches only
        cu = self._cur_hits(tl)
        for c, e in found.items():
            if c.startswith("C:"):
                cu.setdefault(c, set()).update(e)
        keep = {c: e for c, e in keep.items() if not any(e <= e2 for e2 in cu.values())}
        keep.update(cu)
        return set(keep)

    def prf(self, question: str | None, reference: str | None) -> tuple[float, float, float, int, int]:
        """(precision, recall, f1, n_pred_concepts, n_ref_concepts) of question vs reference."""
        P, R = self.concepts(question), self.concepts(reference)
        tp = len(P & R)
        p = tp / len(P) if P else 0.0
        r = tp / len(R) if R else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0), len(P), len(R)


# ---------------------------------------------------------------------------
# Per-process cache keyed on the connection's DSN
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cache: dict[str, ConceptExtractor] = {}


def _db_key(conn) -> str:
    try:
        info = conn.info
        return f"{info.host}:{info.port}/{info.dbname}"
    except Exception:  # noqa: BLE001
        return "default"


def _cached_from_db(conn, cls=ConceptExtractor) -> ConceptExtractor:
    key = _db_key(conn)
    with _lock:
        if key not in _cache:
            _cache[key] = cls(load_inventory(conn))
        return _cache[key]


def load_inventory(conn) -> list[tuple[str, str | None]]:
    """(concept_id, surface_name) pairs from the graph; the same id for a display
    name and its SNOMED description, SNOMED id when grounded."""
    inv: list[tuple[str, str | None]] = []
    with conn.cursor() as cur:
        cur.execute("SELECT diagnosis_id, snomed_id, display_name, snomed_desc FROM diagnoses")
        for did, sn, dn, sd in cur.fetchall():
            cid = f"S{sn}" if sn else f"D{did}"
            inv.append((cid, dn)); inv.append((cid, sd))
        cur.execute("SELECT finding_id, snomed_id, display_name, snomed_desc FROM clinical_findings "
                    "WHERE finding_type::text <> 'demographic'")
        for fid, sn, dn, sd in cur.fetchall():
            cid = f"S{sn}" if sn else f"F{fid}"
            inv.append((cid, dn)); inv.append((cid, sd))
    return inv


def concept_f1_batch(extractor: ConceptExtractor, questions: list[str], references: list[str]) -> dict[str, float]:
    """Mean concept precision / recall / F1 over paired questions and references."""
    if not questions:
        return {"clinical_question_concept_f1": 0.0, "clinical_question_concept_precision": 0.0,
                "clinical_question_concept_recall": 0.0}
    ps, rs, fs = [], [], []
    for q, ref in zip(questions, references):
        p, r, f, _, _ = extractor.prf(q or "", ref or "")
        ps.append(p); rs.append(r); fs.append(f)
    n = len(fs)
    return {
        "clinical_question_concept_f1": sum(fs) / n,
        "clinical_question_concept_precision": sum(ps) / n,
        "clinical_question_concept_recall": sum(rs) / n,
    }
