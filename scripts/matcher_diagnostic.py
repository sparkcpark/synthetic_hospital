"""Matcher diagnostic: bucket every MISSED primary/relevant finding on a stratified
pilot, so we measure WHY the lexical matcher under-credits rather than generalize
from one row. Per the review:

  Bucket A = raw value vs interpretation ("8 AM cortisol 3 ug/dL" vs "Decreased
             morning cortisol level"). Needs reference-range normalization +
             direction, NOT embedding similarity (SapBERT is polarity-blind here).
  Bucket B = paraphrase/synonym, same meaning & direction. Genuine SapBERT territory.
  Bucket C = actually missing. Stays uncredited.

Outputs per-finding evidence (lexical_match, sapbert_cos, proposed_bucket, best
prediction sentence, blank HUMAN_covered) for adjudication, plus recall under
lexical vs a SapBERT-cascade, and an adversarial polarity demo showing SapBERT
would credit a wrong-direction lab.

Reuses run 287's involved_primary/absent predictions; calls Kimi for 5
involved_relevant rows (the stratum the first-8 pilot missed).
"""
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import psycopg

from epic_sim.app.config import settings
from eval import semantic_match
from eval.adapters import create_adapter
from eval.config import MODEL_REGISTRY
from eval.tasks.diagnosis import _assemble_patient_ehr
from eval.tasks.summarization import SummarizationInput, format_prompt, parse_output

DSN = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")
N_PRIMARY = 6
N_RELEVANT = 6
SAPBERT_TAU = 0.70   # cascade threshold for bucket B (will be calibrated later, not by eye)

DIRECTION = {"elevated", "increased", "raised", "high", "decreased", "reduced", "low",
             "deficiency", "deficient", "insufficiency", "excess", "positive", "negative",
             "abnormal", "loss", "hypo", "hyper", "depressed", "suppressed"}
STOP = {"of", "the", "in", "and", "to", "a", "level", "levels", "result", "results",
        "morning", "test", "value", "with", "without", "sign", "status"}


def _load_input(conn, gt_id, pid, gt):
    return SummarizationInput(
        gt_id=gt_id, patient_id=pid,
        clinical_question=gt.get("clinical_question", ""),
        must_include_findings=[], ehr_text=_assemble_patient_ehr(conn.cursor(), pid),
        ground_truth=gt, variant="specialty_conditioned", specialty=gt.get("specialty", ""))


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.;\n])\s+|\n", text or "") if len(s.strip()) > 3]


def _content_tokens(name):
    return [w for w in re.findall(r"[a-z0-9]+", name.lower())
            if w not in DIRECTION and w not in STOP and len(w) >= 4]


def _bucket(name, summary, sap_cos):
    low = name.lower()
    has_dir = any(d in low for d in DIRECTION) or "level" in low or bool(re.search(r"\d", name))
    analyte_present = any(tok in summary.lower() for tok in _content_tokens(name))
    if has_dir and analyte_present:
        return "A_value_vs_interp"
    if sap_cos >= SAPBERT_TAU:
        return "B_paraphrase"
    return "C_missing"


def main():
    conn = psycopg.connect(DSN)

    # fresh stratified pilot on the CORRECTED GT (the old run 287 is gone)
    def pick(stratum, n):
        q = ("SELECT gt_id, patient_id, ground_truth FROM benchmark_ground_truth "
             "WHERE task='context_summarization' AND granularity='patient' AND split='test' "
             "AND ground_truth->>'eval_stratum'=%s ORDER BY gt_id LIMIT %s")
        return [(g, p, (gt if isinstance(gt, dict) else json.loads(gt)))
                for g, p, gt in conn.execute(q, (stratum, n)).fetchall()]
    items = pick("involved_primary", N_PRIMARY) + pick("involved_relevant", N_RELEVANT)
    print(f"calling Kimi on {len(items)} stratified rows (corrected GT)...")

    adapter = create_adapter(MODEL_REGISTRY["kimi-2.5-thinking"])

    def _call(item):
        gt_id, pid, gt = item
        inp = _load_input(conn, gt_id, pid, gt)
        sysp, user = format_prompt(inp, "zero_shot")
        resp = adapter.call_with_retry(sysp, user)
        return gt_id, pid, gt, parse_output(resp.text).get("summary", "")

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(_call, items))

    involved = [r for r in rows if r[2]["involvement"] == "involved"]
    print(f"involved rows analyzed: {len(involved)}")

    # SapBERT over all findings + all prediction sentences
    from etl.ontology.sapbert_embedder import SapBERTEmbedder
    emb = SapBERTEmbedder()

    out = []
    for gt_id, pid, gt, summary in involved:
        for tier in ("primary", "relevant"):
            names = [f["display_name"] for f in gt["tiers"][tier]]
            if not names:
                continue
            sents = _sentences(summary) or [""]
            ne = emb.embed_batch(names)
            se = emb.embed_batch(sents)
            cos = ne @ se.T   # (findings x sentences), both normalized
            for i, name in enumerate(names):
                lex = semantic_match.phrase_in_text(name, summary or "")
                j = int(cos[i].argmax()); sc = float(cos[i][j])
                bucket = "" if lex else _bucket(name, summary or "", sc)
                out.append({"gt_id": gt_id, "specialty": gt["specialty"], "tier": tier,
                            "finding": name, "lexical_match": int(lex),
                            "sapbert_cos": round(sc, 3),
                            "proposed_bucket": "(lexical hit)" if lex else bucket,
                            "best_pred_sentence": sents[j][:80], "HUMAN_covered": ""})

    with open("matcher_diagnostic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    # summary
    from collections import Counter
    miss = [r for r in out if not r["lexical_match"]]
    bc = Counter(r["proposed_bucket"] for r in miss)
    tot = len(out); lex_hit = sum(r["lexical_match"] for r in out)
    b = bc.get("B_paraphrase", 0)
    print(f"\nfindings: {tot} | lexical hits: {lex_hit} ({lex_hit/tot:.2f}) | missed: {len(miss)}")
    print("missed-finding buckets:", dict(bc))
    print(f"\nrecall(lexical only)         = {lex_hit/tot:.3f}")
    print(f"recall(+B SapBERT-cascade)   = {(lex_hit+b)/tot:.3f}   [+{b} bucket-B paraphrases]")
    print(f"recall(+A normalization too) = {(lex_hit+b+bc.get('A_value_vs_interp',0))/tot:.3f}   "
          f"[+{bc.get('A_value_vs_interp',0)} bucket-A; needs value-normalizer, NOT SapBERT]")

    # adversarial polarity demo: SapBERT can't tell low from high
    print("\n=== adversarial polarity demo (why SapBERT is unsafe on bucket A) ===")
    gtf = "Decreased morning cortisol level"
    probes = ["8 AM cortisol 3 ug/dL (low)", "8 AM cortisol 30 ug/dL (elevated)",
              "no orthostatic hypotension", "orthostatic hypotension present"]
    ge = emb.embed_batch([gtf, "Orthostatic hypotension"])
    pe = emb.embed_batch(probes)
    for k, p in enumerate(probes):
        ref = 0 if "cortisol" in p else 1
        print(f"  cos('{(gtf if ref==0 else 'Orthostatic hypotension')[:28]}', '{p}') = {float(ge[ref]@pe[k]):.3f}")
    print("  -> SapBERT scores the WRONG-direction lab ~as high as the right one: polarity-blind.")
    conn.close()


if __name__ == "__main__":
    main()
