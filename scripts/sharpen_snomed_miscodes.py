"""Sharpen the SNOMED-miscode flags with SapBERT.

The deterministic token matcher over-flags eponym/synonym pairs (e.g. "Loiasis"
vs "Infection caused by Loa loa" score ~0 but are the same concept). SapBERT
embeds biomedical terms in a space where true synonyms are close, so we re-score
each (display_name, assigned_snomed_term) pair by SapBERT cosine and drop the
high-cosine pairs (synonyms = false positives), keeping the genuine mismatches
(e.g. "Disorder of sex development" vs "Abnormal number of teeth").
"""
import csv

from etl.ontology.sapbert_embedder import SapBERTEmbedder

# cosine >= KEEP_BELOW -> same concept (synonym, drop); < -> genuine mismatch (keep)
KEEP_BELOW = 0.70


def main():
    rows = [r for r in csv.DictReader(open("miscoding_audit.csv"))
            if r["flag_type"] == "snomed_miscode" and r["snomed_term"]]
    print(f"snomed_miscode rows: {len(rows)}")

    emb = SapBERTEmbedder()
    names = [r["display_name"] for r in rows]
    terms = [r["snomed_term"] for r in rows]
    ne = emb.embed_batch(names)
    te = emb.embed_batch(terms)
    cos = (ne * te).sum(axis=1)  # both L2-normalized -> row-wise cosine

    for r, c in zip(rows, cos):
        r["sapbert_cosine"] = round(float(c), 3)

    genuine = sorted((r for r in rows if r["sapbert_cosine"] < KEEP_BELOW),
                     key=lambda r: r["sapbert_cosine"])
    dropped = len(rows) - len(genuine)

    fields = ["diagnosis_id", "display_name", "assigned_icd10", "assigned_specialty",
              "snomed_id", "snomed_term", "sapbert_cosine", "name_similarity"]
    with open("snomed_miscode_sharpened.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(genuine)

    print(f"SapBERT sharpened: {len(rows)} -> {len(genuine)} genuine "
          f"(dropped {dropped} synonym/eponym false positives at cosine>={KEEP_BELOW})")
    print("lowest-cosine (most clearly wrong) examples:")
    for r in genuine[:10]:
        print(f"  cos={r['sapbert_cosine']:.2f}  {r['display_name'][:38]:38s} != {r['snomed_term'][:34]}")
    print("\nWrote snomed_miscode_sharpened.csv")


if __name__ == "__main__":
    main()
