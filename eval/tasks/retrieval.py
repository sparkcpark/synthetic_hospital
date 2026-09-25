"""Evidence retrieval task: data loading, baselines (BM25/SapBERT/hybrid), LLM reranking."""

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from eval.prompts import get_prompt
from eval.tasks.diagnosis import _strip_markdown

log = logging.getLogger(__name__)

RRF_K = 60  # Reciprocal Rank Fusion constant

# Module-level SapBERT cache (avoid reloading 768MB model per call)
_sapbert_embedder = None


def _get_sapbert_embedder():
    global _sapbert_embedder
    if _sapbert_embedder is None:
        from etl.ontology.sapbert_embedder import SapBERTEmbedder
        _sapbert_embedder = SapBERTEmbedder()
    return _sapbert_embedder


@dataclass
class Passage:
    passage_id: str
    text: str
    source: str  # ehr_section | fact_card


@dataclass
class RetrievalInput:
    gt_id: int
    patient_id: int
    query: str  # concatenated diagnosis names
    corpus: list[Passage] = field(default_factory=list)
    judgments: dict[str, int] = field(default_factory=dict)  # passage_id → grade (0-3)
    ground_truth: dict = field(default_factory=dict)
    # Ontology hints for structured strategy (preloaded in main thread)
    target_diagnoses_with_codes: list[dict] = field(default_factory=list)
    pathognomonic_findings: list[dict] = field(default_factory=list)


def load_inputs(conn, split: str = "public", granularity: str | None = None,
                pilot: int | None = None) -> list[RetrievalInput]:
    """Load evidence retrieval GT items with passage corpus and relevance judgments."""
    cur = conn.cursor()

    sql = """
        SELECT gt_id, patient_id, ground_truth
        FROM benchmark_ground_truth
        WHERE task = 'evidence_retrieval' AND is_diagnostic AND split = %s
        ORDER BY gt_id
    """
    params = [split]
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)

    cur.execute(sql, params)
    rows = cur.fetchall()
    log.info("Loading %d retrieval GT items (split=%s)", len(rows), split)

    inputs = []
    for gt_id, pid, gt_json in rows:
        gt = gt_json if isinstance(gt_json, dict) else json.loads(gt_json)

        # Build query from diagnosis names + collect ICD-10 codes
        query_dx = gt.get("query_diagnoses", [])
        query_parts = []
        target_dx_with_codes = []
        query_dx_ids = []
        for dx in query_dx:
            if isinstance(dx, dict):
                dx_id = dx.get("diagnosis_id")
                if dx_id:
                    cur.execute("SELECT display_name, icd10_code FROM diagnoses WHERE diagnosis_id = %s", (dx_id,))
                    row = cur.fetchone()
                    if row:
                        query_parts.append(row[0])
                        target_dx_with_codes.append({
                            "name": row[0],
                            "icd10_code": row[1] or "",
                        })
                        query_dx_ids.append(dx_id)
        query = " ".join(query_parts) if query_parts else "clinical diagnosis"

        # Pathognomonic/highly suggestive findings for structured hints
        pathognomonic_findings = []
        if query_dx_ids:
            cur.execute("""
                SELECT DISTINCT cf.display_name, cf.snomed_id
                FROM diagnosis_findings df
                JOIN clinical_findings cf ON df.finding_id = cf.finding_id
                WHERE df.diagnosis_id = ANY(%s)
                  AND df.relationship IN ('pathognomonic', 'highly_suggestive')
                LIMIT 10
            """, (query_dx_ids,))
            pathognomonic_findings = [{"name": r[0], "snomed_id": r[1]}
                                      for r in cur.fetchall()]

        # Build corpus: EHR sections + fact cards
        corpus = _build_corpus(cur, pid, gt)

        # Load relevance judgments
        cur.execute("""
            SELECT passage_id, relevance_grade
            FROM relevance_judgments
            WHERE gt_id = %s
        """, (gt_id,))
        judgments = {pid_str: grade for pid_str, grade in cur.fetchall()}

        inputs.append(RetrievalInput(
            gt_id=gt_id,
            patient_id=pid,
            query=query,
            corpus=corpus,
            judgments=judgments,
            ground_truth=gt,
            target_diagnoses_with_codes=target_dx_with_codes,
            pathognomonic_findings=pathognomonic_findings,
        ))

    log.info("Loaded %d retrieval inputs (avg corpus size: %.0f)",
             len(inputs), np.mean([len(x.corpus) for x in inputs]) if inputs else 0)
    return inputs


def _build_corpus(cur, patient_id: int, gt: dict) -> list[Passage]:
    """Build passage corpus from EHR sections and linked fact cards."""
    passages = []

    # EHR sections (encounter_ehr_sections) — exclude assessment/plan
    cur.execute("""
        SELECT ees.id, ees.section_type, ees.section_text
        FROM encounter_ehr_sections ees
        JOIN longitudinal_encounters le ON ees.encounter_id = le.encounter_id
        WHERE le.patient_id = %s
          AND ees.section_type NOT IN ('assessment', 'plan')
        ORDER BY le.encounter_order, ees.section_order
    """, (patient_id,))
    for sec_id, sec_type, sec_text in cur.fetchall():
        passages.append(Passage(
            passage_id=f"ees_{sec_id}",
            text=f"[{sec_type}] {sec_text}",
            source="ehr_section",
        ))

    # Fact cards linked via diagnoses. The released benchmark database ships chart sections only
    # (retrieval is scored over sections; fact cards are source-derived), so skip when absent.
    cur.execute("SELECT to_regclass('fact_cards')")
    has_fact_cards = cur.fetchone()[0] is not None
    query_dx_ids = [dx.get("diagnosis_id") for dx in gt.get("query_diagnoses", []) if isinstance(dx, dict)]
    if query_dx_ids and has_fact_cards:
        cur.execute("""
            SELECT DISTINCT fc.fact_id, fc.fact_text
            FROM fact_cards fc
            JOIN fact_diagnosis_links fdl ON fc.fact_id = fdl.fact_id
            WHERE fdl.diagnosis_id = ANY(%s)
            LIMIT 200
        """, (query_dx_ids,))
        for fact_id, fact_text in cur.fetchall():
            text = (fact_text or "").strip()
            if text:
                passages.append(Passage(
                    passage_id=f"fc_{fact_id}",
                    text=text,
                    source="fact_card",
                ))

    return passages


# ---------------------------------------------------------------------------
# Non-LLM baselines
# ---------------------------------------------------------------------------

def bm25_retrieve(query: str, corpus: list[Passage], k: int = 20) -> list[dict]:
    """BM25 sparse retrieval baseline."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [p.text.lower().split() for p in corpus]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(ranked_indices):
        results.append({
            "passage_id": corpus[idx].passage_id,
            "score": float(scores[idx]),
            "rank": rank + 1,
        })
    return results


def sapbert_retrieve(query: str, corpus: list[Passage], k: int = 20) -> list[dict]:
    """SapBERT dense retrieval baseline."""
    embedder = _get_sapbert_embedder()

    # Embed query (lowercase required)
    q_emb = embedder.embed_batch([query.lower()])[0]

    # Embed corpus passages (batch for efficiency)
    corpus_texts = [p.text.lower() for p in corpus]
    batch_size = 64
    all_embs = []
    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i:i + batch_size]
        embs = embedder.embed_batch(batch)
        all_embs.append(embs)
    corpus_embs = np.vstack(all_embs)

    # Cosine similarity
    scores = corpus_embs @ q_emb
    ranked_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        results.append({
            "passage_id": corpus[idx].passage_id,
            "score": float(scores[idx]),
            "rank": rank + 1,
        })
    return results


def hybrid_retrieve(query: str, corpus: list[Passage], k: int = 20) -> list[dict]:
    """Hybrid retrieval: BM25 + SapBERT with Reciprocal Rank Fusion."""
    bm25_results = bm25_retrieve(query, corpus, k=50)
    sapbert_results = sapbert_retrieve(query, corpus, k=50)

    # RRF fusion
    rrf_scores: dict[str, float] = {}
    for result_list in [bm25_results, sapbert_results]:
        for item in result_list:
            pid = item["passage_id"]
            rank = item["rank"]
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (RRF_K + rank)

    # Sort by fused score
    sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for rank, (pid, score) in enumerate(sorted_pids):
        results.append({
            "passage_id": pid,
            "score": score,
            "rank": rank + 1,
        })
    return results


# ---------------------------------------------------------------------------
# LLM reranking
# ---------------------------------------------------------------------------

def format_prompt(inp: RetrievalInput, strategy: str) -> tuple[str, str]:
    """Format the prompt for LLM reranking (BM25 top-50 pre-filtered)."""
    template = get_prompt("evidence_retrieval", strategy)

    # Few-shot: load examples or fall back to zero_shot
    few_shot_block = ""
    if strategy == "few_shot":
        from eval.examples import get_few_shot_examples
        few_shot_block = get_few_shot_examples("evidence_retrieval")
        if not few_shot_block:
            template = get_prompt("evidence_retrieval", "zero_shot")

    # Pre-filter with BM25 top-50
    bm25_results = bm25_retrieve(inp.query, inp.corpus, k=50)

    # Build passage text for prompt
    passage_texts = []
    pid_to_passage = {p.passage_id: p for p in inp.corpus}
    for r in bm25_results:
        p = pid_to_passage.get(r["passage_id"])
        if p:
            # Truncate long passages for context window
            text = p.text[:500]
            passage_texts.append(f"[{r['passage_id']}] {text}")

    passages_str = "\n\n".join(passage_texts[:30])  # Limit to 30 for context window

    structured_hints = ""
    if strategy == "structured":
        from eval.hints import format_retrieval_hints
        structured_hints = format_retrieval_hints(
            inp.target_diagnoses_with_codes, inp.pathognomonic_findings,
        )

    user = template.user.format(
        diagnosis=inp.query,
        passages=passages_str,
        few_shot_examples=few_shot_block,
        structured_hints=structured_hints,
    )
    return template.system, user


def parse_output(raw_text: str) -> dict:
    """Parse LLM reranking output into ranked results."""
    text = _strip_markdown(raw_text)

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                return {"rankings": [], "parse_error": "Invalid JSON"}
        else:
            return {"rankings": [], "parse_error": "No JSON found"}

    rankings = data.get("rankings", [])
    normalized = []
    for item in rankings:
        normalized.append({
            "passage_id": item.get("passage_id", ""),
            "grade": int(item.get("grade", 0)),
        })
    return {"rankings": normalized}
