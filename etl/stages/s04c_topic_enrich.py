"""Stage 4c: LLM Topic & Organ System Enrichment for fact cards.

Enriches topic and organ_system for fact_cards where either is NULL.
Uses an Anthropic-compatible LLM gateway (model configurable).

Usage:
    python -m etl.stages.s04c_topic_enrich --pilot           # small pilot subset
    python -m etl.stages.s04c_topic_enrich --all             # all null-topic cards
    python -m etl.stages.s04c_topic_enrich --source <name>   # filter to one source
    python -m etl.stages.s04c_topic_enrich --export-csv      # export results for review
    python -m etl.stages.s04c_topic_enrich --dry-run         # count batches only
"""

import os
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

from etl.config import DB_PATH, DATA_DIR
from etl.db import get_connection
from etl.deck_registry import REGISTRY
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s04c_topic_enrich")


def _source_filter_sql(source_filter: str) -> tuple[str, list]:
    """Build a SQL filename filter from a deck-profile name.

    If `source_filter` names a deck profile, restrict to that profile's filename
    globs; otherwise fall back to a substring match on the filename.
    """
    profile = next((p for p in REGISTRY.profiles if p.name == source_filter), None)
    if profile and profile.filename_globs:
        clauses, params = [], []
        for g in profile.filename_globs:
            clauses.append("sd.filename LIKE ?")
            params.append(g.replace("*", "%").replace("?", "_"))
        return " AND (" + " OR ".join(clauses) + ")", params
    return " AND sd.filename LIKE ?", [f"%{source_filter}%"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic-compatible LLM gateway used to build the benchmark (not released); see README.
GATEWAY_URL = os.environ.get("SH_LLM_GATEWAY_URL", "http://localhost:8080")
MODEL = "kimi-k2.5"
BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
TIMEOUT_SECS = 180.0
MAX_FACT_TEXT_CHARS = 300
STAGE_NAME = "s04c_topic_enrich"

PROMPT_TEMPLATE = """You are a medical education specialist classifying USMLE study flashcards.

For each flashcard below, determine:
1. **subject**: The medical discipline (e.g., "Internal Medicine", "Cardiology", "Neurology", "Psychiatry", "Pediatrics", "Surgery", "OB/GYN", "Emergency Medicine", "Pathology", "Pharmacology", "Biostatistics"). Use standard discipline names.
2. **topic**: The specific clinical topic, condition, drug, or concept (e.g., "Myocardial Infarction", "DiGeorge Syndrome", "Beta Blockers", "Krebs Cycle"). Use standard clinical terminology. Be specific.
3. **organ_system**: The organ system category. Use EXACTLY one of these values:
   Cardiovascular, Respiratory, Gastrointestinal, Renal, Endocrine, Reproductive, Musculoskeletal, Hematologic, Immune, Nervous System, Integumentary, Infectious Diseases, Psychiatric/Behavioral, Ophthalmology, ENT, Biostatistics & Epidemiology, Pharmacology, General Principles, Multisystem

Return ONLY a JSON object (no markdown, no explanation):
{
  "results": {
    "<card_id>": {"subject": "...", "topic": "...", "organ_system": "..."},
    ...
  }
}

Every card must have all three fields populated.

---
CARDS:
"""

# Map LLM output variants to canonical organ_system values
_ORGAN_SYSTEM_NORMALIZE: dict[str, str] = {
    "cardiovascular": "Cardiovascular",
    "cardiovascular system": "Cardiovascular",
    "cardiology": "Cardiovascular",
    "cardiac": "Cardiovascular",
    "respiratory": "Respiratory",
    "respiratory system": "Respiratory",
    "pulmonary": "Respiratory",
    "gastrointestinal": "Gastrointestinal",
    "gastrointestinal & nutrition": "Gastrointestinal",
    "gastrointestinal and nutrition": "Gastrointestinal",
    "gi": "Gastrointestinal",
    "renal": "Renal",
    "renal & urinary system": "Renal",
    "renal & urinary": "Renal",
    "nephrology": "Renal",
    "endocrine": "Endocrine",
    "endocrine system": "Endocrine",
    "endocrinology": "Endocrine",
    "reproductive": "Reproductive",
    "reproductive system": "Reproductive",
    "obstetrics": "Reproductive",
    "gynecology": "Reproductive",
    "ob/gyn": "Reproductive",
    "musculoskeletal": "Musculoskeletal",
    "musculoskeletal system": "Musculoskeletal",
    "msk": "Musculoskeletal",
    "orthopedics": "Musculoskeletal",
    "hematologic": "Hematologic",
    "hematologic system": "Hematologic",
    "hematology": "Hematologic",
    "hematology & oncology": "Hematologic",
    "immune": "Immune",
    "immune system": "Immune",
    "immunology": "Immune",
    "nervous system": "Nervous System",
    "neurology": "Nervous System",
    "neuro": "Nervous System",
    "integumentary": "Integumentary",
    "integumentary system": "Integumentary",
    "dermatology": "Integumentary",
    "infectious diseases": "Infectious Diseases",
    "infectious disease": "Infectious Diseases",
    "id": "Infectious Diseases",
    "psychiatric/behavioral": "Psychiatric/Behavioral",
    "psychiatry": "Psychiatric/Behavioral",
    "psych": "Psychiatric/Behavioral",
    "behavioral": "Psychiatric/Behavioral",
    "ophthalmology": "Ophthalmology",
    "ent": "ENT",
    "otolaryngology": "ENT",
    "biostatistics & epidemiology": "Biostatistics & Epidemiology",
    "biostatistics": "Biostatistics & Epidemiology",
    "epidemiology": "Biostatistics & Epidemiology",
    "pharmacology": "Pharmacology",
    "general principles": "General Principles",
    "multisystem": "Multisystem",
}


def _normalize_organ_system(value: str) -> str:
    """Normalize an organ_system value from LLM output."""
    normalized = _ORGAN_SYSTEM_NORMALIZE.get(value.lower().strip())
    if normalized:
        return normalized
    # If not in map, return as-is (LLM may have returned a canonical value)
    return value.strip()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _format_cards_block(cards: list[dict]) -> str:
    """Format cards for the prompt."""
    lines = []
    for c in cards:
        text = (c["fact_text"] or "")[:MAX_FACT_TEXT_CHARS]
        subject = c.get("subject") or "Unknown"
        lines.append(f"[card_id: {c['fact_id']}] Subject: {subject}\n{text}\n")
    return "\n".join(lines)


def _compute_input_hash(cards: list[dict]) -> str:
    """SHA-256 hash of prompt template + card IDs + truncated text for caching."""
    parts = [STAGE_NAME]
    for c in sorted(cards, key=lambda x: x["fact_id"]):
        parts.append(f"{c['fact_id']}:{(c['fact_text'] or '')[:100]}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _check_cache(conn, input_hash: str) -> dict | None:
    """Check llm_call_log for a cached result with this input hash."""
    row = conn.execute(
        "SELECT output_json FROM llm_call_log WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
        (STAGE_NAME, input_hash),
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0], strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _call_kimi(cards: list[dict], timeout: float = TIMEOUT_SECS) -> tuple[dict, int, int, str]:
    """Send a batch to Kimi k2.5 and return (results_dict, input_tokens, output_tokens, raw_text).

    Raises on HTTP or parse errors (caller handles retry).
    """
    prompt = PROMPT_TEMPLATE + _format_cards_block(cards)

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{GATEWAY_URL}/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

    resp_json = json.loads(resp.text, strict=False)

    # Extract text block (skip thinking blocks)
    raw_text = ""
    for item in resp_json.get("content", []):
        if item.get("type") == "text":
            raw_text = item["text"]
            break

    if not raw_text:
        raise ValueError("No text block found in Kimi response")

    # Extract token usage
    usage = resp_json.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Parse JSON from response text (may be wrapped in markdown code block)
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        # Strip markdown code fences
        lines = json_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_text = "\n".join(lines)

    parsed = json.loads(json_text, strict=False)
    results = parsed.get("results", parsed)

    return results, input_tokens, output_tokens, raw_text


def _call_with_retry(cards: list[dict]) -> tuple[dict, int, int, str]:
    """Call Kimi with exponential backoff retry."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_kimi(cards, timeout=TIMEOUT_SECS + (attempt * 60))
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code == 429:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(f"Rate limited (429), retrying in {delay:.0f}s (attempt {attempt + 1})")
                time.sleep(delay)
            elif e.response.status_code >= 500:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(f"Server error ({e.response.status_code}), retrying in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Parse error: {e}, retrying (attempt {attempt + 1})")
                time.sleep(RETRY_BASE_DELAY)
            else:
                raise
        except httpx.TimeoutException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Timeout, retrying with longer timeout (attempt {attempt + 1})")
                time.sleep(RETRY_BASE_DELAY)
            else:
                raise
    raise last_error


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def _fetch_cards(conn, source_filter: str | None = None) -> list[dict]:
    """Fetch fact_cards needing topic or organ_system enrichment."""
    sql = """
        SELECT fc.fact_id, fc.fact_text, fc.subject, fc.specialty,
               fc.fact_format, sd.filename
        FROM fact_cards fc
        JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
        JOIN source_decks sd ON rc.deck_id = sd.deck_id
        WHERE (fc.topic IS NULL OR fc.organ_system IS NULL OR fc.subject IS NULL)
    """
    params: list = []

    if source_filter:
        frag, fparams = _source_filter_sql(source_filter)
        sql += frag
        params.extend(fparams)

    sql += " ORDER BY fc.fact_id"
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "fact_id": r[0],
            "fact_text": r[1],
            "subject": r[2],
            "specialty": r[3],
            "fact_format": r[4],
            "filename": r[5],
        }
        for r in rows
    ]


def _update_card(conn, fact_id: int, topic: str, organ_system: str,
                  subject: str | None = None):
    """Update a single fact card with enriched topic, organ_system, and optionally subject."""
    if subject:
        conn.execute(
            """UPDATE fact_cards
               SET topic = COALESCE(topic, ?),
                   organ_system = COALESCE(organ_system, ?),
                   subject = COALESCE(subject, ?),
                   extraction_method = 'hybrid'
               WHERE fact_id = ?""",
            (topic, organ_system, subject, fact_id),
        )
    else:
        conn.execute(
            """UPDATE fact_cards
               SET topic = COALESCE(topic, ?),
                   organ_system = COALESCE(organ_system, ?),
                   extraction_method = 'hybrid'
               WHERE fact_id = ?""",
            (topic, organ_system, fact_id),
        )


def _log_llm_call(conn, input_hash: str, input_tokens: int, output_tokens: int,
                   latency_ms: int, output_json: str, raw_response: str,
                   error: str | None = None):
    """Log an LLM call to llm_call_log."""
    conn.execute(
        """INSERT INTO llm_call_log
           (stage, model, prompt_template, input_tokens, output_tokens,
            latency_ms, input_hash, output_json, raw_response, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (STAGE_NAME, MODEL, "topic_organ_classify", input_tokens, output_tokens,
         latency_ms, input_hash, output_json, raw_response, error),
    )


def _log_processing(conn, status: str, records_in: int = 0, records_out: int = 0,
                     records_error: int = 0, duration_sec: float = 0.0,
                     config: str | None = None, error_message: str | None = None):
    """Log to processing_log."""
    if status == "started":
        conn.execute(
            """INSERT INTO processing_log (stage, status, records_in, config)
               VALUES (?, 'started', ?, ?)""",
            (STAGE_NAME, records_in, config),
        )
    else:
        conn.execute(
            """UPDATE processing_log
               SET status = ?, records_out = ?, records_error = ?,
                   duration_sec = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                   error_message = ?
               WHERE log_id = (
                   SELECT log_id FROM processing_log
                   WHERE stage = ? AND status = 'started'
                   ORDER BY log_id DESC LIMIT 1
               )""",
            (status, records_out, records_error, duration_sec, error_message, STAGE_NAME),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def _create_batches(cards: list[dict], batch_size: int = BATCH_SIZE) -> list[list[dict]]:
    """Group cards by subject, then chunk into batches.

    Cards within the same subject are kept together for better LLM context.
    """
    from collections import defaultdict
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        by_subject[c.get("subject") or "Unknown"].append(c)

    batches = []
    current_batch: list[dict] = []
    for subject in sorted(by_subject.keys()):
        group = by_subject[subject]
        for card in group:
            current_batch.append(card)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

def enrich(conn, cards: list[dict], dry_run: bool = False,
           batch_size: int = BATCH_SIZE) -> dict:
    """Process all cards through LLM enrichment.

    Returns summary dict with counts.
    """
    batches = _create_batches(cards, batch_size)
    total_cards = len(cards)
    total_batches = len(batches)

    log.info(f"Cards to enrich: {total_cards}, batches: {total_batches}")

    if dry_run:
        log.info("Dry run — no LLM calls will be made")
        return {"cards_in": total_cards, "batches": total_batches, "dry_run": True}

    _log_processing(conn, "started", records_in=total_cards,
                     config=json.dumps({"batch_size": BATCH_SIZE, "model": MODEL}))

    t0 = time.time()
    enriched = 0
    errors = 0
    cached = 0

    for i, batch in enumerate(batches):
        input_hash = _compute_input_hash(batch)

        # Check cache
        cached_result = _check_cache(conn, input_hash)
        if cached_result:
            cached += 1
            results = cached_result.get("results", cached_result)
            for card in batch:
                fid = str(card["fact_id"])
                if fid in results:
                    r = results[fid]
                    topic = r.get("topic", "").strip()
                    organ = _normalize_organ_system(r.get("organ_system", ""))
                    subj = r.get("subject", "").strip() or None
                    if topic and organ:
                        _update_card(conn, card["fact_id"], topic, organ, subj)
                        enriched += 1
            conn.commit()
            if (i + 1) % 10 == 0:
                log.info(f"Batch {i + 1}/{total_batches} (cached), enriched={enriched}")
            continue

        # Call LLM
        batch_t0 = time.time()
        try:
            results, in_tok, out_tok, raw_text = _call_with_retry(batch)
            latency_ms = int((time.time() - batch_t0) * 1000)

            # Log call
            _log_llm_call(conn, input_hash, in_tok, out_tok, latency_ms,
                          json.dumps({"results": results}), raw_text)

            # Apply results
            for card in batch:
                fid = str(card["fact_id"])
                if fid in results:
                    r = results[fid]
                    topic = r.get("topic", "").strip()
                    organ = _normalize_organ_system(r.get("organ_system", ""))
                    subj = r.get("subject", "").strip() or None
                    if topic and organ:
                        _update_card(conn, card["fact_id"], topic, organ, subj)
                        enriched += 1
                    else:
                        log.warning(f"Empty topic/organ for fact_id={card['fact_id']}")
                        errors += 1
                else:
                    log.warning(f"Missing fact_id={card['fact_id']} in LLM response")
                    errors += 1

            conn.commit()

        except Exception as e:
            latency_ms = int((time.time() - batch_t0) * 1000)
            log.error(f"Batch {i + 1} failed: {e}")
            _log_llm_call(conn, input_hash, 0, 0, latency_ms, None, None, str(e))
            conn.commit()
            errors += len(batch)

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            elapsed = time.time() - t0
            log.info(
                f"Batch {i + 1}/{total_batches} | enriched={enriched} | "
                f"errors={errors} | cached={cached} | elapsed={elapsed:.0f}s"
            )

    duration = time.time() - t0
    _log_processing(conn, "completed", records_out=enriched,
                     records_error=errors, duration_sec=duration)

    summary = {
        "cards_in": total_cards,
        "batches": total_batches,
        "enriched": enriched,
        "errors": errors,
        "cached_batches": cached,
        "duration_sec": round(duration, 1),
    }
    log.info(f"Enrichment complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(conn, output_path: Path, source_filter: str | None = None):
    """Export enriched cards to CSV for review."""
    sql = """
        SELECT fc.fact_id, fc.subject, fc.organ_system, fc.topic, fc.specialty,
               SUBSTR(fc.fact_text, 1, 300) as fact_text_preview,
               fc.extraction_method, sd.filename
        FROM fact_cards fc
        JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
        JOIN source_decks sd ON rc.deck_id = sd.deck_id
        WHERE fc.extraction_method = 'hybrid'
    """
    params: list = []

    if source_filter:
        frag, fparams = _source_filter_sql(source_filter)
        sql += frag
        params.extend(fparams)

    sql += " ORDER BY fc.fact_id"
    rows = conn.execute(sql, params).fetchall()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fact_id", "subject", "organ_system", "topic", "specialty",
                         "fact_text_preview", "extraction_method", "filename"])
        writer.writerows(rows)

    log.info(f"Exported {len(rows)} enriched cards to {output_path}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(conn):
    """Run verification queries and log results."""
    log.info("--- Verification ---")

    # Coverage
    row = conn.execute(
        """SELECT COUNT(*) as total,
                  SUM(CASE WHEN topic IS NULL THEN 1 ELSE 0 END) as null_topic,
                  SUM(CASE WHEN organ_system IS NULL THEN 1 ELSE 0 END) as null_organ,
                  SUM(CASE WHEN extraction_method = 'hybrid' THEN 1 ELSE 0 END) as hybrid
           FROM fact_cards"""
    ).fetchone()
    log.info(f"Total: {row[0]}, NULL topic: {row[1]}, NULL organ: {row[2]}, hybrid: {row[3]}")

    # Per-source
    rows = conn.execute(
        """SELECT sd.filename, COUNT(*) as total,
                  SUM(CASE WHEN fc.topic IS NOT NULL THEN 1 ELSE 0 END) as has_topic,
                  SUM(CASE WHEN fc.organ_system IS NOT NULL THEN 1 ELSE 0 END) as has_organ
           FROM fact_cards fc
           JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
           JOIN source_decks sd ON rc.deck_id = sd.deck_id
           GROUP BY sd.filename ORDER BY total DESC"""
    ).fetchall()
    for r in rows:
        log.info(f"  {r[0]}: {r[1]} total, {r[2]} topic, {r[3]} organ")

    # Top topics
    rows = conn.execute(
        """SELECT topic, COUNT(*) as cnt FROM fact_cards
           WHERE topic IS NOT NULL GROUP BY topic ORDER BY cnt DESC LIMIT 15"""
    ).fetchall()
    log.info("Top 15 topics:")
    for r in rows:
        log.info(f"  {r[0]}: {r[1]}")

    # Organ system distribution
    rows = conn.execute(
        """SELECT organ_system, COUNT(*) as cnt FROM fact_cards
           WHERE organ_system IS NOT NULL GROUP BY organ_system ORDER BY cnt DESC"""
    ).fetchall()
    log.info("Organ system distribution:")
    for r in rows:
        log.info(f"  {r[0]}: {r[1]}")

    # LLM call stats
    row = conn.execute(
        """SELECT COUNT(*) as calls,
                  COALESCE(SUM(input_tokens), 0) as total_in,
                  COALESCE(SUM(output_tokens), 0) as total_out,
                  COALESCE(AVG(latency_ms), 0) as avg_latency,
                  SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
           FROM llm_call_log WHERE stage = ?""",
        (STAGE_NAME,),
    ).fetchone()
    log.info(
        f"LLM calls: {row[0]}, input tokens: {row[1]}, output tokens: {row[2]}, "
        f"avg latency: {row[3]:.0f}ms, errors: {row[4]}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4c: LLM Topic & Organ System Enrichment"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true",
                       help="Enrich a single pilot deck (first configured fact profile)")
    group.add_argument("--all", action="store_true",
                       help="Enrich all null-topic cards")
    group.add_argument("--source", type=str,
                       help="Filter by deck-profile name (see etl/deck_profiles/)")
    group.add_argument("--export-csv", action="store_true",
                       help="Export enriched cards to CSV (no LLM calls)")
    group.add_argument("--verify-only", action="store_true",
                       help="Run verification queries only")

    parser.add_argument("--dry-run", action="store_true",
                        help="Count batches without calling LLM")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()

    batch_size = args.batch_size
    conn = get_connection(args.db)

    if args.verify_only:
        verify(conn)
        conn.close()
        return

    if args.export_csv:
        csv_path = DATA_DIR / f"enriched_topics_{STAGE_NAME}.csv"
        export_csv(conn, csv_path)
        conn.close()
        return

    source_filter = None
    if args.pilot:
        # Pilot = the first configured fact deck profile.
        _fact = next((p for p in REGISTRY.profiles if p.role == "fact"), None)
        source_filter = _fact.name if _fact else None
    elif args.source:
        source_filter = args.source

    cards = _fetch_cards(conn, source_filter)
    if not cards:
        log.info("No cards need enrichment — all topics/organ_systems populated")
        conn.close()
        return

    summary = enrich(conn, cards, dry_run=args.dry_run, batch_size=batch_size)

    if not args.dry_run:
        verify(conn)

        # Auto-export CSV for pilot
        if args.pilot:
            csv_path = DATA_DIR / "enriched_topics_pilot.csv"
            export_csv(conn, csv_path, source_filter=source_filter)

    conn.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
