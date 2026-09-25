"""Stage 1d: Ingest PDF documents into fact_cards (standalone).

Converts medical education PDF documents into section-level fact cards,
following the 3-layer insertion pattern: source_decks -> raw_cards -> fact_cards.

Usage:
    python -m etl.stages.s01d_ingest_docs --file <document>.pdf
    python -m etl.stages.s01d_ingest_docs --all
    python -m etl.stages.s01d_ingest_docs --dry-run
    python -m etl.stages.s01d_ingest_docs --verify-only
    python -m etl.stages.s01d_ingest_docs --export-csv
    python -m etl.stages.s01d_ingest_docs --wipe
    python -m etl.stages.s01d_ingest_docs --enrich-topics
"""

import os
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import httpx
import pdfplumber

from etl.config import DB_PATH, PROJECT_ROOT
from etl.db import get_connection
from etl.deck_registry import PDF_REGISTRY
from etl.parsers.pdf_sections import (
    Section,
    extract_section_text,
    parse_pdf_sections,
)
from etl.utils.hashing import sha256_file
from etl.utils.logging import get_logger

log = get_logger("etl.stages.s01d_ingest_docs")

DOC_DIR = PROJECT_ROOT / "doc"
DATA_DIR = PROJECT_ROOT / "data"

# ── PDF metadata ──────────────────────────────────────────────────
# Maps each source PDF filename to a human-readable display name. Replace these
# example entries with your own documents; a file not listed here falls back to
# using its filename as the display name (see _PDF_DECK_NAMES.get below).

_PDF_DECK_NAMES = {
    "example_biostats.pdf": "Example Biostatistics & Social Sciences Notes",
    "example_psychiatry.pdf": "Example Psychiatry Notes",
    "example_obgyn.pdf": "Example Obstetrics & Gynecology Notes",
    "example_pediatrics.pdf": "Example Pediatrics Notes",
    "example_internal_medicine.pdf": "Example Internal Medicine & Surgery Notes",
    "example_step_notes.pdf": "Example Step 2 & 3 Review Notes",
    "example_podcast_notes.pdf": "Example Podcast Lecture Notes",
}

# ── Subject / Organ System Maps ──────────────────────────────────

_SUBJECT_MAP = {
    # Document A: top-level headings
    "CARDIOLOGY": "Cardiology",
    "NEUROLOGY": "Neurology",
    "OPHTHALMOLOGY": "Ophthalmology",
    "OTORHINOLARYNGOLOGY": "ENT",
    "GASTROENTEROLOGY": "Gastroenterology",
    "PULMONOLOGY": "Pulmonology",
    "NEPHROLOGY": "Nephrology",
    "ENDOCRINOLOGY": "Endocrinology",
    "HEMATOLOGY ONCOLOGY": "Hematology/Oncology",
    "INFECTIOUS DISEASE": "Infectious Disease",
    "RHEUMATOLOGY": "Rheumatology",
    "ORTHOPEDICS": "Orthopedics",
    "DERMATOLOGY": "Dermatology",
    "MALE REPRODUCTIVE": "Urology",
    "ENVIRONMENTAL PATHOLOGY": "Emergency Medicine",
    "ANESTHESIA": "Anesthesiology",
    "SURGERY": "Surgery",
    "NUTRITION": "General Principles",
    # Document set B: specialty documents
    "BIOSTATS & SOCIAL SCIENCES": "Biostatistics & Epidemiology",
    "PSYCHIATRY": "Psychiatry",
    "OBSTETRICS & GYNECOLOGY": "Obstetrics/Gynecology",
    "PEDIATRICS": "Pediatrics",
    # Document C: chapter headings
    "Renal": "Nephrology",
    "Pulmonary": "Pulmonology",
    "CVS": "Cardiology",
    "Endocrine": "Endocrinology",
    "Gynae & Reproductive Health": "Obstetrics/Gynecology",
    "Obs": "Obstetrics/Gynecology",
    "GIT": "Gastroenterology",
    "Blood & Oncology": "Hematology/Oncology",
    "Musculoskeletal": "Orthopedics",
    "Dermatology": "Dermatology",
    "CNS, Eye, Ear": "Neurology",
    "Psychiatry & Substance Abuse": "Psychiatry",
    "Short Subjects": "General Principles",
    "General Paeds": "Pediatrics",
    "Geriatrics": "Internal Medicine",
    "Public Health": "Biostatistics & Epidemiology",
    "Ethics & Communication": "Biostatistics & Epidemiology",
    # Document D: parent-topic keywords
    "INTERNAL MEDICINE": "Internal Medicine",
    "EMERGENCY MEDICINE": "Emergency Medicine",
    "PHARMACOLOGY": "Pharmacology",
    "RADIOLOGY": "Radiology",
    "GENETICS": "Genetics",
    "IMMUNE": "Immunology",
    "GENERAL PRINCIPLES": "General Principles",
}

_ORGAN_SYSTEM_MAP = {
    # Document A: top-level
    "CARDIOLOGY": "Cardiovascular",
    "NEUROLOGY": "Nervous System",
    "OPHTHALMOLOGY": "Ophthalmology",
    "OTORHINOLARYNGOLOGY": "ENT",
    "GASTROENTEROLOGY": "Gastrointestinal",
    "PULMONOLOGY": "Respiratory",
    "NEPHROLOGY": "Renal",
    "ENDOCRINOLOGY": "Endocrine",
    "HEMATOLOGY ONCOLOGY": "Hematologic",
    "INFECTIOUS DISEASE": "Infectious Diseases",
    "RHEUMATOLOGY": "Musculoskeletal",
    "ORTHOPEDICS": "Musculoskeletal",
    "DERMATOLOGY": "Integumentary",
    "MALE REPRODUCTIVE": "Reproductive",
    "ENVIRONMENTAL PATHOLOGY": "Multisystem",
    "ANESTHESIA": "Multisystem",
    "SURGERY": "Multisystem",
    "NUTRITION": "General Principles",
    # Document set B: specialty documents
    "BIOSTATS & SOCIAL SCIENCES": "Biostatistics & Epidemiology",
    "PSYCHIATRY": "Psychiatric/Behavioral",
    "OBSTETRICS & GYNECOLOGY": "Reproductive",
    "PEDIATRICS": "Multisystem",
    # Document C: chapter headings
    "Renal": "Renal",
    "Pulmonary": "Respiratory",
    "CVS": "Cardiovascular",
    "Endocrine": "Endocrine",
    "Gynae & Reproductive Health": "Reproductive",
    "Obs": "Reproductive",
    "GIT": "Gastrointestinal",
    "Blood & Oncology": "Hematologic",
    "Musculoskeletal": "Musculoskeletal",
    "Dermatology": "Integumentary",
    "CNS, Eye, Ear": "Nervous System",
    "Psychiatry & Substance Abuse": "Psychiatric/Behavioral",
    "Short Subjects": "Multisystem",
    "General Paeds": "Multisystem",
    "Geriatrics": "Multisystem",
    "Public Health": "Biostatistics & Epidemiology",
    "Ethics & Communication": "Biostatistics & Epidemiology",
    # Document D: parent-topic keywords
    "INTERNAL MEDICINE": "Multisystem",
    "EMERGENCY MEDICINE": "Multisystem",
    "PHARMACOLOGY": "Pharmacology",
    "RADIOLOGY": "Multisystem",
    "GENETICS": "General Principles",
    "IMMUNE": "Immune",
    "GENERAL PRINCIPLES": "General Principles",
}


# ── Database Cleanup ─────────────────────────────────────────────

def _wipe_pdf_data(conn) -> dict:
    """Delete all PDF-derived data in reverse FK order."""
    fact_count = conn.execute(
        """SELECT COUNT(*) FROM fact_cards WHERE raw_card_id IN (
               SELECT raw_card_id FROM raw_cards WHERE deck_id IN (
                   SELECT deck_id FROM source_decks WHERE filename LIKE '%.pdf'))"""
    ).fetchone()[0]
    raw_count = conn.execute(
        """SELECT COUNT(*) FROM raw_cards WHERE deck_id IN (
               SELECT deck_id FROM source_decks WHERE filename LIKE '%.pdf')"""
    ).fetchone()[0]
    deck_count = conn.execute(
        "SELECT COUNT(*) FROM source_decks WHERE filename LIKE '%.pdf'"
    ).fetchone()[0]

    log.info(f"Wiping PDF data: {fact_count} fact_cards, {raw_count} raw_cards, "
             f"{deck_count} source_decks")

    conn.execute(
        """DELETE FROM fact_cards WHERE raw_card_id IN (
               SELECT raw_card_id FROM raw_cards WHERE deck_id IN (
                   SELECT deck_id FROM source_decks WHERE filename LIKE '%.pdf'))"""
    )
    conn.execute(
        """DELETE FROM raw_cards WHERE deck_id IN (
               SELECT deck_id FROM source_decks WHERE filename LIKE '%.pdf')"""
    )
    conn.execute("DELETE FROM source_decks WHERE filename LIKE '%.pdf'")
    conn.execute("DELETE FROM processing_log WHERE stage = 's01d_ingest_docs'")
    conn.commit()

    log.info("Wipe complete")
    return {
        "fact_cards_deleted": fact_count,
        "raw_cards_deleted": raw_count,
        "decks_deleted": deck_count,
    }


# ── Ingestion Pipeline ───────────────────────────────────────────

def _ingest_pdf(conn, pdf_path: Path, dry_run: bool = False) -> dict:
    """Ingest a single PDF into the 3-layer schema.

    Returns dict with counts: sections_found, sections_inserted, skipped.
    """
    filename = pdf_path.name
    profile = PDF_REGISTRY.resolve(filename)
    deck_name = profile.display_name if (profile and profile.display_name) else \
        _PDF_DECK_NAMES.get(filename, filename)
    # Metadata behavior is declared on the PDF profile (falls back to defaults).
    topic_from = profile.topic_from if profile else "section_title"
    context_style = profile.context_style if profile else "hierarchy"

    log.info(f"Processing {filename} ({deck_name})")

    # Check idempotency
    existing = conn.execute(
        "SELECT deck_id FROM source_decks WHERE filename = ?",
        (filename,)
    ).fetchone()
    if existing:
        log.warning(f"  {filename} already ingested (deck_id={existing[0]}), skipping")
        return {"sections_found": 0, "sections_inserted": 0, "skipped": 0,
                "already_exists": True}

    # Open PDF and parse sections
    pdf = pdfplumber.open(str(pdf_path))
    sections = parse_pdf_sections(str(pdf_path), pdf)

    if dry_run:
        log.info(f"  [DRY RUN] {len(sections)} sections found")
        for i, s in enumerate(sections):
            if s.text:
                words = len(s.text.split())
            else:
                text = extract_section_text(pdf, s.page_start, s.page_end, s.source_type)
                words = len(text.split())
            ep_info = f" ep={s.episode_title[:40]}" if s.episode_title else ""
            log.info(f"    {i+1:3d}. [{s.parent or 'ROOT'}] {s.title[:60]} "
                     f"(pp {s.page_start}-{s.page_end-1}, ~{words} words){ep_info}")
        pdf.close()
        return {"sections_found": len(sections), "sections_inserted": 0,
                "skipped": 0, "dry_run": True}

    # Insert source_deck
    file_hash = sha256_file(pdf_path)
    conn.execute(
        """INSERT INTO source_decks
           (filename, deck_name, deck_type, note_count, card_count, sha256_hash)
           VALUES (?, ?, 'fact', ?, ?, ?)""",
        (filename, deck_name, len(sections), len(sections), file_hash),
    )
    deck_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    log.info(f"  Inserted source_deck: deck_id={deck_id}")

    # Insert sections as raw_cards + fact_cards
    inserted = 0
    skipped = 0

    for seq, section in enumerate(sections, start=1):
        # Get text: use section.text if available, else extract from pages
        if section.text:
            text = section.text
        else:
            text = extract_section_text(pdf, section.page_start, section.page_end,
                                        section.source_type)

        # Skip guard: too short
        if len(text.strip()) < 20:
            log.debug(f"  Skipping section '{section.title}' (text too short: {len(text)} chars)")
            skipped += 1
            continue

        # Build field data
        field_obj = {
            "Text": text,
            "Section": section.title,
            "Parent": section.parent or "",
        }
        if section.episode_title:
            field_obj["Episode"] = section.episode_title
        if section.caps_header:
            field_obj["CapsHeader"] = section.caps_header
        field_json = json.dumps(field_obj, ensure_ascii=False)

        # Insert raw_card
        conn.execute(
            """INSERT INTO raw_cards
               (deck_id, anki_note_id, field_data, field_data_text,
                card_type, card_format, classification_method)
               VALUES (?, ?, ?, ?, 'fact', 'basic_fact', 'rule_based')""",
            (deck_id, seq, field_json, field_json),
        )
        raw_card_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Derive metadata
        parent_key = section.parent
        subject = _SUBJECT_MAP.get(parent_key) if parent_key else None
        organ_system = _ORGAN_SYSTEM_MAP.get(parent_key) if parent_key else None

        # Topic: NULL for docs whose profile defers topics to LLM enrichment.
        if topic_from == "llm":
            topic = None
        else:
            topic = section.title

        # Context
        if context_style == "episode":
            ctx_parts = [section.episode_title or section.parent or ""]
            if section.caps_header:
                ctx_parts.append(section.caps_header)
            context = ": ".join(filter(None, ctx_parts))
        else:
            context = f"{section.parent}: {section.title}" if section.parent else section.title

        # Insert fact_card
        conn.execute(
            """INSERT INTO fact_cards
               (raw_card_id, fact_format, fact_text, fact_context,
                subject, organ_system, topic, extraction_method)
               VALUES (?, 'basic_fact', ?, ?, ?, ?, ?, 'rule_based')""",
            (raw_card_id, text, context, subject, organ_system, topic),
        )
        inserted += 1

    conn.commit()
    pdf.close()

    log.info(f"  Done: {inserted} inserted, {skipped} skipped")
    return {"sections_found": len(sections), "sections_inserted": inserted,
            "skipped": skipped}


def run(conn, files: list[str] | None = None, dry_run: bool = False) -> dict:
    """Main entry point: ingest PDF documents.

    Args:
        conn: SQLite connection
        files: list of PDF filenames to process (None = all)
        dry_run: if True, only count sections without inserting
    """
    t0 = time.time()

    # Log start
    if not dry_run:
        conn.execute(
            """INSERT INTO processing_log (stage, status, config)
               VALUES ('s01d_ingest_docs', 'started', ?)""",
            (json.dumps({"files": files or "all", "dry_run": dry_run}),),
        )
        conn.commit()

    # Determine which PDFs to process
    if files:
        pdf_paths = []
        for f in files:
            p = DOC_DIR / f
            if p.exists():
                pdf_paths.append(p)
            else:
                log.error(f"File not found: {p}")
    else:
        pdf_paths = sorted(DOC_DIR.glob("*.pdf"))

    if not pdf_paths:
        log.error("No PDF files found to process")
        return {"total_inserted": 0}

    log.info(f"Processing {len(pdf_paths)} PDF files")

    total_found = 0
    total_inserted = 0
    total_skipped = 0
    results_by_file = {}

    for pdf_path in pdf_paths:
        result = _ingest_pdf(conn, pdf_path, dry_run=dry_run)
        results_by_file[pdf_path.name] = result
        total_found += result["sections_found"]
        total_inserted += result["sections_inserted"]
        total_skipped += result["skipped"]

    elapsed = time.time() - t0

    # Log completion
    if not dry_run:
        conn.execute(
            """INSERT INTO processing_log
               (stage, status, records_in, records_out, records_error, duration_sec)
               VALUES ('s01d_ingest_docs', 'completed', ?, ?, ?, ?)""",
            (total_found, total_inserted, total_skipped, elapsed),
        )
        conn.commit()

    log.info(f"=== Stage 1d Complete ===")
    log.info(f"  PDFs processed: {len(pdf_paths)}")
    log.info(f"  Sections found: {total_found}")
    log.info(f"  Facts inserted: {total_inserted}")
    log.info(f"  Skipped: {total_skipped}")
    log.info(f"  Duration: {elapsed:.1f}s")

    return {
        "total_found": total_found,
        "total_inserted": total_inserted,
        "total_skipped": total_skipped,
        "duration_sec": elapsed,
        "by_file": results_by_file,
    }


# ── PDF LLM Topic Enrichment (for docs with no reliable section titles) ──

# Anthropic-compatible LLM gateway used to build the benchmark (not released); see README.
GATEWAY_URL = os.environ.get("SH_LLM_GATEWAY_URL", "http://localhost:8080")
MODEL = "kimi-k2.5"
ENRICH_BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
TIMEOUT_SECS = 180.0
MAX_FACT_TEXT_CHARS = 400
ENRICH_STAGE = "s01d_pdf_topic"

_TOPIC_PROMPT = """You are a medical education specialist classifying USMLE study material.

For each medical bullet point below, determine the specific clinical topic
(e.g. "Intussusception", "Down Syndrome", "Beta Blockers", "Aortic Stenosis").
Use standard clinical terminology. Be specific but concise.

Return ONLY a JSON object (no markdown, no explanation):
{
  "results": {
    "<id>": {"topic": "..."},
    ...
  }
}

Every entry must have a topic populated.

---
BULLETS:
"""


def _enrich_pdf_topics(conn, dry_run: bool = False) -> dict:
    """Enrich PDF fact_cards (profiles with topic_from: llm) with LLM-derived topics."""
    # Fetch enrichable PDF cards with NULL topic. Enrichable = the card's source
    # PDF profile declares topic_from: llm (see PDF_REGISTRY.enrichment_globs).
    globs = PDF_REGISTRY.enrichment_globs()
    if not globs:
        log.info("No PDF profiles request topic enrichment")
        return {"cards_in": 0, "enriched": 0}
    like_clause = " OR ".join("sd.filename LIKE ?" for _ in globs)
    like_params = [g.replace("*", "%").replace("?", "_") for g in globs]
    rows = conn.execute(
        f"""SELECT fc.fact_id, fc.fact_text, fc.subject
           FROM fact_cards fc
           JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
           JOIN source_decks sd ON rc.deck_id = sd.deck_id
           WHERE ({like_clause}) AND fc.topic IS NULL
           ORDER BY fc.fact_id""",
        like_params,
    ).fetchall()

    cards = [{"fact_id": r[0], "fact_text": r[1], "subject": r[2]} for r in rows]
    if not cards:
        log.info("No PDF cards need topic enrichment")
        return {"cards_in": 0, "enriched": 0}

    # Create batches
    batches = [cards[i:i + ENRICH_BATCH_SIZE]
               for i in range(0, len(cards), ENRICH_BATCH_SIZE)]

    log.info(f"PDF topic enrichment: {len(cards)} cards, {len(batches)} batches")

    if dry_run:
        log.info("[DRY RUN] No LLM calls will be made")
        return {"cards_in": len(cards), "batches": len(batches), "dry_run": True}

    # Log start
    conn.execute(
        """INSERT INTO processing_log (stage, status, records_in, config)
           VALUES (?, 'started', ?, ?)""",
        (ENRICH_STAGE, len(cards),
         json.dumps({"batch_size": ENRICH_BATCH_SIZE, "model": MODEL})),
    )
    conn.commit()

    t0 = time.time()
    enriched = 0
    errors = 0
    cached = 0

    for i, batch in enumerate(batches):
        # Compute cache key
        hash_parts = [ENRICH_STAGE]
        for c in sorted(batch, key=lambda x: x["fact_id"]):
            hash_parts.append(f"{c['fact_id']}:{(c['fact_text'] or '')[:100]}")
        input_hash = hashlib.sha256("\n".join(hash_parts).encode()).hexdigest()

        # Check cache
        cached_row = conn.execute(
            "SELECT output_json FROM llm_call_log "
            "WHERE stage = ? AND input_hash = ? AND error IS NULL LIMIT 1",
            (ENRICH_STAGE, input_hash),
        ).fetchone()
        if cached_row and cached_row[0]:
            try:
                cached_result = json.loads(cached_row[0], strict=False)
                results = cached_result.get("results", cached_result)
                for card in batch:
                    fid = str(card["fact_id"])
                    if fid in results:
                        topic = results[fid].get("topic", "").strip()
                        if topic:
                            conn.execute(
                                "UPDATE fact_cards SET topic = ?, extraction_method = 'hybrid' "
                                "WHERE fact_id = ?",
                                (topic, card["fact_id"]),
                            )
                            enriched += 1
                conn.commit()
                cached += 1
                if (i + 1) % 10 == 0:
                    log.info(f"Batch {i+1}/{len(batches)} (cached), enriched={enriched}")
                continue
            except (json.JSONDecodeError, TypeError):
                pass

        # Format prompt
        lines = []
        for c in batch:
            text = (c["fact_text"] or "")[:MAX_FACT_TEXT_CHARS]
            subject = c.get("subject") or "Unknown"
            lines.append(f"[id: {c['fact_id']}] Subject: {subject}\n{text}\n")
        prompt = _TOPIC_PROMPT + "\n".join(lines)

        # Call Kimi with retry
        batch_t0 = time.time()
        last_error = None
        results = None

        for attempt in range(MAX_RETRIES):
            try:
                timeout = TIMEOUT_SECS + (attempt * 60)
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
                    raise ValueError("No text block in response")

                usage = resp_json.get("usage", {})
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)

                # Parse JSON
                json_text = raw_text.strip()
                if json_text.startswith("```"):
                    json_lines = json_text.split("\n")
                    json_lines = [l for l in json_lines if not l.strip().startswith("```")]
                    json_text = "\n".join(json_lines)

                parsed = json.loads(json_text, strict=False)
                results = parsed.get("results", parsed)

                latency_ms = int((time.time() - batch_t0) * 1000)

                # Log call
                conn.execute(
                    """INSERT INTO llm_call_log
                       (stage, model, prompt_template, input_tokens, output_tokens,
                        latency_ms, input_hash, output_json, raw_response)
                       VALUES (?, ?, 'pdf_topic_classify', ?, ?, ?, ?, ?, ?)""",
                    (ENRICH_STAGE, MODEL, in_tok, out_tok, latency_ms,
                     input_hash, json.dumps({"results": results}), raw_text),
                )
                break  # success

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503):
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    log.warning(f"HTTP {e.response.status_code}, retry in {delay:.0f}s")
                    time.sleep(delay)
                else:
                    raise
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"Parse error: {e}, retrying")
                    time.sleep(RETRY_BASE_DELAY)
                else:
                    raise
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"Timeout, retrying")
                    time.sleep(RETRY_BASE_DELAY)
                else:
                    raise

        if results is None:
            latency_ms = int((time.time() - batch_t0) * 1000)
            log.error(f"Batch {i+1} failed after retries: {last_error}")
            conn.execute(
                """INSERT INTO llm_call_log
                   (stage, model, prompt_template, input_tokens, output_tokens,
                    latency_ms, input_hash, error)
                   VALUES (?, ?, 'pdf_topic_classify', 0, 0, ?, ?, ?)""",
                (ENRICH_STAGE, MODEL, latency_ms, input_hash, str(last_error)),
            )
            conn.commit()
            errors += len(batch)
            continue

        # Apply results
        for card in batch:
            fid = str(card["fact_id"])
            if fid in results:
                topic = results[fid].get("topic", "").strip()
                if topic:
                    conn.execute(
                        "UPDATE fact_cards SET topic = ?, extraction_method = 'hybrid' "
                        "WHERE fact_id = ?",
                        (topic, card["fact_id"]),
                    )
                    enriched += 1
                else:
                    log.warning(f"Empty topic for fact_id={card['fact_id']}")
                    errors += 1
            else:
                log.warning(f"Missing fact_id={card['fact_id']} in LLM response")
                errors += 1

        conn.commit()

        if (i + 1) % 10 == 0 or (i + 1) == len(batches):
            elapsed = time.time() - t0
            log.info(
                f"Batch {i+1}/{len(batches)} | enriched={enriched} | "
                f"errors={errors} | cached={cached} | elapsed={elapsed:.0f}s"
            )

    duration = time.time() - t0

    # Log completion
    conn.execute(
        """UPDATE processing_log
           SET status = 'completed', records_out = ?, records_error = ?,
               duration_sec = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
           WHERE log_id = (
               SELECT log_id FROM processing_log
               WHERE stage = ? AND status = 'started'
               ORDER BY log_id DESC LIMIT 1
           )""",
        (enriched, errors, duration, ENRICH_STAGE),
    )
    conn.commit()

    summary = {
        "cards_in": len(cards),
        "batches": len(batches),
        "enriched": enriched,
        "errors": errors,
        "cached_batches": cached,
        "duration_sec": round(duration, 1),
    }
    log.info(f"PDF topic enrichment complete: {summary}")
    return summary


# ── Verification & Export ─────────────────────────────────────────

def verify(conn) -> None:
    """Print verification queries for document-ingested fact cards."""
    log.info("=== Verification ===")

    # Source decks
    rows = conn.execute(
        """SELECT filename, deck_name, note_count
           FROM source_decks WHERE filename LIKE '%.pdf'
           ORDER BY filename"""
    ).fetchall()
    log.info(f"PDF source decks: {len(rows)}")
    for r in rows:
        log.info(f"  {r[0]:30s} {r[1]:50s} notes={r[2]}")

    # Fact card counts per source
    rows = conn.execute(
        """SELECT sd.filename, COUNT(*) as facts,
                  SUM(CASE WHEN fc.subject IS NOT NULL THEN 1 ELSE 0 END) as has_subj,
                  SUM(CASE WHEN fc.organ_system IS NOT NULL THEN 1 ELSE 0 END) as has_organ,
                  SUM(CASE WHEN fc.topic IS NOT NULL THEN 1 ELSE 0 END) as has_topic
           FROM fact_cards fc
           JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
           JOIN source_decks sd ON rc.deck_id = sd.deck_id
           WHERE sd.filename LIKE '%.pdf'
           GROUP BY sd.filename
           ORDER BY sd.filename"""
    ).fetchall()
    log.info("Fact cards per PDF source:")
    for r in rows:
        log.info(f"  {r[0]:30s} facts={r[1]:4d}  subj={r[2]:4d}  organ={r[3]:4d}  topic={r[4]:4d}")

    # Total fact cards
    total = conn.execute("SELECT COUNT(*) FROM fact_cards").fetchone()[0]
    pdf_total = conn.execute(
        """SELECT COUNT(*) FROM fact_cards fc
           JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
           JOIN source_decks sd ON rc.deck_id = sd.deck_id
           WHERE sd.filename LIKE '%.pdf'"""
    ).fetchone()[0]
    log.info(f"Total fact cards: {total} (APKG: {total - pdf_total}, PDF: {pdf_total})")

    # APKG integrity check
    apkg_count = conn.execute(
        """SELECT COUNT(*) FROM fact_cards fc
           JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
           JOIN source_decks sd ON rc.deck_id = sd.deck_id
           WHERE sd.filename NOT LIKE '%.pdf'"""
    ).fetchone()[0]
    log.info(f"APKG fact cards (must be 30,361): {apkg_count}")

    # Sample cards
    rows = conn.execute(
        """SELECT fc.fact_id, fc.subject, fc.organ_system, fc.topic,
                  SUBSTR(fc.fact_text, 1, 100)
           FROM fact_cards fc
           JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
           JOIN source_decks sd ON rc.deck_id = sd.deck_id
           WHERE sd.filename LIKE '%.pdf'
           ORDER BY RANDOM() LIMIT 10"""
    ).fetchall()
    log.info("Sample document-derived facts:")
    for r in rows:
        log.info(f"  [{r[0]}] {r[1]} / {r[2]} / {r[3]}")
        log.info(f"    {r[4]}...")


def export_csv(conn, output_path: Path | None = None) -> None:
    """Export document-derived fact cards to CSV."""
    if output_path is None:
        output_path = DATA_DIR / "doc_fact_cards.csv"

    rows = conn.execute(
        """SELECT fc.fact_id, sd.filename, fc.subject, fc.organ_system, fc.topic,
                  fc.fact_context, LENGTH(fc.fact_text) as text_len,
                  SUBSTR(fc.fact_text, 1, 200) as text_preview
           FROM fact_cards fc
           JOIN raw_cards rc ON fc.raw_card_id = rc.raw_card_id
           JOIN source_decks sd ON rc.deck_id = sd.deck_id
           WHERE sd.filename LIKE '%.pdf'
           ORDER BY sd.filename, fc.fact_id"""
    ).fetchall()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fact_id", "source", "subject", "organ_system", "topic",
                         "context", "text_len", "text_preview"])
        writer.writerows(rows)

    log.info(f"Exported {len(rows)} rows to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1d: Ingest PDF documents into fact_cards"
    )
    parser.add_argument("--file", help="Single PDF filename to process")
    parser.add_argument("--all", action="store_true", help="Process all PDFs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count sections without inserting")
    parser.add_argument("--verify-only", action="store_true",
                        help="Print verification queries only")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--wipe", action="store_true",
                        help="Wipe all PDF-derived data before processing")
    parser.add_argument("--enrich-topics", action="store_true",
                        help="Run LLM topic enrichment for PDFs whose profile defers topics")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Database path (default: data/benchmark.db)")

    args = parser.parse_args()
    conn = get_connection(args.db)

    if args.wipe:
        result = _wipe_pdf_data(conn)
        log.info(f"Wipe result: {result}")
    elif args.verify_only:
        verify(conn)
    elif args.export_csv:
        export_csv(conn)
    elif args.enrich_topics:
        result = _enrich_pdf_topics(conn, dry_run=args.dry_run)
        log.info(f"Enrichment result: {result}")
        verify(conn)
    elif args.file:
        run(conn, files=[args.file], dry_run=args.dry_run)
        if not args.dry_run:
            verify(conn)
    elif args.all:
        run(conn, dry_run=args.dry_run)
        if not args.dry_run:
            verify(conn)
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
