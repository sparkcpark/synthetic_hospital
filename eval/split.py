"""Dataset splitting: stratified patient-level dev/test split.

Assigns benchmark_ground_truth.split = 'val' (dev) or 'test' (held-out).
Split at patient level to prevent data leakage. Singletons split independently.
"""

import json
import logging
import random
from collections import defaultdict

from eval.config import get_pg_connection

log = logging.getLogger(__name__)


def run_split(seed: int = 42, dev_patients: int = 200) -> None:
    """Assign dev/test splits to all benchmark_ground_truth rows.

    Algorithm:
      1. Load patients with difficulty + encounter count
      2. Stratified sample: ~dev_patients → val, rest → test
      3. Map question_ids to patients via source_question_ids
      4. Propagate to all GT rows
      5. Handle singletons independently
    """
    conn = get_pg_connection()
    try:
        _do_split(conn, seed, dev_patients)
        conn.commit()
        log.info("Split committed successfully")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _do_split(conn, seed: int, dev_patients: int) -> None:
    cur = conn.cursor()

    # -----------------------------------------------------------------------
    # 1. Load patient difficulty + encounter counts
    # -----------------------------------------------------------------------
    cur.execute("""
        SELECT bgt.patient_id, bgt.difficulty,
               COALESCE(lp.num_encounters, 0) AS num_enc
        FROM benchmark_ground_truth bgt
        JOIN longitudinal_patients lp ON bgt.patient_id = lp.patient_id
        WHERE bgt.task = 'diagnosis_accuracy'
          AND bgt.granularity = 'patient'
    """)
    patients = []
    for pid, difficulty, num_enc in cur.fetchall():
        # Bucket encounter count: small(2-3), medium(4-5), large(6+)
        if num_enc <= 3:
            enc_bucket = "small"
        elif num_enc <= 5:
            enc_bucket = "medium"
        else:
            enc_bucket = "large"
        patients.append({
            "patient_id": pid,
            "difficulty": difficulty or "medium",
            "enc_bucket": enc_bucket,
            "stratum": f"{difficulty or 'medium'}_{enc_bucket}",
        })

    log.info("Loaded %d patients for splitting", len(patients))

    # -----------------------------------------------------------------------
    # 2. Stratified sampling
    # -----------------------------------------------------------------------
    rng = random.Random(seed)

    # Group by stratum
    strata: dict[str, list] = defaultdict(list)
    for p in patients:
        strata[p["stratum"]].append(p)

    dev_pids: set[int] = set()
    test_pids: set[int] = set()
    total = len(patients)
    dev_ratio = dev_patients / total

    for stratum_name, members in sorted(strata.items()):
        rng.shuffle(members)
        n_dev = max(1, round(len(members) * dev_ratio))
        for p in members[:n_dev]:
            dev_pids.add(p["patient_id"])
        for p in members[n_dev:]:
            test_pids.add(p["patient_id"])

    log.info("Split: %d dev patients, %d test patients", len(dev_pids), len(test_pids))

    # -----------------------------------------------------------------------
    # 3. Build question → patient mapping
    # -----------------------------------------------------------------------
    cur.execute("""
        SELECT patient_id, source_question_ids
        FROM longitudinal_encounters
        WHERE source_question_ids IS NOT NULL
    """)
    qid_to_pid: dict[int, int] = {}
    for pid, sq_ids_str in cur.fetchall():
        try:
            qids = json.loads(sq_ids_str)
            if isinstance(qids, list):
                for qid in qids:
                    qid_to_pid[int(qid)] = pid
        except (json.JSONDecodeError, TypeError):
            pass

    log.info("Mapped %d question_ids to patients", len(qid_to_pid))

    # -----------------------------------------------------------------------
    # 4. Assign splits to all GT rows
    # -----------------------------------------------------------------------

    # 4a. Patient-level GT (dx/patient, summarization, retrieval)
    updated = 0
    for split_val, pids in [("val", dev_pids), ("test", test_pids)]:
        if not pids:
            continue
        pid_list = list(pids)
        cur.execute("""
            UPDATE benchmark_ground_truth
            SET split = %s
            WHERE patient_id = ANY(%s)
              AND granularity = 'patient'
        """, (split_val, pid_list))
        updated += cur.rowcount

    log.info("Updated %d patient-level GT rows", updated)

    # 4b. Encounter-level GT (imaging) — inherit from patient via encounter
    updated = 0
    for split_val, pids in [("val", dev_pids), ("test", test_pids)]:
        if not pids:
            continue
        pid_list = list(pids)
        cur.execute("""
            UPDATE benchmark_ground_truth bgt
            SET split = %s
            FROM longitudinal_encounters le
            WHERE bgt.encounter_id = le.encounter_id
              AND le.patient_id = ANY(%s)
              AND bgt.granularity = 'encounter'
        """, (split_val, pid_list))
        updated += cur.rowcount

    log.info("Updated %d encounter-level GT rows", updated)

    # 4c. Question-level GT — mapped questions inherit patient split
    updated_mapped = 0
    for split_val, pids in [("val", dev_pids), ("test", test_pids)]:
        mapped_qids = [qid for qid, pid in qid_to_pid.items() if pid in pids]
        if not mapped_qids:
            continue
        cur.execute("""
            UPDATE benchmark_ground_truth
            SET split = %s
            WHERE question_id = ANY(%s)
              AND granularity = 'question'
        """, (split_val, mapped_qids))
        updated_mapped += cur.rowcount

    log.info("Updated %d mapped question-level GT rows", updated_mapped)

    # 4d. Singleton questions — split independently by difficulty
    cur.execute("""
        SELECT gt_id, difficulty
        FROM benchmark_ground_truth
        WHERE granularity = 'question'
          AND split IS NULL
    """)
    singletons = cur.fetchall()
    log.info("Found %d singleton questions to split independently", len(singletons))

    if singletons:
        # Group by difficulty, same ratio
        sing_strata: dict[str, list[int]] = defaultdict(list)
        for gt_id, diff in singletons:
            sing_strata[diff or "medium"].append(gt_id)

        for diff, gt_ids in sorted(sing_strata.items()):
            rng.shuffle(gt_ids)
            n_dev = max(1, round(len(gt_ids) * dev_ratio))
            if gt_ids[:n_dev]:
                cur.execute("""
                    UPDATE benchmark_ground_truth
                    SET split = 'val'
                    WHERE gt_id = ANY(%s)
                """, (gt_ids[:n_dev],))
            if gt_ids[n_dev:]:
                cur.execute("""
                    UPDATE benchmark_ground_truth
                    SET split = 'test'
                    WHERE gt_id = ANY(%s)
                """, (gt_ids[n_dev:],))

    # -----------------------------------------------------------------------
    # 5. Verify
    # -----------------------------------------------------------------------
    cur.execute("SELECT COUNT(*) FROM benchmark_ground_truth WHERE split IS NULL")
    null_count = cur.fetchone()[0]
    if null_count > 0:
        raise RuntimeError(f"Split incomplete: {null_count} rows still have NULL split")

    cur.execute("""
        SELECT split, task, granularity, COUNT(*)
        FROM benchmark_ground_truth
        GROUP BY split, task, granularity
        ORDER BY split, task, granularity
    """)
    log.info("Split distribution:")
    for split_val, task, gran, cnt in cur.fetchall():
        log.info("  %s | %s | %s | %d", split_val, task, gran, cnt)


def verify_split() -> None:
    """Verify split integrity: no NULLs, no patient overlap between val/test."""
    conn = get_pg_connection()
    try:
        cur = conn.cursor()

        # Check for NULLs
        cur.execute("SELECT COUNT(*) FROM benchmark_ground_truth WHERE split IS NULL")
        null_count = cur.fetchone()[0]
        if null_count > 0:
            log.error("FAIL: %d rows have NULL split", null_count)
        else:
            log.info("PASS: No NULL splits")

        # Check patient overlap
        cur.execute("""
            SELECT COUNT(DISTINCT patient_id) FROM benchmark_ground_truth
            WHERE patient_id IS NOT NULL AND split = 'val'
        """)
        dev_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(DISTINCT patient_id) FROM benchmark_ground_truth
            WHERE patient_id IS NOT NULL AND split = 'test'
        """)
        test_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT patient_id FROM benchmark_ground_truth
                WHERE patient_id IS NOT NULL AND split = 'val'
                INTERSECT
                SELECT patient_id FROM benchmark_ground_truth
                WHERE patient_id IS NOT NULL AND split = 'test'
            ) overlap
        """)
        overlap = cur.fetchone()[0]
        if overlap > 0:
            log.error("FAIL: %d patients in BOTH val and test", overlap)
        else:
            log.info("PASS: No patient overlap between val and test")

        # Distribution summary
        cur.execute("""
            SELECT split, task, granularity, COUNT(*)
            FROM benchmark_ground_truth
            GROUP BY split, task, granularity
            ORDER BY split, task, granularity
        """)
        log.info("Split distribution:")
        for split_val, task, gran, cnt in cur.fetchall():
            log.info("  %s | %-24s | %-10s | %d", split_val, task, gran, cnt)

        log.info("Dev patients: %d, Test patients: %d", dev_count, test_count)
    finally:
        conn.close()
