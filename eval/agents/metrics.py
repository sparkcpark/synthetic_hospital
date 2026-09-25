"""Agent-specific metrics for Phase F evaluation (§F6).

Computes efficiency, information gathering, clinical workflow, and arm comparison metrics.
"""

import logging
from collections import Counter
from itertools import combinations

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)

# Canonical clinical chart review order for Kendall's tau
CANONICAL_SECTION_ORDER = [
    "demographics",
    "chief_complaint",
    "hpi",
    "pmh",
    "psh",
    "medications",
    "allergies",
    "family_history",
    "social_history",
    "ros",
    "vitals",
    "physical_exam",
    "labs",
    "imaging",
    "pathology",
    "other_studies",
    "assessment",
    "plan",
]
SECTION_RANK = {s: i for i, s in enumerate(CANONICAL_SECTION_ORDER)}


# ---------------------------------------------------------------------------
# F6.1 Efficiency Metrics
# ---------------------------------------------------------------------------

def compute_efficiency_metrics(results: list[dict]) -> dict:
    """Compute efficiency metrics across agent results.

    Args:
        results: List of dicts with keys: turns_used, submitted, budget,
                 total_input_tokens, total_output_tokens, total_latency_ms
    """
    if not results:
        return {}

    turns = [r["turns_used"] for r in results]
    budgets = [r.get("budget", 40) for r in results]
    submitted = [r["submitted"] for r in results]

    return {
        "mean_actions_used": float(np.mean(turns)),
        "median_actions_used": float(np.median(turns)),
        "submission_rate": sum(submitted) / len(submitted),
        "budget_utilization": float(np.mean([t / b for t, b in zip(turns, budgets)])),
        "mean_cost_tokens": float(np.mean([
            r["total_input_tokens"] + r["total_output_tokens"] for r in results
        ])),
        "mean_time_ms": float(np.mean([r["total_latency_ms"] for r in results])),
    }


# ---------------------------------------------------------------------------
# F6.2 Information Gathering Metrics
# ---------------------------------------------------------------------------

def compute_info_gathering_metrics(traces: list[list[dict]]) -> dict:
    """Compute information gathering metrics from agent traces.

    Args:
        traces: List of traces (one per item), each a list of trace entries.
    """
    if not traces:
        return {}

    all_unique_sections = []
    all_failure_rates = []
    all_redundancy_rates = []

    for trace in traces:
        # Count unique (encounter, section_type) pairs accessed
        sections_seen = set()
        query_count = 0
        failure_count = 0
        seen_queries = set()
        redundant_count = 0

        for entry in trace:
            if entry.get("action_type") in ("tool", "command"):
                query_count += 1

                # Check for failures (empty output or errors)
                output_len = entry.get("output_length", 0)
                if output_len == 0:
                    failure_count += 1

                # Track sections accessed (from action args)
                args = entry.get("action_args", {})
                if "encounter_id" in args and "section_type" in args:
                    key = (args["encounter_id"], args["section_type"])
                    if key in sections_seen:
                        redundant_count += 1
                    sections_seen.add(key)

                # Track query deduplication
                query_key = str(args)
                if query_key in seen_queries:
                    redundant_count += 1
                seen_queries.add(query_key)

        all_unique_sections.append(len(sections_seen))
        if query_count > 0:
            all_failure_rates.append(failure_count / query_count)
            all_redundancy_rates.append(redundant_count / query_count)

    return {
        "mean_unique_sections_accessed": float(np.mean(all_unique_sections)) if all_unique_sections else 0.0,
        "mean_query_failure_rate": float(np.mean(all_failure_rates)) if all_failure_rates else 0.0,
        "mean_redundant_query_rate": float(np.mean(all_redundancy_rates)) if all_redundancy_rates else 0.0,
    }


# ---------------------------------------------------------------------------
# F6.3 Clinical Workflow Metrics
# ---------------------------------------------------------------------------

def compute_workflow_metrics(traces: list[list[dict]]) -> dict:
    """Compute clinical workflow adherence metrics.

    Measures how closely the agent's section access order matches the
    canonical clinical chart review order using Kendall's tau.
    """
    if not traces:
        return {}

    tau_scores = []
    breadth_depth_ratios = []
    first_actions = Counter()

    for trace in traces:
        # Extract section access order
        section_order = []
        encounter_counts = Counter()

        for i, entry in enumerate(trace):
            if entry.get("action_type") in ("tool", "command"):
                # Categorize first action
                if i == 0:
                    action_name = entry.get("action_name", "unknown")
                    if action_name in ("open_chart", "view_encounters"):
                        first_actions["chart_overview"] += 1
                    elif action_name in ("view_encounter_detail", "view_section"):
                        first_actions["specific_query"] += 1
                    else:
                        first_actions["other"] += 1

                # Track section types for workflow adherence
                args = entry.get("action_args", {})
                if isinstance(args, dict):
                    section = args.get("section_type")
                    if section and section in SECTION_RANK:
                        section_order.append(SECTION_RANK[section])

                    enc = args.get("encounter_id")
                    if enc:
                        encounter_counts[enc] += 1

        # Kendall's tau between agent order and canonical order
        if len(section_order) >= 2:
            ideal = sorted(section_order)
            tau, _ = stats.kendalltau(section_order, ideal)
            if not np.isnan(tau):
                tau_scores.append(tau)

        # Breadth vs depth
        if encounter_counts:
            unique_encounters = len(encounter_counts)
            max_depth = max(encounter_counts.values())
            breadth_depth_ratios.append(unique_encounters / max_depth if max_depth > 0 else 0)

    total_first = sum(first_actions.values()) or 1
    return {
        "mean_workflow_adherence_tau": float(np.mean(tau_scores)) if tau_scores else None,
        "mean_breadth_depth_ratio": float(np.mean(breadth_depth_ratios)) if breadth_depth_ratios else None,
        "first_action_chart_overview_pct": first_actions["chart_overview"] / total_first,
        "first_action_specific_query_pct": first_actions["specific_query"] / total_first,
        "first_action_other_pct": first_actions["other"] / total_first,
    }


# ---------------------------------------------------------------------------
# F6.4 Arm Comparison Metrics
# ---------------------------------------------------------------------------

def compute_arm_comparison(
    structured_scores: list[float],
    bash_scores: list[float],
) -> dict:
    """Compute paired comparison metrics between structured and bash arms.

    Args:
        structured_scores: Per-item primary metric scores for structured arm.
        bash_scores: Per-item primary metric scores for bash arm (same items).
    """
    if not structured_scores or not bash_scores:
        return {}
    if len(structured_scores) != len(bash_scores):
        log.warning("Score lists have different lengths: %d vs %d",
                     len(structured_scores), len(bash_scores))

    n = min(len(structured_scores), len(bash_scores))
    s = np.array(structured_scores[:n])
    b = np.array(bash_scores[:n])

    # Structured advantage
    sa = float(np.mean(s) - np.mean(b))

    # Paired Wilcoxon signed-rank test
    try:
        stat, p_value = stats.wilcoxon(s, b, alternative="two-sided")
    except ValueError:
        # All differences are zero
        stat, p_value = 0.0, 1.0

    # Cohen's d (paired)
    diffs = s - b
    d_mean = np.mean(diffs)
    d_std = np.std(diffs, ddof=1) if len(diffs) > 1 else 1.0
    cohens_d = float(d_mean / d_std) if d_std > 0 else 0.0

    return {
        "structured_advantage": sa,
        "structured_mean": float(np.mean(s)),
        "bash_mean": float(np.mean(b)),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value": float(p_value),
        "cohens_d": cohens_d,
        "n_items": n,
    }


def compute_efficiency_ratio(
    structured_scores: list[float],
    structured_turns: list[int],
    bash_scores: list[float],
    bash_turns: list[int],
) -> float | None:
    """Compute efficiency ratio: (score/actions)_structured / (score/actions)_bash."""
    if not structured_scores or not bash_scores:
        return None

    s_eff = [s / t if t > 0 else 0 for s, t in zip(structured_scores, structured_turns)]
    b_eff = [s / t if t > 0 else 0 for s, t in zip(bash_scores, bash_turns)]

    mean_b = np.mean(b_eff)
    if mean_b == 0:
        return None
    return float(np.mean(s_eff) / mean_b)
