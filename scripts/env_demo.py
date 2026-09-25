"""Smoke test / template for driving the environment with your own policy.

Runs a fixed scripted policy (no LLM) on a few sampled training episodes:
open the chart, list encounters, read the problem list, then submit what the
problem list says. Prints the reward per episode. Replace `scripted_policy`
with a model call that consumes `obs.instructions`, `obs.intro`, `obs.tools`
and the observation texts.

    python scripts/env_demo.py --base-url http://localhost:8000 --task patient_diagnosis --episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.env_client import SyntheticHospitalEnv  # noqa: E402


def scripted_policy(env: SyntheticHospitalEnv, obs):
    """A tiny deterministic policy: read the problem list and submit it as chronic conditions."""
    pid = obs.patient_id
    trace = []
    for name, args in (("open_chart", {"patient_id": pid}),
                       ("view_encounters", {"patient_id": pid}),
                       ("view_problem_list", {"patient_id": pid})):
        o, r, done, info = env.step(name, args)
        trace.append((name, info["remaining"], len(info["observation_text"])))
        problems = o if name == "view_problem_list" and isinstance(o, list) else None

    if obs.task == "patient_diagnosis":
        entries = [{"icd10": p.get("icd10_code") or "", "name": p.get("display_name", ""), "acuity": "chronic"}
                   for p in (problems or []) if p.get("icd10_code")]
        submission = {"active_diagnoses": [], "chronic_conditions": entries}
    elif obs.task == "context_summarization":
        submission = {"summary": "; ".join(p.get("display_name", "") for p in (problems or []))}
    elif obs.task == "evidence_retrieval":
        submission = {"rankings": []}
    else:  # imaging_indication
        submission = {"clinical_question": obs.task_inputs.get("clinical_indication", ""),
                      "pre_read_summary": "", "must_include_findings": [], "differential": []}
    o, reward, done, info = env.step(obs.submit_tool, submission)
    return reward, info, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("SH_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production"))
    ap.add_argument("--task", default="patient_diagnosis")
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with SyntheticHospitalEnv(args.base_url, scorer_token=args.token) as env:
        for i in range(args.episodes):
            obs = env.reset(task=args.task, split=args.split, seed=args.seed + i)
            reward, info, trace = scripted_policy(env, obs)
            print(f"episode {i}: gt_id={obs.gt_id} patient={obs.patient_id} task={obs.task} "
                  f"steps={info['step']} reward={reward:.3f} ({info['reward_metric']})")
            for name, remaining, nchars in trace:
                print(f"    {name:20s} remaining={remaining:2d} obs_chars={nchars}")


if __name__ == "__main__":
    main()
