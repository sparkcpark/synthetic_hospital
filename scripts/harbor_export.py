"""Export benchmark instances as Harbor tasks (https://harborframework.com).

One task directory per instance:

    <out>/sh-<task>-<gt_id>/
    ├── task.toml                  metadata, timeouts, the scorer token for the verifier and the oracle
    ├── instruction.md             the paper's agent prompt + patient assignment + how to act
    ├── environment/
    │   ├── Dockerfile             agent container (`main`): shell + sh-agent CLI
    │   ├── agent_cli.py
    │   └── docker-compose.yaml    app (simulator, autostarts this instance's episode), postgres, redis
    ├── solution/solve.sh          oracle: submits the label-derived answer
    └── tests/test.sh              verifier: reads the episode outcome, writes /logs/verifier/reward.json

Tasks reference the published simulator image (database baked in), pulled automatically:
    ghcr.io/sparkcpark/synthetic-hospital:1.3-data
To use a locally built image instead, build it under that name or pass --image / SH_IMAGE:
    docker build --target with-data -t ghcr.io/sparkcpark/synthetic-hospital:1.3-data .

Usage (against a running stack, which supplies the per-instance prompt inputs):
    docker compose up -d
    python scripts/harbor_export.py --out harbor_tasks --task patient_diagnosis --split train --limit 50
    uvx harbor run -p harbor_tasks -a oracle          # sanity check: every task should score 1.0

A fresh scorer token is generated per export and written into every task.toml
([verifier].env, [solution].env) and compose file (app service). Keep the
exported directory private if the token matters to you; regenerate with
--token to rotate.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.agents.prompts import build_patient_intro, build_system_prompt  # noqa: E402
from eval.config import EVAL_TASKS, get_pg_connection  # noqa: E402
from eval.score_one import list_instances  # noqa: E402
from epic_sim.app.services.env_service import _load_context, submit_tool_for  # noqa: E402

TEMPLATES = ROOT / "harbor" / "templates"

COMPOSE = """\
# Harbor multi-container environment. Harbor supplies the `main` (agent) service
# from ./Dockerfile; we only add its dependency on the simulator. postgres and
# redis sit on an internal network the agent container cannot reach, so the
# labels in the database are not accessible from the agent's shell.
services:
  main:
    depends_on:
      app:
        condition: service_healthy

  app:
    image: {image}
    environment:
      EPIC_SIM_AUTOSTART_GT_ID: "{gt_id}"
      EPIC_SIM_AUTOSTART_BUDGET: "{budget}"
      EPIC_SIM_SCORER_TOKEN: "{token}"
      EPIC_SIM_DOWNLOAD_ICD10: "0"
      EPIC_SIM_LOAD_TERMINOLOGY: "0"
      EPIC_SIM_SETUP_SANDBOX_ROLE: "0"
    networks: [default, data]
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/env/current', timeout=2).status == 200 else 1)"]
      interval: 5s
      timeout: 3s
      retries: 60
      start_period: 20s

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: epic_sim
      POSTGRES_USER: epic_sim
      POSTGRES_PASSWORD: dev_password
    networks: [data]
    shm_size: "256m"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U epic_sim"]
      interval: 3s
      timeout: 3s
      retries: 20

  redis:
    image: redis:7-alpine
    networks: [data]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 3s
      timeout: 3s
      retries: 20

networks:
  data:
    internal: true
"""

TASK_TOML = """\
version = "1.0"

[metadata]
author_name = "Synthetic Hospital"
author_email = "{email}"
difficulty_explanation = "{difficulty}"
category = "clinical-ehr"
tags = ["synthetic-hospital", "ehr", "{task}", "{split}"]
gt_id = {gt_id}
task = "{task}"
split = "{split}"
patient_id = {patient_id}
benchmark_version = "1.3"

[verifier]
timeout_sec = 300.0
env = {{ EPIC_SIM_SCORER_TOKEN = "{token}", SH_BASE_URL = "http://app:8000" }}

[agent]
timeout_sec = {agent_timeout}

[solution]
env = {{ EPIC_SIM_SCORER_TOKEN = "{token}", SH_BASE_URL = "http://app:8000" }}

[environment]
build_timeout_sec = 900.0
"""

INSTRUCTION = """\
{instructions}

{intro}

## How to act

The patient's record is served by a simulated EHR at `http://app:8000`. You have a
budget of {budget} actions. Use the `sh-agent` command in this shell:

    sh-agent brief                      # this brief again
    sh-agent tools                      # tool schemas (JSON)
    sh-agent call open_chart '{{"patient_id": {patient_id}}}'
    sh-agent call view_encounters '{{"patient_id": {patient_id}}}'
    sh-agent call view_encounter_detail '{{"encounter_id": <id>}}'
    sh-agent call search_chart '{{"patient_id": {patient_id}, "query": "<terms>"}}'
    sh-agent state                      # actions used and remaining

Each `call` counts as one action. When you are done, submit exactly once with the
`{submit_tool}` schema shown above; this ends the task and no further actions are possible:

    sh-agent submit '<json>'

Nothing else is graded: only the submitted JSON is scored.
"""


def _chmod_x(p: Path) -> None:
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def export_task(out: Path, gt_id: int, *, image: str, token: str, budget: int, email: str,
                agent_timeout: float) -> Path:
    ctx = _load_context(gt_id)
    task = ctx["task"]
    kwargs = dict(ctx["task_kwargs"])
    instructions = build_system_prompt(task=task, arm="structured", budget=budget,
                                       encounter_id=ctx["encounter_id"], gt_id=gt_id, **kwargs)
    intro = build_patient_intro(patient_id=ctx["patient_id"], task=task, encounter_id=ctx["encounter_id"], **kwargs)

    name = f"sh-{task.replace('_', '-')}-{gt_id}"
    d = out / name
    if d.exists():
        shutil.rmtree(d)
    (d / "environment").mkdir(parents=True)
    (d / "solution").mkdir()
    (d / "tests").mkdir()

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT difficulty::text FROM benchmark_ground_truth WHERE gt_id = %s", (gt_id,))
            difficulty = (cur.fetchone() or ["unknown"])[0] or "unknown"

    (d / "task.toml").write_text(TASK_TOML.format(
        email=email, difficulty=f"{difficulty} (benchmark difficulty label)", task=task, split=ctx["split"],
        gt_id=gt_id, patient_id=ctx["patient_id"] or 0, token=token, agent_timeout=agent_timeout))
    (d / "instruction.md").write_text(INSTRUCTION.format(
        instructions=instructions, intro=intro, budget=budget, patient_id=ctx["patient_id"],
        submit_tool=submit_tool_for(task)))
    (d / "environment" / "docker-compose.yaml").write_text(COMPOSE.format(
        image=image, gt_id=gt_id, budget=budget, token=token))
    shutil.copy(TEMPLATES / "Dockerfile.main", d / "environment" / "Dockerfile")
    shutil.copy(TEMPLATES / "agent_cli.py", d / "environment" / "agent_cli.py")
    shutil.copy(TEMPLATES / "solve.sh", d / "solution" / "solve.sh")
    shutil.copy(TEMPLATES / "test.sh", d / "tests" / "test.sh")
    _chmod_x(d / "solution" / "solve.sh")
    _chmod_x(d / "tests" / "test.sh")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="harbor_tasks")
    ap.add_argument("--task", default="all", help="one of %s or 'all'" % EVAL_TASKS)
    ap.add_argument("--split", default="train", choices=["public", "heldout", "train"])
    ap.add_argument("--variant", default=None, help="context_summarization only (e.g. unconditioned, specialty_conditioned)")
    ap.add_argument("--limit", type=int, default=20, help="tasks per benchmark task (0 = all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--gt-id", type=int, action="append", default=[], help="export these instances only (repeatable)")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--agent-timeout", type=float, default=1800.0)
    ap.add_argument("--image", default=os.environ.get("SH_IMAGE", "ghcr.io/sparkcpark/synthetic-hospital:1.3-data"),
                    help="simulator image with the database baked in (default: the published one)")
    ap.add_argument("--token", default=os.environ.get("HARBOR_SCORER_TOKEN") or secrets.token_urlsafe(24))
    ap.add_argument("--email", default="synthetic-hospital@example.org")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.gt_id:
        ids = list(args.gt_id)
    else:
        tasks = EVAL_TASKS if args.task == "all" else [args.task]
        ids = []
        with get_pg_connection() as conn:
            for t in tasks:
                items = list_instances(conn, task=t, split=args.split, variant=args.variant,
                                       limit=args.limit or 100000, offset=args.offset)
                ids += [it["gt_id"] for it in items]

    made = []
    for gt_id in ids:
        d = export_task(out, gt_id, image=args.image, token=args.token, budget=args.budget,
                        email=args.email, agent_timeout=args.agent_timeout)
        made.append(d.name)
        print(f"  {d.name}")
    (out / "dataset.json").write_text(json.dumps({
        "benchmark": "synthetic-hospital", "version": "1.3", "split": args.split, "budget": args.budget,
        "image": args.image, "tasks": made}, indent=1))
    print(f"\n{len(made)} tasks written to {out}\n"
          f"run:  uvx harbor run -p {out} -a oracle        (expects reward 1.0 on every task)")


if __name__ == "__main__":
    main()
