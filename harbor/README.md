# Synthetic Hospital as Harbor tasks

[Harbor](https://harborframework.com) runs any agent against a task packaged as a
directory (`task.toml`, `instruction.md`, `environment/`, `tests/`, `solution/`).
`scripts/harbor_export.py` turns benchmark instances into such tasks, one per
instance, so the whole training split (7,619 instances) or any subset can be run
with Harbor's agents or used as an RL environment through Harbor's trajectory
tooling.

## Layout of an exported task

```
sh-patient-diagnosis-7641/
├── task.toml                  metadata; scorer token for the verifier and the oracle only
├── instruction.md             the paper's agent prompt, patient assignment, and how to act
├── environment/
│   ├── Dockerfile             agent container ("main"): shell, curl, jq, sh-agent
│   ├── agent_cli.py
│   └── docker-compose.yaml    app (simulator, autostarts this episode) + postgres + redis
├── solution/solve.sh          oracle: fetches the label-derived submission and submits it
└── tests/test.sh              verifier: reads the episode outcome -> /logs/verifier/reward.json
```

How the pieces fit:

- The `app` service is the published release image with the database baked in
  (`ghcr.io/sparkcpark/synthetic-hospital:1.3-data`, pulled automatically; build it locally under that name or pass
  `--image` to use another). On boot it
  loads the database and starts one episode for the task's instance
  (`EPIC_SIM_AUTOSTART_GT_ID`).
- The agent works in `main` with `sh-agent`, which fetches the episode brief from
  `GET /env/current` and acts through `POST /env/step` with a per-episode agent
  token. That token cannot read the reward or the labels.
- `postgres` and `redis` are on an internal network the agent container is not
  attached to, so the ground truth in the database is out of the agent's reach.
- The verifier runs after the agent with the scorer token from `[verifier].env`,
  reads the episode's reward (0 if nothing was submitted) and writes
  `reward.json` with `reward`, `steps` and `submitted`.
- The oracle (`harbor run -a oracle`) uses `[solution].env` to fetch a
  label-derived submission from `GET /env/oracle/{id}` and submits it; every task
  should score 1.0, which is the export's self-check.

## Export and run

```bash
docker compose up -d                                              # exporter reads prompt inputs from the stack
python scripts/harbor_export.py --out harbor_tasks --task patient_diagnosis --split train --limit 50
uvx harbor run -p harbor_tasks -a oracle                          # self-check
uvx harbor run -p harbor_tasks -a terminus-2 -m <provider/model>   # any Harbor agent
```

`--task all --limit 0` exports every instance of a split. A random scorer token is
generated per export and embedded in the task files; pass `--token` to choose one.
Rewards are the paper's primary metrics (see the main README, Scoring Endpoint),
in [0, 1], so `harbor` reports them directly.
