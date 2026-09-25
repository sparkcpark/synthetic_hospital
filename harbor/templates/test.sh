#!/bin/bash
# Harbor verifier for a Synthetic Hospital task.
#
# Runs in the agent container after the agent finishes. Reads the outcome of the
# container's autostart episode from the simulator with the scorer token, which
# Harbor injects only during verification ([verifier].env in task.toml), and
# writes /logs/verifier/reward.json. An episode the agent never submitted is
# closed and scores 0.
set -euo pipefail

BASE="${SH_BASE_URL:-http://app:8000}"
mkdir -p /logs/verifier

python3 - "$BASE" "${EPIC_SIM_SCORER_TOKEN:-}" <<'PY'
import json, sys, urllib.request, urllib.error
base, token = sys.argv[1], sys.argv[2]
if not token:
    sys.exit("EPIC_SIM_SCORER_TOKEN is not set for the verifier")

def http(method, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers={"Content-Type": "application/json", "X-Scorer-Token": token})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

ep = http("GET", "/env/current")
state = http("GET", f"/env/state/{ep['episode_id']}")
if not state["done"]:
    state = http("POST", "/env/close", {"episode_id": ep["episode_id"]})
reward = float(state.get("reward") or 0.0)
out = {"reward": reward, "steps": state["steps"], "submitted": 1.0 if state["submitted"] else 0.0}
json.dump(out, open("/logs/verifier/reward.json", "w"))
print(f"gt_id={ep['gt_id']} task={ep['task']} submitted={state['submitted']} steps={state['steps']} reward={reward:.4f}")
PY
