#!/bin/bash
# Oracle solution for a Synthetic Hospital task: fetch the label-derived
# submission from the simulator (needs the scorer token, injected through
# [solution].env) and submit it through the ordinary agent path.
set -euo pipefail

BASE="${SH_BASE_URL:-http://app:8000}"

python3 - "$BASE" "${EPIC_SIM_SCORER_TOKEN:-}" <<'PY'
import json, sys, urllib.request
base, token = sys.argv[1], sys.argv[2]
if not token:
    sys.exit("EPIC_SIM_SCORER_TOKEN is not set for the solution")

def http(method, path, body=None, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())

ep = http("GET", "/env/current")
oracle = http("GET", f"/env/oracle/{ep['episode_id']}", headers={"X-Scorer-Token": token})
# one look at the chart, then submit, exactly as an agent would
http("POST", "/env/step", {"episode_id": ep["episode_id"], "name": "open_chart",
                            "arguments": {"patient_id": ep["patient_id"]}}, headers={"X-Episode-Token": ep["agent_token"]})
out = http("POST", "/env/step", {"episode_id": ep["episode_id"], "name": oracle["submit_tool"],
                                  "arguments": oracle["arguments"]}, headers={"X-Episode-Token": ep["agent_token"]})
print("submitted:", out["done"], "(reward withheld from the agent path)")
PY
