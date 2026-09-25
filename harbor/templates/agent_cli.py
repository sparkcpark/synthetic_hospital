#!/usr/bin/env python3
"""sh-agent: the agent-side command line for a Synthetic Hospital Harbor task.

Talks to the simulator service (SH_BASE_URL, default http://app:8000) using only
the per-episode agent token that the service hands out for the episode this
container was started for. It never sees the labels or the reward.

    sh-agent brief                      task instructions, patient assignment, budget
    sh-agent tools                      tool schemas (JSON)
    sh-agent call <tool> ['<json>']     call an EHR tool, e.g. sh-agent call open_chart '{"patient_id": 1973}'
    sh-agent submit '<json>'            final answer in the task's submission schema (ends the episode)
    sh-agent state                      steps used / remaining / done
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("SH_BASE_URL", "http://app:8000").rstrip("/")


def _http(method: str, path: str, body: dict | None = None, token: str | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          **({"X-Episode-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"error {exc.code} from {path}: {detail}")


def current() -> dict:
    return _http("GET", "/env/current")


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    ep = current()
    eid, tok = ep["episode_id"], ep["agent_token"]

    if cmd == "brief":
        print(ep["instructions"])
        print("\n" + ep["intro"])
        print(f"\nBudget: {ep['budget']} actions, {ep['remaining']} remaining. "
              f"Finish with: sh-agent submit '<json in the {ep['submit_tool']} schema>'")
        print("Tools: " + ", ".join(t["function"]["name"] for t in ep["tools"]))
    elif cmd == "tools":
        print(json.dumps(ep["tools"], indent=2))
    elif cmd in ("call", "submit"):
        if cmd == "call":
            if not rest:
                sys.exit("usage: sh-agent call <tool> ['<json arguments>']")
            name, args = rest[0], json.loads(rest[1]) if len(rest) > 1 else {}
        else:
            if not rest:
                sys.exit("usage: sh-agent submit '<json>'")
            name, args = ep["submit_tool"], json.loads(rest[0])
        out = _http("POST", "/env/step", {"episode_id": eid, "name": name, "arguments": args}, token=tok)
        print(out["observation_text"])
        if out["done"]:
            print("\n[episode finished: submission recorded]")
        else:
            print(f"\n[{out['remaining']} actions remaining]")
    elif cmd == "state":
        print(json.dumps(_http("GET", f"/env/state/{eid}", token=tok), indent=2))
    else:
        sys.exit(f"unknown command {cmd!r}; see sh-agent --help")


if __name__ == "__main__":
    main(sys.argv[1:])
