"""Gymnasium-style Python client for the Synthetic Hospital environment.

    from eval.env_client import SyntheticHospitalEnv

    env = SyntheticHospitalEnv("http://localhost:8000", scorer_token=TOKEN)
    obs = env.reset(task="patient_diagnosis", split="train", seed=0)
    # obs.instructions -> system prompt, obs.intro -> first user message,
    # obs.tools -> OpenAI-style function schemas (incl. the submit tool)
    obs_json, reward, done, info = env.step("open_chart", {"patient_id": obs.patient_id})
    ...
    obs_json, reward, done, info = env.step(obs.submit_tool, {"active_diagnoses": [...], "chronic_conditions": []})
    assert done and 0.0 <= reward <= 1.0

The policy is yours to drive: feed `instructions`, `intro` and `tools` to any
model, translate its tool calls into `step()`, and use the terminal reward for
training. Only this client (the harness) holds the scorer token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ResetObservation:
    episode_id: str
    agent_token: str
    gt_id: int
    task: str
    split: str
    variant: str | None
    patient_id: int | None
    encounter_id: int | None
    budget: int
    remaining: int
    submit_tool: str
    instructions: str
    intro: str
    task_inputs: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)


class SyntheticHospitalEnv:
    def __init__(self, base_url: str = "http://localhost:8000", scorer_token: str = "",
                 budget: int | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.budget = budget
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout,
                                    headers={"X-Scorer-Token": scorer_token})
        self.episode_id: str | None = None
        self.last_reset: ResetObservation | None = None

    # -- helpers -----------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        r = self._client.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{path} -> {r.status_code}: {r.text}")
        return r.json()

    # -- instance discovery ------------------------------------------------
    def instances(self, task: str, split: str = "train", limit: int = 500, offset: int = 0,
                  variant: str | None = None) -> list[dict]:
        params = {"task": task, "split": split, "limit": limit, "offset": offset}
        if variant:
            params["variant"] = variant
        r = self._client.get("/score/instances", params=params)
        r.raise_for_status()
        return r.json()

    # -- gym-style API -----------------------------------------------------
    def reset(self, gt_id: int | None = None, task: str | None = None, split: str | None = None,
              seed: int | None = None, budget: int | None = None) -> ResetObservation:
        body = {"gt_id": gt_id, "task": task, "split": split, "seed": seed,
                "budget": budget or self.budget}
        data = self._post("/env/reset", {k: v for k, v in body.items() if v is not None})
        self.last_reset = ResetObservation(**data)
        self.episode_id = self.last_reset.episode_id
        return self.last_reset

    def step(self, name: str, arguments: dict | None = None) -> tuple[Any, float, bool, dict]:
        """Returns (observation, reward, done, info). `observation` is the tool's JSON
        result (or {"error": ...}); info["observation_text"] is its rendered, truncated form."""
        if not self.episode_id:
            raise RuntimeError("call reset() first")
        data = self._post("/env/step", {"episode_id": self.episode_id, "name": name,
                                        "arguments": arguments or {}})
        info = dict(data["info"])
        if data["reward"] is None:      # agent-token caller: reward withheld by design
            data["reward"] = 0.0
            info["reward_withheld"] = True
        info.update({"remaining": data["remaining"], "step": data["step"],
                     "observation_text": data["observation_text"]})
        if data["done"]:
            self.episode_id = None
        return data["observation"], float(data["reward"]), bool(data["done"]), info

    def state(self, episode_id: str | None = None) -> dict:
        eid = episode_id or self.episode_id
        r = self._client.get(f"/env/state/{eid}")
        r.raise_for_status()
        return r.json()

    def close(self) -> dict | None:
        if not self.episode_id:
            return None
        data = self._post("/env/close", {"episode_id": self.episode_id})
        self.episode_id = None
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        finally:
            self._client.close()
