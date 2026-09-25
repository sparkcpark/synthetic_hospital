"""Structured agent harness: HTTP dispatch to Epic API tools with session management.

Reuses the existing agent protocol (§8.6) with session tracking.
One session per (patient, task) combination.
"""

import json
import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

# The LLM-only baselines assemble patient context with
# `section_type NOT IN ('assessment', 'plan')` — those sections state the
# conclusion (e.g. an assessment reading "acute myocardial infarction") and would
# hand the agent the answer. The agent authenticates as `attending`, which has
# unrestricted section access in the simulator's RBAC, so without this filter the
# agentic arms would see strictly more than the baseline they are compared against.
# Set AGENT_SHOW_OUTCOME_SECTIONS=1 to restore the old (leaky) behaviour.
OUTCOME_SECTIONS = {"assessment", "plan"}
HIDE_OUTCOME_SECTIONS = os.environ.get("AGENT_SHOW_OUTCOME_SECTIONS", "") != "1"


def _strip_outcome_sections(obj):
    """Recursively drop assessment/plan section entries from a tool result."""
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict) and str(item.get("section_type", "")).lower() in OUTCOME_SECTIONS:
                continue
            out.append(_strip_outcome_sections(item))
        return out
    if isinstance(obj, dict):
        return {k: _strip_outcome_sections(v) for k, v in obj.items()
                if not (k.lower() in OUTCOME_SECTIONS and isinstance(v, str))}
    return obj


class StructuredAgentHarness:
    """Manages structured tool-use agent sessions via the Epic API."""

    def __init__(
        self,
        api_base: str,
        token: str,
        session_id: str,
        gt_id: int,
        max_calls: int = 40,
    ):
        self.api_base = api_base
        self.token = token
        self.session_id = session_id
        self.gt_id = gt_id
        self.max_calls = max_calls
        self.tool_log: list[dict] = []
        self._budget_used = 0
        self._tool_schemas: list[dict] | None = None

    def get_tool_schemas_for_api(self) -> list[dict]:
        """Fetch tool definitions from the Epic API and convert to OpenAI
        function-calling format for passing to the model adapter.

        Returns list of:
        {
            "type": "function",
            "function": {
                "name": "open_chart",
                "description": "...",
                "parameters": { ... json schema ... }
            }
        }
        """
        if self._tool_schemas is not None:
            return self._tool_schemas

        raw_tools = self.get_available_tools()  # GET /agent/tools

        schemas = []
        for tool in raw_tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {
                        "type": "object",
                        "properties": {},
                    }),
                },
            })

        self._tool_schemas = schemas
        log.info("Loaded %d tool schemas for structured arm", len(schemas))
        return schemas

    def call_tool(self, tool_name: str, arguments: dict) -> tuple[dict, dict]:
        """Execute a tool call via POST /agent/tools.

        Returns (result_dict, log_entry).
        """
        t0 = time.monotonic()

        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
            "session_id": self.session_id,
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.api_base}/agent/tools",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.token}",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            data = {"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
        except Exception as e:
            data = {"error": str(e)}

        latency_ms = int((time.monotonic() - t0) * 1000)
        self._budget_used += 1

        result = data.get("result", data)
        if HIDE_OUTCOME_SECTIONS:
            result = _strip_outcome_sections(result)
        result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)

        entry = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "result_length": len(result_str),
            "latency_ms": latency_ms,
        }
        self.tool_log.append(entry)

        return result, entry

    def get_available_tools(self) -> list[dict]:
        """GET /agent/tools — returns tool schemas for the agent's function calling."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.api_base}/agent/tools",
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            log.error("Failed to fetch tool schemas: %s", e)
            return []

    def get_trace(self) -> list[dict]:
        """Return full tool call trace for analysis."""
        return list(self.tool_log)

    def remaining_budget(self) -> int:
        """Tool calls remaining in budget."""
        return self.max_calls - self._budget_used
