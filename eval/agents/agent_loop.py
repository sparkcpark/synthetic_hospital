"""Task-agnostic agent loop for both structured and bash arms.

The LLM receives context, decides an action (tool call or shell command),
observes the result, and iterates until it submits a prediction or exhausts
its budget.

Structured arm uses native function calling (tools parameter in API request).
Bash arm parses commands from ```bash``` code blocks in model text output.
"""

import json
import logging
import re
import time

from eval.adapters import ModelAdapter, ModelResponse
from eval.agents import Action, AgentResult
from eval.agents.bash_agent import BashAgentHarness
from eval.agents.prompts import SUBMIT_TOOL_SCHEMAS
from eval.agents.structured_agent import StructuredAgentHarness

log = logging.getLogger(__name__)

# Max chars per observation before truncation
MAX_OBS_CHARS = 8000

# Submission tool names
SUBMIT_TOOLS = frozenset({
    "submit_diagnosis", "submit_summary", "submit_pre_read", "submit_rankings",
})


def _pair_tool_messages(msgs: list[dict]) -> list[dict]:
    """Drop tool_calls/tool-response pairs that are not mutually complete.

    Strict providers reject a conversation where an assistant message carries
    tool_calls without a matching `tool` response ("Not the same number of
    function calls and responses") or where a tool result references an unknown
    call id ("Unexpected tool call id ..."). Either can arise when a tool
    invocation errors out, leaving a dangling assistant message that poisons
    every subsequent turn until the session aborts. Keep only matched pairs.
    """
    answered: set[str] = {
        m.get("tool_call_id") for m in msgs
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    out: list[dict] = []
    kept_ids: set[str] = set()
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = [tc for tc in m["tool_calls"] if tc.get("id") in answered]
            if not calls:
                continue  # no responses arrived -> drop the dangling call
            m = {**m, "tool_calls": calls}
            kept_ids.update(tc["id"] for tc in calls)
        elif m.get("role") == "tool":
            if m.get("tool_call_id") not in kept_ids:
                continue  # response whose call was dropped/never in window
        out.append(m)
    return out


class AgentLoop:
    """Task-agnostic agent loop for both structured and bash arms."""

    def __init__(
        self,
        model_adapter: ModelAdapter,
        harness: BashAgentHarness | StructuredAgentHarness,
        task: str,
        patient_id: int,
        gt_id: int,
        system_prompt: str,
        user_intro: str,
        max_turns: int = 40,
        encounter_id: int | None = None,
    ):
        self.adapter = model_adapter
        self.harness = harness
        self.task = task
        self.patient_id = patient_id
        self.gt_id = gt_id
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.encounter_id = encounter_id
        self.submitted = False
        self.forced_submission = False
        self.prediction: dict | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_ms = 0
        self.trace: list[dict] = []
        self.context_compressions = 0

        # Multi-turn conversation history
        self.messages: list[dict] = [
            {"role": "user", "content": user_intro},
        ]

        # Tool schemas for structured arm (fetched once)
        self._tool_schemas: list[dict] | None = None

    @property
    def is_bash(self) -> bool:
        return isinstance(self.harness, BashAgentHarness)

    @property
    def is_structured(self) -> bool:
        return isinstance(self.harness, StructuredAgentHarness)

    def _build_tool_schemas(self) -> list[dict]:
        """Build tool schemas for the structured arm.

        Fetches schemas from the Epic API, then replaces generic submit tools
        with task-specific versions that match the scoring schema.
        """
        api_schemas = self.harness.get_tool_schemas_for_api()

        # Get the task-specific submit schema
        submit_override = SUBMIT_TOOL_SCHEMAS.get(self.task)
        if not submit_override:
            return api_schemas

        submit_name = submit_override["function"]["name"]

        # Replace the matching submit tool, drop other submit tools
        result = []
        for schema in api_schemas:
            fn_name = schema["function"]["name"]
            if fn_name == submit_name:
                result.append(submit_override)
            elif fn_name in ("submit_diagnosis", "submit_summary", "submit_pre_read", "submit_rankings"):
                continue  # Drop non-matching submit tools
            else:
                result.append(schema)

        return result

    def run(self) -> AgentResult:
        """Execute the agent loop until submission or budget exhaustion."""
        # Fetch tool schemas once for structured arm, with task-specific submit overrides
        if self.is_structured:
            self._tool_schemas = self._build_tool_schemas()

        turns_used = 0
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        for turn in range(self.max_turns):
            remaining = self.harness.remaining_budget()
            if remaining <= 0 and not self.submitted:
                log.warning("Budget exhausted for patient=%d task=%s", self.patient_id, self.task)
                break

            # Call model with full conversation history + tools
            t0 = time.monotonic()
            try:
                # Gateway models (self-hosted) are 503-prone; use more retries
                max_retries = 5 if not self.adapter.config.api_key_env else 3
                response = self.adapter.call_multi_turn_with_retry(
                    self.system_prompt,
                    self._build_messages(),
                    tools=self._tool_schemas,  # None for bash arm
                    max_retries=max_retries,
                )
                consecutive_errors = 0  # reset on success
            except Exception as e:
                consecutive_errors += 1
                log.error("Model call failed on turn %d (consecutive=%d): %s",
                          turn + 1, consecutive_errors, e)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.error("Aborting session after %d consecutive errors", MAX_CONSECUTIVE_ERRORS)
                    break
                # Wait before retrying the next turn
                time.sleep(10)
                continue

            latency_ms = int((time.monotonic() - t0) * 1000)
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.total_latency_ms += latency_ms
            turns_used = turn + 1

            # Parse action from response
            action = self._parse_action(response)

            # --- No action: nudge the model ---
            if action.type == "none":
                # Add assistant's text to history
                self.messages.append({"role": "assistant", "content": response.text or "(no action)"})
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Please take an action. Either call a tool/run a command, "
                        "or submit your answer. "
                        f"You have {remaining} actions remaining."
                    ),
                })
                continue

            # --- Submission ---
            if action.type == "submit":
                self.submitted = True
                self.prediction = self._extract_submission(action)
                self.trace.append({
                    "turn": turns_used,
                    "action_type": "submit",
                    "action_name": action.tool_name or "submit",
                    "action_args": action.payload,
                    "output_length": 0,
                    "latency_ms": 0,
                })
                # Add assistant message to history for completeness
                if self.is_structured and response.raw_tool_calls:
                    self.messages.append({
                        "role": "assistant",
                        "content": response.text or "",
                        "tool_calls": response.raw_tool_calls,
                    })
                else:
                    self.messages.append({"role": "assistant", "content": response.text or ""})
                break

            # --- Tool call (structured) or command (bash) ---
            # Add assistant message to history
            if self.is_structured and response.raw_tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": response.raw_tool_calls,
                })
            else:
                self.messages.append({"role": "assistant", "content": response.text or ""})

            # Execute the action
            observation, trace_entry = self._execute_action(action, turns_used)
            self.trace.append(trace_entry)

            # Add observation to history
            remaining_after = self.harness.remaining_budget()

            if self.is_structured and response.raw_tool_calls:
                # OpenAI tool result format
                tool_call_id = response.raw_tool_calls[0].get("id", "")
                obs_content = observation
                if remaining_after <= 15 and remaining_after > 5:
                    obs_content += (
                        "\n\n[You have used more than half your action budget. "
                        "If you have gathered sufficient evidence for your answer, "
                        "consider submitting now rather than risking budget exhaustion. "
                        "You can always submit a partial answer — an incomplete "
                        "submission scores higher than no submission.]"
                    )
                if remaining_after <= 5:
                    obs_content += f"\n\n[You have {remaining_after} actions remaining. Consider submitting your answer soon.]"
                if remaining_after <= 1:
                    obs_content += "\n[This is your FINAL action. You MUST submit your answer now.]"
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": obs_content,
                })
            else:
                # Bash arm: observation as user message
                obs_parts = [observation]
                if remaining_after <= 15 and remaining_after > 5:
                    obs_parts.append(
                        "\n[You have used more than half your action budget. "
                        "If you have gathered sufficient evidence for your answer, "
                        "consider submitting now rather than risking budget exhaustion. "
                        "You can always submit a partial answer — an incomplete "
                        "submission scores higher than no submission.]"
                    )
                if remaining_after <= 5:
                    obs_parts.append(
                        f"\n[You have {remaining_after} actions remaining. "
                        "Consider submitting your answer soon.]"
                    )
                if remaining_after <= 1:
                    obs_parts.append(
                        "\n[This is your FINAL action. You MUST submit your answer now.]"
                    )
                self.messages.append({"role": "user", "content": "\n".join(obs_parts)})

            # Manage context window
            self._compress_if_needed()

        # --- Forced final submission ---
        # Weaker models can spend the whole action budget exploring and never call
        # the submit tool. Previously that produced no prediction at all, which the
        # patient_diagnosis scorer skips rather than scoring zero -- so a model that
        # answered 43 of 100 patients was ranked on those 43 alone. Give the agent
        # one last turn, tools withdrawn, to commit to an answer with what it has.
        # This only fires when the loop would otherwise return nothing, so arms that
        # always submit in time (e.g. GPT 5.3) are unaffected.
        if not self.submitted and turns_used > 0:
            log.warning("Budget spent without submission (patient=%d task=%s); "
                        "requesting forced submission", self.patient_id, self.task)
            # Spell out the required fields. After 40 turns of tool output the model
            # has often lost track of what it was asked for, and a generic "use the
            # required format" prompt draws a plausible-looking answer to the wrong
            # task (imaging pre-reads came back as bare diagnoses).
            schema = SUBMIT_TOOL_SCHEMAS.get(self.task) or {}
            props = (schema.get("function", {})
                           .get("parameters", {})
                           .get("properties", {}))
            fields = "".join(
                f"\n  - {k}: {(v or {}).get('description') or (v or {}).get('type','')}"
                for k, v in props.items()
            )
            self.messages.append({
                "role": "user",
                "content": (
                    "You have no actions remaining and have not submitted an answer. "
                    "Do not call any further tools. Submit your best answer NOW for "
                    f"the {self.task} task, using the information you have already "
                    "gathered. An incomplete answer is better than no answer."
                    + (f"\n\nYour answer must contain exactly these fields:{fields}"
                       if fields else "")
                    + "\n\nReturn it by calling the submit tool, or as a single JSON "
                      "object with those fields and nothing else."
                ),
            })
            # Offer only the submit tool, so the retrieval tools cannot tempt the
            # model into another exploration turn it has no budget for.
            submit_only = None
            if self.is_structured:
                schema = SUBMIT_TOOL_SCHEMAS.get(self.task)
                submit_only = [schema] if schema else self._tool_schemas
            try:
                response = self.adapter.call_multi_turn_with_retry(
                    self.system_prompt, self._build_messages(),
                    tools=submit_only, max_retries=3,
                )
                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens
                action = self._parse_action(response)
                submission = (self._extract_submission(action)
                              if action.type == "submit" else None)
                if submission is not None:
                    submission = self._normalize_submission(submission)
                if submission is None:
                    # Model answered in prose instead of calling the tool.
                    submission = self._submission_from_text(response.text or "")
                if submission:
                    self.submitted = True
                    self.forced_submission = True
                    self.prediction = submission
                    self.trace.append({
                        "turn": turns_used + 1, "action_type": "submit",
                        "action_name": "forced_submit", "action_args": {},
                        "output_length": 0, "latency_ms": 0,
                    })
            except Exception as e:
                log.error("Forced submission failed (patient=%d): %s", self.patient_id, e)

        return AgentResult(
            prediction=self.prediction,
            turns_used=turns_used,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_latency_ms=self.total_latency_ms,
            trace=self.trace,
            submitted=self.submitted,
            forced_submission=self.forced_submission,
            budget_exhausted=self.harness.remaining_budget() <= 0 and not self.submitted,
        )

    def _build_messages(self) -> list[dict]:
        """Build conversation messages for the current turn.

        For short conversations (≤8 messages), return as-is.
        For longer ones, compress early messages to save context window.

        Both paths are sanitized: a dangling assistant tool_call (one whose tool
        result never arrived, e.g. the call errored) is rejected outright by strict
        providers, and it persists in history, so every later turn fails until the
        session aborts. This happens independently of truncation, so it must be
        handled on the short path too.
        """
        if len(self.messages) <= 8:
            return _pair_tool_messages(self.messages)

        # Keep last 8 messages (4 exchanges) in full
        early = self.messages[:-8]
        recent = self.messages[-8:]

        # A `tool` message is only valid if the assistant message carrying its
        # tool_calls is still in the window. Truncation can orphan one, and strict
        # providers (Mistral) then reject the whole request with 400 Bad Request,
        # aborting the session after 3 consecutive failures. Drop leading orphans.
        while recent and recent[0].get("role") == "tool":
            early = early + [recent[0]]
            recent = recent[1:]

        recent = _pair_tool_messages(recent)

        # Compress early messages into a summary
        summaries = []
        for msg in early:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Anthropic-style content blocks
                content = str(content)[:200]
            elif len(content) > 200:
                content = content[:200] + "..."

            role = msg["role"]
            if role == "tool":
                summaries.append(f"[Tool result: {content}]")
            elif role == "assistant":
                if msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    fn = tc.get("function", tc)
                    summaries.append(f"[Called {fn.get('name', 'tool')}]")
                else:
                    summaries.append(f"[Assistant: {content}]")
            else:
                summaries.append(f"[{role.title()}: {content}]")

        compressed = {"role": "user", "content": "EARLIER CONTEXT:\n" + "\n".join(summaries)}
        return [compressed] + recent

    def _parse_action(self, response: ModelResponse) -> Action:
        """Extract the agent's intended action from the model response."""
        if self.is_structured:
            return self._parse_structured_action(response)
        return self._parse_bash_action(response.text)

    def _parse_structured_action(self, response: ModelResponse) -> Action:
        """Parse action from native tool calls in response."""
        if response.tool_calls and len(response.tool_calls) > 0:
            tc = response.tool_calls[0]  # Take the first tool call
            tool_name = tc["name"]
            arguments = tc["arguments"]

            if tool_name in SUBMIT_TOOLS:
                return Action(
                    type="submit",
                    payload=arguments,
                    tool_name=tool_name,
                )

            return Action(
                type="tool",
                payload=arguments,
                tool_name=tool_name,
            )

        # Model returned text without a tool call — treat as no-op
        return Action(type="none")

    def _parse_bash_action(self, text: str) -> Action:
        """Extract command from ```bash ... ``` code block or inline command."""
        # Check for submission via curl to submit endpoint
        for submit_tool in SUBMIT_TOOLS:
            if submit_tool in text:
                submission = self._extract_curl_submission(text)
                if submission:
                    return Action(type="submit", payload=submission, tool_name=submit_tool)

        # Try ```bash ... ``` blocks
        bash_match = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.DOTALL)
        if bash_match:
            command = bash_match.group(1).strip()
            if command:
                return Action(type="command", payload=command)

        # Try single backtick commands
        inline_match = re.search(r"`((?:psql|grep|jq|cat|head|tail|wc|curl|echo)\b[^`]+)`", text)
        if inline_match:
            return Action(type="command", payload=inline_match.group(1).strip())

        # Try lines starting with $ or common command prefixes
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("$ "):
                return Action(type="command", payload=line[2:].strip())
            if line.startswith(("psql ", "grep ", "curl ", "jq ", "cat ", "head ", "tail ", "echo ", "wc ")):
                return Action(type="command", payload=line.strip())

        return Action(type="none")

    def _extract_curl_submission(self, text: str) -> dict | None:
        """Extract submission payload from a curl command in bash output."""
        d_match = re.search(r"-d\s+'([^']+)'", text, re.DOTALL)
        if not d_match:
            d_match = re.search(r'-d\s+"([^"]+)"', text, re.DOTALL)
        if not d_match:
            return None

        try:
            data = json.loads(d_match.group(1))
            return data.get("arguments", {}).get("payload", data)
        except json.JSONDecodeError:
            return None

    # Top-level keys the scorer expects from each task's submission.
    TASK_KEYS = {
        "patient_diagnosis": ("active_diagnoses", "chronic_conditions"),
        "context_summarization": ("summary",),
        "evidence_retrieval": ("rankings",),
        "imaging_indication": ("clinical_question", "differential", "findings"),
    }

    def _normalize_submission(self, obj) -> dict | None:
        """Coerce a recovered payload into the shape the scorer expects.

        A model that is told to submit without the tool schema in front of it often
        returns the inner record -- a lone diagnosis `{"icd10":..,"name":..}`, or a
        bare list of them -- rather than the wrapper object. Scoring such a payload
        as-is yields an empty problem list, which is indistinguishable from the
        non-submission this turn exists to prevent.
        """
        keys = self.TASK_KEYS.get(self.task, ())
        if isinstance(obj, dict):
            if any(k in obj for k in keys):
                return obj
            if self.task == "patient_diagnosis" and "icd10" in obj:
                return {"active_diagnoses": [obj], "chronic_conditions": []}
            if self.task == "context_summarization":
                for k in ("text", "narrative", "assessment"):
                    if isinstance(obj.get(k), str):
                        return {"summary": obj[k]}
            # Nothing here matches this task's schema. Returning it anyway would
            # record a "submission" the scorer can only score zero -- e.g. a bare
            # {"diagnosis","icd10"} scraped from prose on an imaging pre-read --
            # which inflates apparent coverage while contributing nothing. Refuse.
            return None
        if isinstance(obj, list) and obj:
            if self.task == "patient_diagnosis":
                dx = [d for d in obj if isinstance(d, dict) and "icd10" in d]
                if dx:
                    return {"active_diagnoses": dx, "chronic_conditions": []}
            if self.task == "evidence_retrieval":
                return {"rankings": obj}
        return None

    def _submission_from_text(self, text: str) -> dict | None:
        """Recover a submission from a prose reply on the forced-submit turn.

        Scans every balanced JSON value in the reply and prefers the one that
        already carries this task's expected keys, falling back to the largest
        parseable value. Taking the *first* match instead would grab an inner
        record out of a list and silently drop the rest of the answer.
        """
        candidates: list[tuple[int, object]] = []
        for m in re.finditer(r"[\{\[]", text):
            opener = text[m.start()]
            closer = "}" if opener == "{" else "]"
            depth = 0
            for i in range(m.start(), len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            candidates.append(
                                (i + 1 - m.start(), json.loads(text[m.start():i + 1]))
                            )
                        except json.JSONDecodeError:
                            pass
                        break

        keys = self.TASK_KEYS.get(self.task, ())
        for _, obj in sorted(candidates, key=lambda c: -c[0]):
            payload = obj.get("payload", obj) if isinstance(obj, dict) else obj
            if isinstance(payload, dict) and any(k in payload for k in keys):
                return payload
        for _, obj in sorted(candidates, key=lambda c: -c[0]):
            norm = self._normalize_submission(obj.get("payload", obj)
                                              if isinstance(obj, dict) else obj)
            if norm:
                return norm

        # No complete top-level object carried this task's fields. Two shapes show
        # up here: the object *body* without its outer braces, and a reply cut off
        # mid-structure. Both are recoverable, and both otherwise fall through to a
        # nested record (an imaging differential entry looks like a diagnosis), so
        # try to repair before giving up.
        for repaired in self._repair_candidates(text):
            try:
                obj = json.loads(repaired)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                payload = obj.get("payload", obj)
                if isinstance(payload, dict) and any(k in payload for k in keys):
                    return payload

        stripped = text.strip()
        if stripped and self.task == "context_summarization":
            return {"summary": stripped}
        return None

    @staticmethod
    def _repair_candidates(text: str) -> list[str]:
        """Yield plausible repairs of a not-quite-JSON reply, best guess first."""
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        out: list[str] = []
        bodies = [t] if t.startswith("{") else [t, "{" + t + "}"]
        for body in bodies:
            out.append(body)
            # Close whatever is still open, dropping any partial trailing token so
            # the truncation point lands on a syntactically complete value.
            depth_c = body.count("{") - body.count("}")
            depth_b = body.count("[") - body.count("]")
            if depth_c <= 0 and depth_b <= 0:
                continue
            trimmed = body[:max(body.rfind(","), body.rfind("]"), body.rfind("}"))]
            for base in (body, trimmed):
                if not base:
                    continue
                if base.count('"') % 2:  # truncated mid-string
                    base += '"'
                out.append(base + "]" * max(0, base.count("[") - base.count("]"))
                                + "}" * max(0, base.count("{") - base.count("}")))
        return out

    def _extract_submission(self, action: Action) -> dict | None:
        """Extract the prediction dict from a submission action."""
        if isinstance(action.payload, dict):
            # For structured: payload is the tool arguments dict
            # Check if it has a "payload" key (nested) or is the payload itself
            if "payload" in action.payload:
                return action.payload["payload"]
            # Check for direct active_diagnoses/summary/rankings keys
            for key in ("active_diagnoses", "chronic_conditions", "summary",
                        "rankings", "clinical_question", "differential"):
                if key in action.payload:
                    return action.payload
            return action.payload
        return None

    def _execute_action(self, action: Action, turn_num: int) -> tuple[str, dict]:
        """Execute action via the appropriate harness and return (observation, trace_entry)."""
        if self.is_bash:
            command = action.payload if isinstance(action.payload, str) else str(action.payload)
            output, exit_code, entry = self.harness.execute(command)
            trace_entry = {
                "turn": turn_num,
                "action_type": "command" if not entry.get("blocked") else "blocked",
                "action_name": command.split()[0] if command.split() else "unknown",
                "action_args": {"command": command},
                "output_length": len(output),
                "latency_ms": entry.get("latency_ms", 0),
                "truncated": entry.get("truncated", False),
                "warned": entry.get("warned", False),
            }
            return output, trace_entry
        else:
            tool_name = action.tool_name or "unknown"
            arguments = action.payload if isinstance(action.payload, dict) else {}
            result, entry = self.harness.call_tool(tool_name, arguments)
            result_str = json.dumps(result, indent=2, default=str) if isinstance(result, (dict, list)) else str(result)
            if len(result_str) > MAX_OBS_CHARS:
                result_str = result_str[:MAX_OBS_CHARS] + "\n[OUTPUT TRUNCATED]"
            trace_entry = {
                "turn": turn_num,
                "action_type": "tool",
                "action_name": tool_name,
                "action_args": arguments,
                "output_length": entry.get("result_length", len(result_str)),
                "latency_ms": entry.get("latency_ms", 0),
            }
            return result_str, trace_entry

    def _compress_if_needed(self):
        """Compress conversation if it's getting too long."""
        total_chars = sum(
            len(m.get("content", "")) if isinstance(m.get("content"), str)
            else len(str(m.get("content", "")))
            for m in self.messages
        )
        # Rough threshold: 80% of a 128K context window at ~4 chars/token
        if total_chars > 100_000:
            early = self.messages[:-8]
            recent = self.messages[-8:]

            summary_parts = ["CONVERSATION HISTORY SUMMARY:"]
            for i, msg in enumerate(early):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = str(content)
                role = msg["role"]
                if role == "assistant" and msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    fn = tc.get("function", tc)
                    summary_parts.append(f"Turn {i}: Called {fn.get('name', 'tool')}")
                elif role == "tool":
                    summary_parts.append(f"Turn {i}: Tool result ({len(content)} chars)")
                else:
                    summary_parts.append(f"Turn {i}: {role}={content[:100]}...")

            self.messages = [
                {"role": "user", "content": "\n".join(summary_parts)},
            ] + recent
            self.context_compressions += 1
            log.info("Compressed context (compression #%d)", self.context_compressions)
