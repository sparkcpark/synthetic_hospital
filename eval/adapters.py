"""Model adapters for LLM API calls.

Three concrete adapters:
  - OpenAICompatibleAdapter: GPT, Llama, Qwen, DeepSeek, Mistral, Gemma, GLM, vLLM, Ollama
  - AnthropicAdapter: Anthropic Messages API (native) and the self-hosted gateway
  - GoogleAdapter: Gemini via google-genai SDK
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field

import httpx

from eval.config import AdapterType, ModelConfig, get_api_key

log = logging.getLogger(__name__)


# Retry policy (exponential backoff with jitter, honouring Retry-After):
# exponential backoff capped at max_backoff, plus jitter, with 502/5xx/timeout/429
# retryable and 4xx (<429) not. Env-gated so defaults are unchanged; the robustness
# Gateway-served models raise EVAL_MAX_RETRIES to ride out transient 502s.
def _retry_max(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return int(os.environ.get("EVAL_MAX_RETRIES", "3"))


def _retry_backoff(attempt: int) -> float:
    base = float(os.environ.get("EVAL_RETRY_BASE_DELAY", "5"))
    cap = float(os.environ.get("EVAL_RETRY_MAX_BACKOFF", "16"))
    return min(base * (2 ** attempt), cap) + random.uniform(0, 1)


# ---------------------------------------------------------------------------
# Uniform response
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    """Uniform response from any model adapter."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    raw_json: str | dict | None
    tool_calls: list[dict] | None = None       # parsed: [{"name": ..., "arguments": dict}]
    raw_tool_calls: list[dict] | None = None   # original format for conversation history


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class ModelAdapter:
    """Base class for LLM adapters."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = get_api_key(config.name)

    def call(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        raise NotImplementedError

    def call_messages(
        self, system_prompt: str, messages: list[dict]
    ) -> ModelResponse:
        """Multi-turn call with a list of {"role": ..., "content": ...} messages.

        Default implementation concatenates into a single user prompt.
        Subclasses should override for native multi-turn support.
        """
        combined = "\n\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in messages
        )
        return self.call(system_prompt, combined)

    def call_multi_turn(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        """Multi-turn call with optional native tool calling.

        Args:
            system_prompt: System prompt.
            messages: Conversation history (may include tool/assistant messages).
            tools: OpenAI-format tool schemas for native function calling.
        """
        # Default: ignore tools, delegate to call_messages
        return self.call_messages(system_prompt, messages)

    def call_with_retry(
        self, system_prompt: str, user_prompt: str, max_retries: int | None = None
    ) -> ModelResponse:
        """Call with exponential backoff on transient errors."""
        max_retries = _retry_max(max_retries)
        last_err = None
        for attempt in range(max_retries):
            try:
                return self.call(system_prompt, user_prompt)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_err = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 429:
                    raise  # 4xx client errors (not 429) are not retryable
                wait = _retry_backoff(attempt)
                log.warning(
                    "Retry %d/%d for %s: %s (waiting %ds)",
                    attempt + 1, max_retries, self.config.name, e, wait,
                )
                time.sleep(wait)
            except Exception as e:
                last_err = e
                wait = _retry_backoff(attempt)
                log.warning(
                    "Retry %d/%d for %s: %s (waiting %ds)",
                    attempt + 1, max_retries, self.config.name, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"All {max_retries} retries exhausted for {self.config.name}"
        ) from last_err

    def call_messages_with_retry(
        self, system_prompt: str, messages: list[dict], max_retries: int | None = None
    ) -> ModelResponse:
        """Multi-turn call with exponential backoff on transient errors."""
        max_retries = _retry_max(max_retries)
        last_err = None
        for attempt in range(max_retries):
            try:
                return self.call_messages(system_prompt, messages)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_err = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 429:
                    raise
                wait = _retry_backoff(attempt)
                log.warning(
                    "Retry %d/%d for %s: %s (waiting %ds)",
                    attempt + 1, max_retries, self.config.name, e, wait,
                )
                time.sleep(wait)
            except Exception as e:
                last_err = e
                wait = _retry_backoff(attempt)
                log.warning(
                    "Retry %d/%d for %s: %s (waiting %ds)",
                    attempt + 1, max_retries, self.config.name, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"All {max_retries} retries exhausted for {self.config.name}"
        ) from last_err

    def call_multi_turn_with_retry(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int | None = None,
    ) -> ModelResponse:
        """Multi-turn + tools call with exponential backoff."""
        max_retries = _retry_max(max_retries)
        last_err = None
        for attempt in range(max_retries):
            try:
                return self.call_multi_turn(system_prompt, messages, tools)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_err = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 429:
                    raise
                wait = _retry_backoff(attempt)
                log.warning(
                    "Retry %d/%d for %s: %s (waiting %ds)",
                    attempt + 1, max_retries, self.config.name, e, wait,
                )
                time.sleep(wait)
            except Exception as e:
                last_err = e
                wait = _retry_backoff(attempt)
                log.warning(
                    "Retry %d/%d for %s: %s (waiting %ds)",
                    attempt + 1, max_retries, self.config.name, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"All {max_retries} retries exhausted for {self.config.name}"
        ) from last_err


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter
# ---------------------------------------------------------------------------

class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for OpenAI-compatible chat completions API.

    Covers: GPT, Llama, Qwen, DeepSeek, Mistral, Gemma, GLM, vLLM, Ollama.
    """

    def _send(self, messages: list[dict], tools: list[dict] | None = None) -> ModelResponse:
        """Shared send logic for single and multi-turn calls."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        # Reasoning/thinking passthrough (e.g. OpenRouter `reasoning` param for Kimi-thinking).
        if self.config.extra.get("reasoning"):
            payload["reasoning"] = self.config.extra["reasoning"]
        # OpenRouter provider routing. Gemma 3 is served by five providers but only
        # DeepInfra supports function calling; unpinned requests get routed to
        # tool-less providers and return 429s, which stalls the agent loop.
        if self.config.extra.get("provider"):
            payload["provider"] = self.config.extra["provider"]

        t0 = time.monotonic()
        with httpx.Client(timeout=self.config.timeout_secs) as client:
            resp = client.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            raw_json = resp.text
            data = resp.json()

        latency_ms = int((time.monotonic() - t0) * 1000)
        message = data["choices"][0]["message"]
        text = message.get("content") or ""
        usage = data.get("usage", {})

        # Parse native tool calls
        parsed_tool_calls = None
        raw_tool_calls = None
        if message.get("tool_calls"):
            raw_tool_calls = message["tool_calls"]
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                fn = tc["function"]
                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parsed_tool_calls.append({"name": fn["name"], "arguments": args})

        return ModelResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            raw_json=raw_json,
            tool_calls=parsed_tool_calls,
            raw_tool_calls=raw_tool_calls,
        )

    def call(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        return self._send([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    def call_messages(self, system_prompt: str, messages: list[dict]) -> ModelResponse:
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)
        return self._send(api_messages)

    def call_multi_turn(
        self, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)
        return self._send(api_messages, tools=tools)


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Messages API.

    Covers: the Anthropic Messages API (native) and the self-hosted gateway (SH_LLM_GATEWAY_URL).
    Pitfalls:
      - Kimi returns thinking blocks: filter for type='text'
      - Kimi JSON: use json.loads(strict=False)
    """

    def _send(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        """Shared send logic for single and multi-turn calls.

        When `gateway_openai_response` is set in config.extra, the gateway
        accepts requests at /v1/messages but returns OpenAI-format responses
        (e.g., GLM-5 via ZhipuAI provider). In this mode:
        - Tools are passed in OpenAI format (not converted to Anthropic)
        - Response is parsed as OpenAI chat completion format
        """
        openai_mode = self.config.extra.get("gateway_openai_response", False)

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        payload = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if self.config.temperature > 0:
            payload["temperature"] = self.config.temperature
        if self.config.extra.get("disable_thinking"):
            payload["thinking"] = {"type": "disabled"}

        if tools:
            if openai_mode:
                # gateway in OpenAI-response mode: pass tools in OpenAI format directly
                payload["tools"] = tools
            else:
                # Standard Anthropic: convert OpenAI → Anthropic tool format
                anthropic_tools = []
                for t in tools:
                    if t.get("type") == "function":
                        fn = t["function"]
                        anthropic_tools.append({
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                        })
                    else:
                        anthropic_tools.append(t)
                payload["tools"] = anthropic_tools

        t0 = time.monotonic()
        with httpx.Client(timeout=self.config.timeout_secs) as client:
            resp = client.post(
                f"{self.config.base_url}/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            raw_json = resp.text
            data = json.loads(raw_json, strict=False)

        latency_ms = int((time.monotonic() - t0) * 1000)

        if openai_mode:
            return self._parse_openai_response(data, raw_json, latency_ms)
        return self._parse_anthropic_response(data, raw_json, latency_ms)

    def _parse_anthropic_response(
        self, data: dict, raw_json: str, latency_ms: int
    ) -> ModelResponse:
        """Parse standard Anthropic Messages API response."""
        text_parts = []
        parsed_tool_calls = []
        raw_tool_calls_anthropic = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_use_id = block.get("id", "")
                name = block["name"]
                arguments = block.get("input", {})
                parsed_tool_calls.append({"name": name, "arguments": arguments})
                # Synthesize OpenAI-format raw_tool_calls for conversation history
                raw_tool_calls_anthropic.append({
                    "id": tool_use_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                })

        text = "\n".join(text_parts)
        usage = data.get("usage", {})
        return ModelResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            raw_json=raw_json,
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
            raw_tool_calls=raw_tool_calls_anthropic if raw_tool_calls_anthropic else None,
        )

    def _parse_openai_response(
        self, data: dict, raw_json: str, latency_ms: int
    ) -> ModelResponse:
        """Parse an OpenAI-format response returned by the gateway."""
        message = data["choices"][0]["message"]
        text = message.get("content") or ""
        usage = data.get("usage", {})

        parsed_tool_calls = None
        raw_tool_calls = None
        if message.get("tool_calls"):
            raw_tool_calls = message["tool_calls"]
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                fn = tc["function"]
                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parsed_tool_calls.append({"name": fn["name"], "arguments": args})

        return ModelResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            raw_json=raw_json,
            tool_calls=parsed_tool_calls,
            raw_tool_calls=raw_tool_calls,
        )

    @staticmethod
    def _convert_messages_to_anthropic(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages (including tool results) to Anthropic format."""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "tool":
                # OpenAI tool result → Anthropic tool_result in a user message
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                # OpenAI assistant with tool_calls → Anthropic assistant with tool_use blocks
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn["name"],
                        "input": args,
                    })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
            else:
                anthropic_messages.append(msg)
        return anthropic_messages

    def call(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        return self._send(system_prompt, [{"role": "user", "content": user_prompt}])

    def call_messages(self, system_prompt: str, messages: list[dict]) -> ModelResponse:
        return self._send(system_prompt, messages)

    def call_multi_turn(
        self, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        if self.config.extra.get("gateway_openai_response"):
            # gateway in OpenAI-response mode: messages stay in OpenAI format (tool_calls, tool role)
            return self._send(system_prompt, messages, tools=tools)
        # Standard Anthropic: convert OpenAI-format tool calls/results
        anthropic_messages = self._convert_messages_to_anthropic(messages)
        return self._send(system_prompt, anthropic_messages, tools=tools)


# ---------------------------------------------------------------------------
# Google adapter
# ---------------------------------------------------------------------------

class GoogleAdapter(ModelAdapter):
    """Adapter for Google Gemini via google-genai SDK."""

    def _send(self, system_prompt: str, contents) -> ModelResponse:
        """Shared send logic for single and multi-turn calls."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("pip install google-genai for Gemini support")

        client = genai.Client(api_key=self.api_key)

        t0 = time.monotonic()
        response = client.models.generate_content(
            model=self.config.model_id,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            ),
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        text = response.text or ""
        usage = response.usage_metadata
        return ModelResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_ms=latency_ms,
            raw_json=str(response),
        )

    def call(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        return self._send(system_prompt, user_prompt)

    def call_messages(self, system_prompt: str, messages: list[dict]) -> ModelResponse:
        # Convert to Gemini Content format
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai for Gemini support")

        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        return self._send(system_prompt, contents)


# ---------------------------------------------------------------------------
# Token budget helper
# ---------------------------------------------------------------------------

def get_input_budget(model_name: str, task: str, strategy: str) -> int:
    """Available input tokens after reserving for system prompt, strategy overhead, and output."""
    from eval.config import MODEL_CONTEXT_WINDOWS, TASK_MAX_TOKENS

    SYSTEM_PROMPT_BUFFER = 500
    STRATEGY_OVERHEAD = {
        "zero_shot": 200, "few_shot": 1200, "cot": 800, "structured": 500,
    }
    context = MODEL_CONTEXT_WINDOWS.get(model_name, 128000)
    return (context
            - TASK_MAX_TOKENS.get(task, 512)
            - STRATEGY_OVERHEAD.get(strategy, 500)
            - SYSTEM_PROMPT_BUFFER)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(config: ModelConfig) -> ModelAdapter:
    """Create the appropriate adapter for a model configuration."""
    if config.adapter_type == AdapterType.OPENAI_COMPATIBLE:
        return OpenAICompatibleAdapter(config)
    elif config.adapter_type == AdapterType.ANTHROPIC:
        return AnthropicAdapter(config)
    elif config.adapter_type == AdapterType.GOOGLE:
        return GoogleAdapter(config)
    else:
        raise ValueError(f"Unknown adapter type: {config.adapter_type}")
