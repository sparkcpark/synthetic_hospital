"""Phase F: Agent tool-use evaluation — structured Epic tools vs. bash baseline."""

from dataclasses import dataclass, field


@dataclass
class AgentResult:
    """Result from a single agent session (one patient × one task × one arm)."""
    prediction: dict | None  # Parsed submission JSON, or None if not submitted
    turns_used: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: int
    trace: list[dict] = field(default_factory=list)
    submitted: bool = False
    budget_exhausted: bool = False
    # True when the answer came from the forced-submit turn granted after the
    # action budget ran out, rather than from the agent choosing to submit.
    forced_submission: bool = False


@dataclass
class Action:
    """Parsed agent action from model output."""
    type: str  # 'tool' | 'command' | 'submit' | 'none'
    payload: dict | str | None = None  # tool args dict, command string, or submission dict
    tool_name: str | None = None  # For structured arm
