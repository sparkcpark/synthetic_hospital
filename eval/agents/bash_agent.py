"""Bash agent harness: command validation, Docker exec, output capture.

Manages bash agent sessions in a restricted Docker container with read-only
PostgreSQL access (9 whitelisted clinical tables) and limited shell commands.
"""

import logging
import re
import subprocess
import time

log = logging.getLogger(__name__)

# 9 clinical tables the bash agent can SELECT from
ALLOWED_TABLES = frozenset({
    "longitudinal_patients",
    "longitudinal_encounters",
    "encounter_ehr_sections",
    "diagnoses",
    "clinical_findings",
    "diagnosis_findings",
    "fact_cards",
    "imaging_orders",
    "terminology_codes",
})

# Whitelisted binaries
ALLOWED_BINARIES = frozenset({
    "psql", "echo", "grep", "jq", "cat", "head", "tail", "wc", "curl",
})

# SQL write keywords (case-insensitive)
SQL_WRITE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|COPY)\b",
    re.IGNORECASE,
)

# Extract binary name from a command (first word, handling pipes)
BINARY_RE = re.compile(r"^\s*(\S+)")

# Table name pattern for SQL queries
TABLE_RE = re.compile(
    r"\b(FROM|JOIN|INTO|TABLE|UPDATE)\s+(\w+)",
    re.IGNORECASE,
)


class BashAgentHarness:
    """Manages bash agent sessions: command validation, execution, output capture."""

    def __init__(
        self,
        container_id: str,
        api_base: str,
        auth_token: str,
        max_commands: int = 40,
        max_output_chars: int = 8000,
    ):
        self.container_id = container_id
        self.api_base = api_base
        self.auth_token = auth_token
        self.max_commands = max_commands
        self.max_output_chars = max_output_chars
        self.command_log: list[dict] = []
        self._budget_used = 0  # Only counts non-blocked commands

    def validate_command(self, command: str) -> tuple[bool, str | None]:
        """Check command against whitelist + SQL write keywords.

        Returns (allowed, warning_or_rejection_reason):
        - (True, None): command is clean
        - (True, warning_str): command references non-whitelisted table (warn, allow)
        - (False, rejection_str): command is blocked
        """
        # 1. Check for SQL write keywords
        if SQL_WRITE_RE.search(command):
            return False, "SQL write operations (INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE/COPY) are not allowed."

        # 2. Check binaries in pipeline
        segments = re.split(r"\|", command)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            m = BINARY_RE.match(seg)
            if m:
                binary = m.group(1).split("/")[-1]  # handle /usr/bin/psql
                if binary not in ALLOWED_BINARIES:
                    return False, f"Binary '{binary}' is not in the allowed set: {sorted(ALLOWED_BINARIES)}"

        # 3. Check curl target
        if "curl" in command:
            # Allow only curl to the submission API endpoint
            if self.api_base not in command:
                return False, f"curl is only allowed to the submission endpoint ({self.api_base})"

        # 4. Check for non-whitelisted tables (warn, don't block)
        warnings = []
        for match in TABLE_RE.finditer(command):
            table_name = match.group(2).lower()
            if table_name not in ALLOWED_TABLES and not table_name.startswith("pg_"):
                warnings.append(table_name)

        if warnings:
            return True, (
                f"[WARNING: Table(s) {warnings} not in your allowed table set. "
                f"The query will likely fail with 'permission denied'.]"
            )

        return True, None

    def execute(self, command: str) -> tuple[str, int, dict]:
        """Validate and execute command in Docker container.

        Returns (output, exit_code, log_entry).
        Blocked commands return error message without counting against budget.
        """
        t0 = time.monotonic()

        # Validate
        allowed, message = self.validate_command(command)

        if not allowed:
            entry = {
                "command": command,
                "output": f"[BLOCKED] {message}",
                "exit_code": 1,
                "latency_ms": 0,
                "truncated": False,
                "blocked": True,
                "warned": False,
            }
            self.command_log.append(entry)
            return f"[BLOCKED] {message}", 1, entry

        # Execute via docker exec
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_id, "sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout + result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            output = "[TIMEOUT] Command exceeded 30 second limit."
            exit_code = 124
        except Exception as e:
            output = f"[ERROR] {e}"
            exit_code = 1

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Truncate output
        truncated = len(output) > self.max_output_chars
        if truncated:
            output = output[:self.max_output_chars] + "\n[OUTPUT TRUNCATED — use LIMIT, head, or WHERE to narrow results]"

        # Prepend warning if applicable
        warned = message is not None
        if warned:
            output = f"{message}\n{output}"

        # Count against budget
        self._budget_used += 1

        entry = {
            "command": command,
            "output": output,
            "exit_code": exit_code,
            "latency_ms": latency_ms,
            "truncated": truncated,
            "blocked": False,
            "warned": warned,
        }
        self.command_log.append(entry)

        return output, exit_code, entry

    def get_trace(self) -> list[dict]:
        """Return full command trace for analysis."""
        return list(self.command_log)

    def remaining_budget(self) -> int:
        """Commands remaining in budget (excludes blocked commands)."""
        return self.max_commands - self._budget_used
