"""Redis-backed session management for agent evaluation.

Sessions are stored as JSON blobs in Redis with auto-expiry (TTL).
Supports rate limiting, tool call tracing, and graceful degradation
when Redis is unavailable.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request

from epic_sim.app.config import settings

log = logging.getLogger(__name__)


def _key(session_id: str) -> str:
    return f"{settings.redis_session_prefix}{session_id}"


class SessionData:
    """In-memory representation of a Redis session."""

    def __init__(
        self,
        session_id: str,
        patient_id: int | None = None,
        role: str = "attending",
        department: str = "Internal Medicine",
        user_id: str = "",
        gt_id: int | None = None,
        max_api_calls: int | None = None,
    ):
        self.session_id = session_id
        self.patient_id = patient_id
        self.role = role
        self.department = department
        self.user_id = user_id
        self.gt_id = gt_id
        self.api_call_count = 0
        self.max_api_calls = max_api_calls or settings.max_api_calls_per_session
        self.tool_calls: list[dict[str, Any]] = []
        self.status = "active"
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = now
        self.updated_at = now

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "role": self.role,
            "department": self.department,
            "user_id": self.user_id,
            "gt_id": self.gt_id,
            "api_call_count": self.api_call_count,
            "max_api_calls": self.max_api_calls,
            "tool_calls": self.tool_calls,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionData:
        s = cls(
            session_id=data["session_id"],
            patient_id=data.get("patient_id"),
            role=data.get("role", "attending"),
            department=data.get("department", ""),
            user_id=data.get("user_id", ""),
            gt_id=data.get("gt_id"),
            max_api_calls=data.get("max_api_calls"),
        )
        s.api_call_count = data.get("api_call_count", 0)
        s.tool_calls = data.get("tool_calls", [])
        s.status = data.get("status", "active")
        s.created_at = data.get("created_at", "")
        s.updated_at = data.get("updated_at", "")
        return s


async def create_session(
    redis: Any,
    patient_id: int | None,
    role: str,
    department: str,
    user_id: str,
    gt_id: int | None = None,
    max_api_calls: int | None = None,
) -> SessionData:
    """Create a new session in Redis."""
    session = SessionData(
        session_id=str(uuid.uuid4()),
        patient_id=patient_id,
        role=role,
        department=department,
        user_id=user_id,
        gt_id=gt_id,
        max_api_calls=max_api_calls,
    )
    await redis.setex(
        _key(session.session_id),
        settings.session_ttl_seconds,
        json.dumps(session.to_dict()),
    )
    return session


async def get_session(redis: Any, session_id: str) -> SessionData | None:
    """Retrieve session from Redis."""
    data = await redis.get(_key(session_id))
    if data is None:
        return None
    return SessionData.from_dict(json.loads(data))


async def record_tool_call(
    redis: Any,
    session_id: str,
    tool_name: str,
    args: dict,
    latency_ms: int,
) -> tuple[int, int]:
    """Record a tool call. Returns (new_count, remaining). Raises 429 if limit exceeded."""
    session = await get_session(redis, session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found or expired")

    if session.api_call_count >= session.max_api_calls:
        session.status = "rate_limited"
        await redis.setex(
            _key(session_id),
            settings.session_ttl_seconds,
            json.dumps(session.to_dict()),
        )
        raise HTTPException(429, "Rate limit exceeded")

    session.api_call_count += 1
    session.tool_calls.append({
        "sequence": session.api_call_count,
        "tool_name": tool_name,
        "arguments": args,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    session.updated_at = datetime.now(timezone.utc).isoformat()

    await redis.setex(
        _key(session_id),
        settings.session_ttl_seconds,
        json.dumps(session.to_dict()),
    )
    remaining = session.max_api_calls - session.api_call_count
    return session.api_call_count, remaining


async def finalize_session(redis: Any, session_id: str) -> SessionData | None:
    """Mark session as completed, return final state."""
    session = await get_session(redis, session_id)
    if session is None:
        return None
    session.status = "completed"
    session.updated_at = datetime.now(timezone.utc).isoformat()
    await redis.setex(
        _key(session_id),
        settings.session_ttl_seconds,
        json.dumps(session.to_dict()),
    )
    return session


async def get_redis(request: Request):
    """FastAPI dependency: return Redis connection or None."""
    return getattr(request.app.state, "redis", None)
