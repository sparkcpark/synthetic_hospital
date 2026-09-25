"""OAuth2 + JWT implementation for the Epic EHR simulation.

Provides:
- JWT token creation and validation
- FastAPI dependency for extracting the current user from bearer tokens
- User registration with password hashing
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.auth.scopes import scopes_for_role
from epic_sim.app.config import settings
from epic_sim.app.models.auth import AuthUser
from epic_sim.app.models.base import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(
    user_id: str,
    username: str,
    role: str,
    scopes: list[str],
    department: str | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.jwt_expire_minutes))
    payload: dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "role": role,
        "scopes": scopes,
        "department": department,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("sub") is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


class CurrentUser:
    """Parsed user context from a JWT token."""

    def __init__(self, payload: dict[str, Any]):
        self.user_id: str = payload["sub"]
        self.username: str = payload.get("username", "")
        self.role: str = payload.get("role", "")
        self.scopes: list[str] = payload.get("scopes", [])
        self.department: str | None = payload.get("department")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> CurrentUser:
    """FastAPI dependency: extract and validate the current user from the bearer token."""
    payload = decode_token(token)
    return CurrentUser(payload)


async def get_optional_user(
    token: str | None = Depends(oauth2_scheme),
) -> CurrentUser | None:
    """FastAPI dependency: optionally extract user (for endpoints that work with or without auth)."""
    if token is None:
        return None
    try:
        payload = decode_token(token)
        return CurrentUser(payload)
    except HTTPException:
        return None


async def authenticate_user(
    db: AsyncSession,
    username: str,
    password: str,
) -> AuthUser | None:
    """Verify username/password against the database."""
    result = await db.execute(select(AuthUser).where(AuthUser.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    if not user.is_active:
        return None
    return user


async def create_user(
    db: AsyncSession,
    username: str,
    password: str,
    role: str,
    display_name: str,
    department: str | None = None,
) -> AuthUser:
    """Create a new user with hashed password."""
    user = AuthUser(
        username=username,
        password_hash=hash_password(password),
        role=role,
        display_name=display_name,
        department=department,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
