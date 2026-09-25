"""Auth router — token issuance and user registration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.auth.oauth2 import (
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    CurrentUser,
)
from epic_sim.app.auth.scopes import scopes_for_role
from epic_sim.app.models.base import get_db

router = APIRouter()


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str
    role: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str
    display_name: str
    department: str | None = None


class RegisterResponse(BaseModel):
    user_id: str
    username: str
    role: str
    display_name: str
    department: str | None


class UserInfoResponse(BaseModel):
    user_id: str
    username: str
    role: str
    department: str | None
    scopes: list[str]


@router.post("/token", response_model=TokenResponse)
async def login(body: TokenRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate and issue a JWT access token (OAuth2 password grant)."""
    user = await authenticate_user(db, body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scopes = sorted(scopes_for_role(user.role))
    from epic_sim.app.config import settings

    token = create_access_token(
        user_id=str(user.user_id),
        username=user.username,
        role=user.role,
        scopes=scopes,
        department=user.department,
    )

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        scope=" ".join(scopes),
        role=user.role,
    )


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user (for evaluation setup)."""
    valid_roles = {"attending", "resident", "nurse", "radiologist", "lab_tech", "pharmacist"}
    if body.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(sorted(valid_roles))}",
        )

    try:
        user = await create_user(
            db,
            username=body.username,
            password=body.password,
            role=body.role,
            display_name=body.display_name,
            department=body.department,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' already exists",
        )

    return RegisterResponse(
        user_id=str(user.user_id),
        username=user.username,
        role=user.role,
        display_name=user.display_name,
        department=user.department,
    )


@router.get("/me", response_model=UserInfoResponse)
async def user_info(current_user: CurrentUser = Depends(get_current_user)):
    """Return current user info from the JWT token."""
    return UserInfoResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        department=current_user.department,
        scopes=current_user.scopes,
    )
