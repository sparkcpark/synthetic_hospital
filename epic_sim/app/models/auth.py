"""Auth models for RBAC and partial observability."""

import uuid

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base


class AuthUser(Base):
    __tablename__ = "auth_users"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    department: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    token_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("auth_users.user_id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    expires_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())


class PatientAssignment(Base):
    __tablename__ = "patient_assignments"

    assignment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("auth_users.user_id"), nullable=False)
    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("longitudinal_patients.patient_id"), nullable=False
    )
    department: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "patient_id", name="uq_pa_user_patient"),
        Index("idx_pa_user", "user_id"),
        Index("idx_pa_patient", "patient_id"),
    )


class AgentSubmission(Base):
    __tablename__ = "agent_submissions"

    submission_id: Mapped[str] = mapped_column(
        Text, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    patient_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("longitudinal_patients.patient_id")
    )
    encounter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("longitudinal_encounters.encounter_id")
    )
    gt_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("benchmark_ground_truth.gt_id")
    )
    submission_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[str] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_agent_sub_session", "session_id"),
    )
