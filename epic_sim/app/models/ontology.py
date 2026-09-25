"""Layer 2: Ontology & clinical coding models."""

from sqlalchemy import Boolean, Enum, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base
from epic_sim.app.models.enums import Acuity, FindingType


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    diagnosis_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    icd10_code: Mapped[str | None] = mapped_column(Text)
    icd10_desc: Mapped[str | None] = mapped_column(Text)
    snomed_id: Mapped[str | None] = mapped_column(Text)
    snomed_desc: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(Text)
    acuity: Mapped[Acuity | None] = mapped_column(Enum(Acuity, name="acuity"))
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("icd10_code", "snomed_id", name="uq_diagnoses_icd10_snomed"),
        Index("idx_dx_icd10", "icd10_code"),
        Index("idx_dx_snomed", "snomed_id"),
        Index("idx_dx_category", "category"),
    )


class ClinicalFinding(Base):
    __tablename__ = "clinical_findings"

    finding_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snomed_id: Mapped[str | None] = mapped_column(Text)
    snomed_desc: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    finding_type: Mapped[FindingType] = mapped_column(Enum(FindingType, name="finding_type"), nullable=False)
    normal_range: Mapped[str | None] = mapped_column(Text)
    loinc_code: Mapped[str | None] = mapped_column(Text)
    loinc_desc: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_cf_snomed", "snomed_id"),
        Index("idx_cf_type", "finding_type"),
        Index("idx_cf_loinc", "loinc_code"),
    )


class TerminologyCode(Base):
    __tablename__ = "terminology_codes"

    code_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    system: Mapped[str] = mapped_column(Text, nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    display: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    properties: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        UniqueConstraint("system", "code", name="uq_terminology_system_code"),
        Index("idx_term_system_code", "system", "code"),
    )
