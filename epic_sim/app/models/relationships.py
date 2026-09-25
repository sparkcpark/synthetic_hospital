"""Layer 3: Relationship mapping models."""

from sqlalchemy import Enum, Float, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from epic_sim.app.models.base import Base
from epic_sim.app.models.enums import (
    DataSource,
    DxFindingRelationship,
    DxRole,
    FactCfRelevance,
    FactDxRelevance,
    FindingRelevance,
)


class QuestionDiagnosis(Base):
    __tablename__ = "question_diagnoses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("board_questions.question_id"), nullable=False)
    diagnosis_id: Mapped[int] = mapped_column(Integer, ForeignKey("diagnoses.diagnosis_id"), nullable=False)
    role: Mapped[DxRole] = mapped_column(Enum(DxRole, name="dx_role"), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[DataSource] = mapped_column(Enum(DataSource, name="data_source"), nullable=False)

    __table_args__ = (
        UniqueConstraint("question_id", "diagnosis_id", "role", name="uq_qd_question_dx_role"),
        Index("idx_qd_question", "question_id"),
        Index("idx_qd_diagnosis", "diagnosis_id"),
    )


class QuestionFinding(Base):
    __tablename__ = "question_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("board_questions.question_id"), nullable=False)
    finding_id: Mapped[int] = mapped_column(Integer, ForeignKey("clinical_findings.finding_id"), nullable=False)
    present: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    value_text: Mapped[str | None] = mapped_column(Text)
    value_numeric: Mapped[float | None] = mapped_column(Float)
    ehr_section: Mapped[str | None] = mapped_column(Text)
    relevance: Mapped[FindingRelevance | None] = mapped_column(Enum(FindingRelevance, name="finding_relevance"))
    source: Mapped[DataSource] = mapped_column(Enum(DataSource, name="data_source"), nullable=False)

    __table_args__ = (
        UniqueConstraint("question_id", "finding_id", name="uq_qf_question_finding"),
        Index("idx_qf_question", "question_id"),
        Index("idx_qf_finding", "finding_id"),
    )


class DiagnosisFinding(Base):
    __tablename__ = "diagnosis_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    diagnosis_id: Mapped[int] = mapped_column(Integer, ForeignKey("diagnoses.diagnosis_id"), nullable=False)
    finding_id: Mapped[int] = mapped_column(Integer, ForeignKey("clinical_findings.finding_id"), nullable=False)
    relationship: Mapped[DxFindingRelationship] = mapped_column(
        Enum(DxFindingRelationship, name="dx_finding_rel"), nullable=False
    )
    frequency: Mapped[float | None] = mapped_column(Float)
    evidence_source: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("diagnosis_id", "finding_id", "relationship", name="uq_df_dx_finding_rel"),
        Index("idx_df_diagnosis", "diagnosis_id"),
        Index("idx_df_finding", "finding_id"),
    )


class FactDiagnosisLink(Base):
    __tablename__ = "fact_diagnosis_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fact_id: Mapped[int] = mapped_column(Integer, ForeignKey("fact_cards.fact_id"), nullable=False)
    diagnosis_id: Mapped[int] = mapped_column(Integer, ForeignKey("diagnoses.diagnosis_id"), nullable=False)
    relevance: Mapped[FactDxRelevance | None] = mapped_column(Enum(FactDxRelevance, name="fact_dx_relevance"))

    __table_args__ = (
        UniqueConstraint("fact_id", "diagnosis_id", name="uq_fdl_fact_dx"),
    )


class FactFindingLink(Base):
    __tablename__ = "fact_finding_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fact_id: Mapped[int] = mapped_column(Integer, ForeignKey("fact_cards.fact_id"), nullable=False)
    finding_id: Mapped[int] = mapped_column(Integer, ForeignKey("clinical_findings.finding_id"), nullable=False)
    relevance: Mapped[FactCfRelevance | None] = mapped_column(Enum(FactCfRelevance, name="fact_cf_relevance"))

    __table_args__ = (
        UniqueConstraint("fact_id", "finding_id", name="uq_ffl_fact_finding"),
    )
