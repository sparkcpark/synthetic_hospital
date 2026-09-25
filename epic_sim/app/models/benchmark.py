"""Layer 6: Benchmark evaluation models."""

from sqlalchemy import CheckConstraint, Enum, Float, ForeignKey, Index, Integer, SmallInteger, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base
from epic_sim.app.models.enums import (
    Difficulty,
    EvalTask,
    GtGranularity,
    MethodType,
    OrderPriority,
    PassageSource,
    SplitType,
)


class BenchmarkGroundTruth(Base):
    __tablename__ = "benchmark_ground_truth"

    gt_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task: Mapped[EvalTask] = mapped_column(Enum(EvalTask, name="eval_task"), nullable=False)
    granularity: Mapped[GtGranularity] = mapped_column(Enum(GtGranularity, name="gt_granularity"), nullable=False)
    question_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("board_questions.question_id"))
    patient_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("longitudinal_patients.patient_id"))
    encounter_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("longitudinal_encounters.encounter_id"))
    ground_truth: Mapped[dict] = mapped_column(JSONB, nullable=False)
    difficulty: Mapped[Difficulty | None] = mapped_column(Enum(Difficulty, name="difficulty"))
    split: Mapped[SplitType | None] = mapped_column(Enum(SplitType, name="split_type"))
    num_diagnoses: Mapped[int | None] = mapped_column(Integer)
    num_evidence: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "(granularity = 'question' AND question_id IS NOT NULL) OR "
            "(granularity = 'patient' AND patient_id IS NOT NULL) OR "
            "(granularity = 'encounter' AND encounter_id IS NOT NULL)",
            name="ck_bgt_granularity_id",
        ),
        Index("idx_bgt_task", "task"),
        Index("idx_bgt_split", "split"),
        Index("idx_bgt_question", "question_id"),
        Index("idx_bgt_patient", "patient_id"),
        Index("idx_bgt_encounter", "encounter_id"),
    )


class RelevanceJudgment(Base):
    __tablename__ = "relevance_judgments"

    judgment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    gt_id: Mapped[int] = mapped_column(Integer, ForeignKey("benchmark_ground_truth.gt_id"), nullable=False)
    passage_id: Mapped[str] = mapped_column(Text, nullable=False)
    passage_source: Mapped[PassageSource] = mapped_column(
        Enum(PassageSource, name="passage_source"), nullable=False
    )
    relevance_grade: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint("relevance_grade BETWEEN 0 AND 3", name="ck_rj_grade_range"),
        Index("idx_rj_gt", "gt_id"),
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_name: Mapped[str] = mapped_column(Text, nullable=False)
    method_name: Mapped[str] = mapped_column(Text, nullable=False)
    method_type: Mapped[MethodType] = mapped_column(Enum(MethodType, name="method_type"), nullable=False)
    task: Mapped[EvalTask] = mapped_column(Enum(EvalTask, name="eval_task"), nullable=False)
    split: Mapped[SplitType] = mapped_column(Enum(SplitType, name="split_type"), nullable=False)
    config: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    started_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[str | None] = mapped_column(TIMESTAMP(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)
    prompt_strategy: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(Text)


class EvaluationPrediction(Base):
    __tablename__ = "evaluation_predictions"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("evaluation_runs.run_id"), nullable=False)
    gt_id: Mapped[int] = mapped_column(Integer, ForeignKey("benchmark_ground_truth.gt_id"), nullable=False)
    prediction: Mapped[dict] = mapped_column(JSONB, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    prompt_strategy: Mapped[str | None] = mapped_column(Text)
    raw_output: Mapped[str | None] = mapped_column(Text)
    error_categories: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("run_id", "gt_id", name="uq_ep_run_gt"),
        Index("idx_ep_run", "run_id"),
        Index("idx_ep_gt", "gt_id"),
    )


class ImagingOrder(Base):
    __tablename__ = "imaging_orders"

    order_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    encounter_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("longitudinal_encounters.encounter_id"), nullable=False
    )
    gt_id: Mapped[int] = mapped_column(Integer, ForeignKey("benchmark_ground_truth.gt_id"), nullable=False)
    modality: Mapped[str] = mapped_column(Text, nullable=False)
    body_region: Mapped[str] = mapped_column(Text, nullable=False)
    clinical_indication: Mapped[str] = mapped_column(Text, nullable=False)
    ordering_provider: Mapped[str | None] = mapped_column(Text)
    order_priority: Mapped[OrderPriority | None] = mapped_column(Enum(OrderPriority, name="order_priority"))
    order_datetime: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_io_encounter", "encounter_id"),
        Index("idx_io_gt", "gt_id"),
    )
