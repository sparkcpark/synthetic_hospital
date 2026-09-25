"""Layer 1: Structured medical content models."""

from sqlalchemy import Enum, Float, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base
from epic_sim.app.models.enums import ExtractionMethod, FactFormat, QuestionFormat


class BoardQuestion(Base):
    __tablename__ = "board_questions"

    question_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    raw_card_id: Mapped[int] = mapped_column(Integer, ForeignKey("raw_cards.raw_card_id"), nullable=False)
    source_qid: Mapped[str | None] = mapped_column(Text)
    question_format: Mapped[QuestionFormat] = mapped_column(
        Enum(QuestionFormat, name="question_format"), nullable=False
    )
    vignette_text: Mapped[str | None] = mapped_column(Text)
    vignette_html: Mapped[str | None] = mapped_column(Text)
    question_stem: Mapped[str | None] = mapped_column(Text)
    clinical_setting: Mapped[str | None] = mapped_column(Text)
    answer_choices: Mapped[str | None] = mapped_column(Text)
    correct_answer: Mapped[str | None] = mapped_column(Text)
    correct_explanation: Mapped[str | None] = mapped_column(Text)
    distractor_explanations: Mapped[str | None] = mapped_column(Text)
    cloze_raw: Mapped[str | None] = mapped_column(Text)
    cloze_answers: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    organ_system: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    step_level: Mapped[str | None] = mapped_column(Text)
    difficulty: Mapped[str | None] = mapped_column(Text)
    tags_normalized: Mapped[str | None] = mapped_column(Text)
    extraction_method: Mapped[ExtractionMethod] = mapped_column(
        Enum(ExtractionMethod, name="extraction_method"), nullable=False
    )
    extraction_model: Mapped[str | None] = mapped_column(Text)
    extraction_confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_bq_format", "question_format"),
        Index("idx_bq_subject", "subject"),
        Index("idx_bq_system", "organ_system"),
        Index("idx_bq_step", "step_level"),
    )


class FactCard(Base):
    __tablename__ = "fact_cards"

    fact_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    raw_card_id: Mapped[int] = mapped_column(Integer, ForeignKey("raw_cards.raw_card_id"), nullable=False)
    fact_format: Mapped[FactFormat] = mapped_column(Enum(FactFormat, name="fact_format"), nullable=False)
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)
    fact_text_cloze: Mapped[str | None] = mapped_column(Text)
    fact_context: Mapped[str | None] = mapped_column(Text)
    fact_summary: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    organ_system: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    specialty: Mapped[str | None] = mapped_column(Text)
    resource_refs: Mapped[str | None] = mapped_column(Text)
    tags_normalized: Mapped[str | None] = mapped_column(Text)
    extraction_method: Mapped[ExtractionMethod] = mapped_column(
        Enum(ExtractionMethod, name="extraction_method"), nullable=False
    )
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_fc_subject", "subject"),
        Index("idx_fc_specialty", "specialty"),
    )
