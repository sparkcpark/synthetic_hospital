"""Layer 4: EHR section parsing models."""

from sqlalchemy import Enum, ForeignKey, Index, Integer, SmallInteger, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base
from epic_sim.app.models.enums import ExtractionMethod, SectionType


class EhrSection(Base):
    __tablename__ = "ehr_sections"

    section_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("board_questions.question_id"), nullable=False)
    section_type: Mapped[SectionType] = mapped_column(Enum(SectionType, name="section_type"), nullable=False)
    section_text: Mapped[str] = mapped_column(Text, nullable=False)
    section_order: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    extraction_method: Mapped[ExtractionMethod] = mapped_column(
        Enum(ExtractionMethod, name="extraction_method"), nullable=False
    )
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ehr_question", "question_id"),
        Index("idx_ehr_type", "section_type"),
    )
