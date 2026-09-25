"""Layer 5: Longitudinal record generation models."""

from sqlalchemy import Enum, ForeignKey, Index, Integer, SmallInteger, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base
from epic_sim.app.models.enums import EncounterType, GenerationMethod


class LongitudinalPatient(Base):
    __tablename__ = "longitudinal_patients"

    patient_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile: Mapped[dict] = mapped_column(JSONB, nullable=False)
    age: Mapped[int | None] = mapped_column(SmallInteger)
    sex: Mapped[str | None] = mapped_column(Text)
    race_ethnicity: Mapped[str | None] = mapped_column(Text)
    insurance: Mapped[str | None] = mapped_column(Text)
    pcp_name: Mapped[str | None] = mapped_column(Text)
    num_encounters: Mapped[int] = mapped_column(Integer, default=0)
    primary_diagnoses: Mapped[str | None] = mapped_column(Text)
    comorbidities: Mapped[str | None] = mapped_column(Text)
    generation_seed: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    encounters: Mapped[list["LongitudinalEncounter"]] = relationship(back_populates="patient")


class LongitudinalEncounter(Base):
    __tablename__ = "longitudinal_encounters"

    encounter_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("longitudinal_patients.patient_id"), nullable=False
    )
    encounter_date: Mapped[str] = mapped_column(Text, nullable=False)
    encounter_type: Mapped[EncounterType] = mapped_column(
        Enum(EncounterType, name="encounter_type"), nullable=False
    )
    chief_complaint: Mapped[str | None] = mapped_column(Text)
    attending_name: Mapped[str | None] = mapped_column(Text)
    department: Mapped[str | None] = mapped_column(Text)
    source_question_ids: Mapped[str | None] = mapped_column(Text)
    encounter_order: Mapped[int] = mapped_column(Integer, nullable=False)
    note_text: Mapped[str | None] = mapped_column(Text)
    generation_method: Mapped[GenerationMethod | None] = mapped_column(
        Enum(GenerationMethod, name="generation_method")
    )
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    patient: Mapped["LongitudinalPatient"] = relationship(back_populates="encounters")
    sections: Mapped[list["EncounterEhrSection"]] = relationship(back_populates="encounter")

    __table_args__ = (
        Index("idx_le_patient", "patient_id"),
        Index("idx_le_date", "encounter_date"),
    )


class EncounterEhrSection(Base):
    __tablename__ = "encounter_ehr_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    encounter_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("longitudinal_encounters.encounter_id"), nullable=False
    )
    section_type: Mapped[str] = mapped_column(Text, nullable=False)
    section_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_section_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("ehr_sections.section_id"))
    is_modified: Mapped[int] = mapped_column(Integer, default=0)
    section_order: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    encounter: Mapped["LongitudinalEncounter"] = relationship(back_populates="sections")

    __table_args__ = (
        Index("idx_ees_encounter", "encounter_id"),
    )
