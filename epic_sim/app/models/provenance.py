"""Layer 0: Provenance & raw ingestion models."""

from sqlalchemy import BigInteger, Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base
from epic_sim.app.models.enums import CardType, ClassificationMethod, DeckType


class SourceDeck(Base):
    __tablename__ = "source_decks"

    deck_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    deck_name: Mapped[str | None] = mapped_column(Text)
    deck_type: Mapped[DeckType] = mapped_column(Enum(DeckType, name="deck_type"), nullable=False)
    anki_db_version: Mapped[str | None] = mapped_column(Text)
    note_count: Mapped[int | None] = mapped_column(Integer)
    card_count: Mapped[int | None] = mapped_column(Integer)
    model_names: Mapped[str | None] = mapped_column(Text)
    deck_hierarchy: Mapped[str | None] = mapped_column(Text)
    ingested_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    sha256_hash: Mapped[str | None] = mapped_column(String(64))

    raw_cards: Mapped[list["RawCard"]] = relationship(back_populates="deck")


class RawCard(Base):
    __tablename__ = "raw_cards"

    raw_card_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    deck_id: Mapped[int] = mapped_column(Integer, ForeignKey("source_decks.deck_id"), nullable=False)
    anki_note_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    anki_card_id: Mapped[int | None] = mapped_column(BigInteger)
    anki_model_name: Mapped[str | None] = mapped_column(Text)
    anki_deck_name: Mapped[str | None] = mapped_column(Text)
    anki_tags: Mapped[str | None] = mapped_column(Text)
    field_data: Mapped[str] = mapped_column(Text, nullable=False)
    field_data_text: Mapped[str | None] = mapped_column(Text)
    media_refs: Mapped[str | None] = mapped_column(Text)
    card_ordinal: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    card_type: Mapped[CardType | None] = mapped_column(Enum(CardType, name="card_type"))
    card_format: Mapped[str | None] = mapped_column(Text)
    classification_method: Mapped[ClassificationMethod | None] = mapped_column(
        Enum(ClassificationMethod, name="classification_method")
    )

    deck: Mapped["SourceDeck"] = relationship(back_populates="raw_cards")

    __table_args__ = (
        UniqueConstraint("deck_id", "anki_note_id", "card_ordinal", name="uq_raw_cards_deck_note_ord"),
        Index("idx_raw_cards_deck", "deck_id"),
        Index("idx_raw_cards_anki_note", "anki_note_id"),
    )
