"""Allow a source-free benchmark database and the v1.3 split labels (v1.3 release).

The released benchmark ships board-question metadata (subject, organ system, difficulty) without the
source cards it was extracted from. Relax the columns that tied every board question to a raw source
card so the release can be loaded into the simulator unchanged.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Split labels used by the v1.3 release (public = reported benchmark, heldout = private evaluation
    # set, train = training pool). 'val' and 'test' remain for databases built before v1.3.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE split_type ADD VALUE IF NOT EXISTS 'public'")
        op.execute("ALTER TYPE split_type ADD VALUE IF NOT EXISTS 'heldout'")
    op.drop_constraint("board_questions_raw_card_id_fkey", "board_questions", type_="foreignkey")
    op.alter_column("board_questions", "raw_card_id", existing_type=sa.Integer(), nullable=True)
    op.alter_column("board_questions", "extraction_method", existing_type=sa.String(), nullable=True)


def downgrade() -> None:
    op.alter_column("board_questions", "extraction_method", existing_type=sa.String(), nullable=False)
    op.alter_column("board_questions", "raw_card_id", existing_type=sa.Integer(), nullable=False)
    op.create_foreign_key("board_questions_raw_card_id_fkey", "board_questions", "raw_cards", ["raw_card_id"], ["raw_card_id"])
