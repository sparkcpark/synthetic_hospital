"""add_error_categories

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-09 10:00:00.000000

Adds error_categories JSONB column + GIN index to evaluation_predictions.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE evaluation_predictions ADD COLUMN error_categories JSONB")
    op.execute(
        "CREATE INDEX idx_ep_errors ON evaluation_predictions "
        "USING gin (error_categories)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_ep_errors")
    op.execute("ALTER TABLE evaluation_predictions DROP COLUMN IF EXISTS error_categories")
