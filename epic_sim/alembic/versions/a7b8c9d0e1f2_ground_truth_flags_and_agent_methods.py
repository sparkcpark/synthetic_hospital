"""Ground-truth exclusion flags and agent method types.

Two changes that had been applied to the original database by hand and were therefore missing
from a fresh install:
  * benchmark_ground_truth.is_diagnostic / exclusion_reason (set by scripts/apply_gt_exclusions.py;
    every task loader filters on is_diagnostic)
  * method_type enum values for the agentic conditions (an earlier migration looked for a type named
    'methodtype', so its ADD VALUE never ran against the actual 'method_type' enum)

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE benchmark_ground_truth ADD COLUMN IF NOT EXISTS is_diagnostic BOOLEAN NOT NULL DEFAULT TRUE")
    op.execute("ALTER TABLE benchmark_ground_truth ADD COLUMN IF NOT EXISTS exclusion_reason TEXT")
    with op.get_context().autocommit_block():
        for v in ("agent_structured", "agent_bash", "agent_context"):
            op.execute(f"ALTER TYPE method_type ADD VALUE IF NOT EXISTS '{v}'")


def downgrade() -> None:
    op.drop_column("benchmark_ground_truth", "exclusion_reason")
    op.drop_column("benchmark_ground_truth", "is_diagnostic")
    # enum values cannot be removed in PostgreSQL; left in place
