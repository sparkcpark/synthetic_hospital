"""phase_e_eval_schema

Revision ID: a1b2c3d4e5f6
Revises: 2f90ee19c4d4
Create Date: 2026-03-02 16:00:00.000000

Adds:
- evaluation_predictions: ALTER score INT→FLOAT, ADD prompt_strategy/input_tokens/output_tokens, ADD UNIQUE(run_id,gt_id)
- evaluation_runs: ADD prompt_strategy, model_name
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "2f90ee19c4d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # evaluation_predictions: ALTER score from INTEGER to FLOAT
    op.alter_column(
        "evaluation_predictions", "score",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    # evaluation_predictions: ADD new columns
    op.add_column("evaluation_predictions", sa.Column("input_tokens", sa.Integer(), nullable=True))
    op.add_column("evaluation_predictions", sa.Column("output_tokens", sa.Integer(), nullable=True))
    op.add_column("evaluation_predictions", sa.Column("prompt_strategy", sa.Text(), nullable=True))
    # evaluation_predictions: ADD UNIQUE constraint for resume support
    op.create_unique_constraint("uq_ep_run_gt", "evaluation_predictions", ["run_id", "gt_id"])

    # evaluation_runs: ADD new columns
    op.add_column("evaluation_runs", sa.Column("prompt_strategy", sa.Text(), nullable=True))
    op.add_column("evaluation_runs", sa.Column("model_name", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("evaluation_runs", "model_name")
    op.drop_column("evaluation_runs", "prompt_strategy")
    op.drop_constraint("uq_ep_run_gt", "evaluation_predictions", type_="unique")
    op.drop_column("evaluation_predictions", "prompt_strategy")
    op.drop_column("evaluation_predictions", "output_tokens")
    op.drop_column("evaluation_predictions", "input_tokens")
    op.alter_column(
        "evaluation_predictions", "score",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
