"""add_loinc_to_clinical_findings

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-03 16:00:00.000000

Adds loinc_code and loinc_desc columns to clinical_findings table
for LOINC coding of lab_value findings.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("clinical_findings", sa.Column("loinc_code", sa.Text(), nullable=True))
    op.add_column("clinical_findings", sa.Column("loinc_desc", sa.Text(), nullable=True))
    op.create_index("idx_cf_loinc", "clinical_findings", ["loinc_code"])


def downgrade() -> None:
    op.drop_index("idx_cf_loinc", table_name="clinical_findings")
    op.drop_column("clinical_findings", "loinc_desc")
    op.drop_column("clinical_findings", "loinc_code")
