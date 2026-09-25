"""phase_d_agent_submissions_tsvector

Revision ID: 2f90ee19c4d4
Revises: 334fe9b3010d
Create Date: 2026-03-02 11:53:44.348836

Adds:
- agent_submissions table for durable storage of agent predictions
- tsvector column + GIN index on encounter_ehr_sections for full-text search
- Trigger to auto-update tsvector on INSERT/UPDATE
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '2f90ee19c4d4'
down_revision: Union[str, None] = '334fe9b3010d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. agent_submissions table
    op.create_table('agent_submissions',
        sa.Column('submission_id', sa.Text(), nullable=False),
        sa.Column('session_id', sa.Text(), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=True),
        sa.Column('encounter_id', sa.Integer(), nullable=True),
        sa.Column('gt_id', sa.Integer(), nullable=True),
        sa.Column('submission_type', sa.Text(), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['encounter_id'], ['longitudinal_encounters.encounter_id']),
        sa.ForeignKeyConstraint(['gt_id'], ['benchmark_ground_truth.gt_id']),
        sa.ForeignKeyConstraint(['patient_id'], ['longitudinal_patients.patient_id']),
        sa.PrimaryKeyConstraint('submission_id'),
    )
    op.create_index('idx_agent_sub_session', 'agent_submissions', ['session_id'], unique=False)

    # 2. tsvector column on encounter_ehr_sections for full-text search
    op.add_column(
        'encounter_ehr_sections',
        sa.Column('search_vector', postgresql.TSVECTOR(), nullable=True),
    )

    # 3. Populate tsvector for existing rows
    op.execute(
        "UPDATE encounter_ehr_sections SET search_vector = to_tsvector('english', section_text)"
    )

    # 4. GIN index on the tsvector column
    op.create_index(
        'idx_ees_search',
        'encounter_ehr_sections',
        ['search_vector'],
        unique=False,
        postgresql_using='gin',
    )

    # 5. Trigger to auto-update tsvector on INSERT/UPDATE
    op.execute("""
        CREATE OR REPLACE FUNCTION ees_search_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english', NEW.section_text);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER trg_ees_search
            BEFORE INSERT OR UPDATE OF section_text ON encounter_ehr_sections
            FOR EACH ROW EXECUTE FUNCTION ees_search_update()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_ees_search ON encounter_ehr_sections")
    op.execute("DROP FUNCTION IF EXISTS ees_search_update()")
    op.drop_index('idx_ees_search', table_name='encounter_ehr_sections', postgresql_using='gin')
    op.drop_column('encounter_ehr_sections', 'search_vector')
    op.drop_index('idx_agent_sub_session', table_name='agent_submissions')
    op.drop_table('agent_submissions')
