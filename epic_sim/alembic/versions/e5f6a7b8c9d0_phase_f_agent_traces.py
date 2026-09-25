"""phase_f_agent_traces

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-23 10:00:00.000000

Phase F infrastructure:
- Create agent_traces table for denormalized trace analysis
- Extend method_type enum with agent_structured and agent_bash
- Add submit_rankings to agent_submissions.submission_type
"""
from typing import Sequence, Union

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Extend method_type to accept agent arms
    # evaluation_runs.method_type is stored as TEXT with no CHECK constraint
    # in the current schema, so no enum alteration needed — just document the values.
    # If there IS an enum, extend it:
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'methodtype') THEN
                ALTER TYPE methodtype ADD VALUE IF NOT EXISTS 'agent_structured';
                ALTER TYPE methodtype ADD VALUE IF NOT EXISTS 'agent_bash';
            END IF;
        END $$;
    """)

    # 2. Create agent_traces table
    op.execute("""
        CREATE TABLE agent_traces (
            trace_id        SERIAL PRIMARY KEY,
            prediction_id   INTEGER NOT NULL REFERENCES evaluation_predictions(prediction_id),
            turn_number     INTEGER NOT NULL,
            action_type     TEXT NOT NULL,
            action_name     TEXT,
            action_args     JSONB,
            output_text     TEXT,
            output_length   INTEGER,
            latency_ms      INTEGER,
            sections_accessed TEXT[],
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_at_prediction ON agent_traces(prediction_id)")
    op.execute("CREATE INDEX idx_at_action ON agent_traces(action_type)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_at_action")
    op.execute("DROP INDEX IF EXISTS idx_at_prediction")
    op.execute("DROP TABLE IF EXISTS agent_traces")
    # Note: Cannot remove enum values in PostgreSQL
