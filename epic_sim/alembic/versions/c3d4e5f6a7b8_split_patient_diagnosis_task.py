"""split_patient_diagnosis_task

Revision ID: c3d4e5f6a7b8
Revises: a1b2c3d4e5f6
Create Date: 2026-03-07 10:00:00.000000

Splits patient-level diagnosis items into a new 'patient_diagnosis' task:
- Updates CHECK constraints on benchmark_ground_truth.task and evaluation_runs.task
- Migrates existing patient-level rows from 'diagnosis_accuracy' to 'patient_diagnosis'
- Adds composite index on (task, granularity)
"""
from typing import Sequence, Union

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TASKS_NEW = (
    "'diagnosis_accuracy', 'patient_diagnosis', 'context_summarization', "
    "'evidence_retrieval', 'imaging_indication'"
)
TASKS_OLD = (
    "'diagnosis_accuracy', 'context_summarization', "
    "'evidence_retrieval', 'imaging_indication'"
)


def upgrade() -> None:
    # 0. The task columns are the eval_task enum; the new value must exist (and be committed)
    #    before any CHECK constraint or UPDATE refers to it. On the original database it was
    #    added by hand, which left fresh installs unable to run this migration.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE eval_task ADD VALUE IF NOT EXISTS 'patient_diagnosis'")

    # 1. Update benchmark_ground_truth task CHECK
    op.execute("ALTER TABLE benchmark_ground_truth DROP CONSTRAINT IF EXISTS benchmark_ground_truth_task_check")
    op.execute(
        f"ALTER TABLE benchmark_ground_truth ADD CONSTRAINT benchmark_ground_truth_task_check "
        f"CHECK (task IN ({TASKS_NEW}))"
    )

    # 2. Update evaluation_runs task CHECK
    op.execute("ALTER TABLE evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_task_check")
    op.execute(
        f"ALTER TABLE evaluation_runs ADD CONSTRAINT evaluation_runs_task_check "
        f"CHECK (task IN ({TASKS_NEW}))"
    )

    # 3. Migrate patient-level rows to new task
    op.execute("""
        UPDATE benchmark_ground_truth
        SET task = 'patient_diagnosis'
        WHERE task = 'diagnosis_accuracy' AND granularity = 'patient'
    """)

    # 4. Add composite index for task+granularity queries
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_bgt_task_granularity "
        "ON benchmark_ground_truth(task, granularity)"
    )


def downgrade() -> None:
    # Reverse migration
    op.execute("""
        UPDATE benchmark_ground_truth
        SET task = 'diagnosis_accuracy'
        WHERE task = 'patient_diagnosis'
    """)

    op.execute("DROP INDEX IF EXISTS idx_bgt_task_granularity")

    op.execute("ALTER TABLE evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_task_check")
    op.execute(
        f"ALTER TABLE evaluation_runs ADD CONSTRAINT evaluation_runs_task_check "
        f"CHECK (task IN ({TASKS_OLD}))"
    )

    op.execute("ALTER TABLE benchmark_ground_truth DROP CONSTRAINT IF EXISTS benchmark_ground_truth_task_check")
    op.execute(
        f"ALTER TABLE benchmark_ground_truth ADD CONSTRAINT benchmark_ground_truth_task_check "
        f"CHECK (task IN ({TASKS_OLD}))"
    )
