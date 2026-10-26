"""add debiased generation metadata

Revision ID: d5e8f1a3c7b9
Revises: c4e1a7b9d2f5
Create Date: 2026-09-30 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d5e8f1a3c7b9"
down_revision: Union[str, None] = "c4e1a7b9d2f5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "question_benchmark_outputs",
        sa.Column("generation_metadata_json", sa.JSON(), nullable=True),
    )
    op.add_column(
        "synthetic_answer_samples",
        sa.Column("generation_metadata_json", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("synthetic_answer_samples", "generation_metadata_json")
    op.drop_column("question_benchmark_outputs", "generation_metadata_json")
