"""add logical slot to question benchmark outputs

Revision ID: b2d8f4a6c1e9
Revises: a1c7d5e9f3b4
Create Date: 2026-09-24 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2d8f4a6c1e9"
down_revision: Union[str, None] = "a1c7d5e9f3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "question_benchmark_outputs",
        sa.Column("logical_slot", sa.String(length=30), nullable=True),
    )
    op.execute(
        """
        UPDATE question_benchmark_outputs
        SET logical_slot = CASE
            WHEN provider = 'groq' THEN 'slot_groq'
            WHEN provider = 'cloudflare' THEN 'slot_cloudflare'
            WHEN provider IN ('cohere', 'inception') THEN 'slot_third_ai'
            WHEN provider IN ('huggingface', 'opentyphoon') THEN 'slot_fourth_ai'
            ELSE logical_slot
        END
        """
    )
    op.create_index(
        "ix_question_benchmark_outputs_logical_slot",
        "question_benchmark_outputs",
        ["logical_slot"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_question_benchmark_outputs_logical_slot",
        table_name="question_benchmark_outputs",
    )
    op.drop_column("question_benchmark_outputs", "logical_slot")
