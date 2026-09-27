"""use generic logical slots for the dynamic provider pool

Revision ID: c4e1a7b9d2f5
Revises: b2d8f4a6c1e9
Create Date: 2026-09-24 17:30:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "c4e1a7b9d2f5"
down_revision: Union[str, None] = "b2d8f4a6c1e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE question_benchmark_outputs
        SET logical_slot = CASE logical_slot
            WHEN 'slot_groq' THEN 'slot_1'
            WHEN 'slot_cloudflare' THEN 'slot_2'
            WHEN 'slot_third_ai' THEN 'slot_3'
            WHEN 'slot_fourth_ai' THEN 'slot_4'
            ELSE logical_slot
        END
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE question_benchmark_outputs
        SET logical_slot = CASE logical_slot
            WHEN 'slot_1' THEN 'slot_groq'
            WHEN 'slot_2' THEN 'slot_cloudflare'
            WHEN 'slot_3' THEN 'slot_third_ai'
            WHEN 'slot_4' THEN 'slot_fourth_ai'
            ELSE logical_slot
        END
        """
    )
