"""add interview evidence summaries

Revision ID: f7a2c5d9e3b8
Revises: e6f1a4c8d2b7
Create Date: 2026-10-01 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f7a2c5d9e3b8"
down_revision: Union[str, None] = "e6f1a4c8d2b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "interview_evidence_summaries",
        sa.Column("summary_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("aggregator_version", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("competency_scores_json", sa.JSON(), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("flags_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["interview_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("summary_id"),
        sa.UniqueConstraint(
            "session_id",
            "aggregator_version",
            name="uq_interview_evidence_session_version",
        ),
    )
    op.create_index(
        "ix_interview_evidence_summaries_summary_id",
        "interview_evidence_summaries",
        ["summary_id"],
    )
    op.create_index(
        "ix_interview_evidence_summaries_session_id",
        "interview_evidence_summaries",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_interview_evidence_summaries_session_id",
        table_name="interview_evidence_summaries",
    )
    op.drop_index(
        "ix_interview_evidence_summaries_summary_id",
        table_name="interview_evidence_summaries",
    )
    op.drop_table("interview_evidence_summaries")
