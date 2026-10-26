"""add realtime interview model outputs

Revision ID: e6f1a4c8d2b7
Revises: d5e8f1a3c7b9
Create Date: 2026-10-01 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e6f1a4c8d2b7"
down_revision: Union[str, None] = "d5e8f1a3c7b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column(
            "context_source",
            sa.String(length=30),
            nullable=False,
            server_default="user_supplied",
        ),
    )
    op.add_column("interview_turns", sa.Column("provider_name", sa.String(length=50), nullable=True))
    op.add_column("interview_turns", sa.Column("question_candidates_json", sa.JSON(), nullable=True))
    op.add_column("interview_turns", sa.Column("selected_question_score", sa.Float(), nullable=True))
    op.add_column("interview_turns", sa.Column("question_ranker_version", sa.String(length=80), nullable=True))
    op.add_column(
        "interview_turns",
        sa.Column("question_selection_latency_ms", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("interview_turns", sa.Column("answer_scores_json", sa.JSON(), nullable=True))
    op.add_column("interview_turns", sa.Column("answer_scorer_version", sa.String(length=80), nullable=True))
    op.add_column(
        "interview_turns",
        sa.Column("answer_scoring_latency_ms", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("interview_turns", "answer_scoring_latency_ms")
    op.drop_column("interview_turns", "answer_scorer_version")
    op.drop_column("interview_turns", "answer_scores_json")
    op.drop_column("interview_turns", "question_selection_latency_ms")
    op.drop_column("interview_turns", "question_ranker_version")
    op.drop_column("interview_turns", "selected_question_score")
    op.drop_column("interview_turns", "question_candidates_json")
    op.drop_column("interview_turns", "provider_name")
    op.drop_column("interview_sessions", "context_source")
