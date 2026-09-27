"""add interview sessions and turns

Revision ID: d8f4a2c7e1b9
Revises: f6a2b8c9d1e4
Create Date: 2026-09-07 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8f4a2c7e1b9"
down_revision: Union[str, None] = "f6a2b8c9d1e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "interview_sessions",
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("language", sa.String(length=5), nullable=False, server_default="vi"),
        sa.Column("max_questions", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("current_turn_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_stage", sa.String(length=40), nullable=False, server_default="introduction"),
        sa.Column("allow_external_ai", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("research_consent", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("cv_text", sa.Text(), nullable=False),
        sa.Column("jd_text", sa.Text(), nullable=False),
        sa.Column("cv_profile_json", sa.JSON(), nullable=False),
        sa.Column("job_profile_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_interview_sessions_status", "interview_sessions", ["status"])
    op.create_index("ix_interview_sessions_user_id", "interview_sessions", ["user_id"])

    op.create_table(
        "interview_turns",
        sa.Column("turn_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("criterion", sa.String(length=60), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("expected_signal", sa.Text(), nullable=True),
        sa.Column("question_source", sa.String(length=20), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("prompt_version", sa.String(length=30), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("generation_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("answer_word_count", sa.Integer(), nullable=True),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["interview_sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("turn_id"),
        sa.UniqueConstraint("session_id", "turn_index", name="uq_interview_turn_session_index"),
    )
    op.create_index("ix_interview_turns_session_id", "interview_turns", ["session_id"])
    op.create_index("ix_interview_turns_turn_id", "interview_turns", ["turn_id"])


def downgrade() -> None:
    op.drop_index("ix_interview_turns_turn_id", table_name="interview_turns")
    op.drop_index("ix_interview_turns_session_id", table_name="interview_turns")
    op.drop_table("interview_turns")
    op.drop_index("ix_interview_sessions_user_id", table_name="interview_sessions")
    op.drop_index("ix_interview_sessions_status", table_name="interview_sessions")
    op.drop_table("interview_sessions")
