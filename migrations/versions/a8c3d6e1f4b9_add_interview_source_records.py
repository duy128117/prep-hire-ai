"""add interview source records

Revision ID: a8c3d6e1f4b9
Revises: f7a2c5d9e3b8
Create Date: 2026-10-05 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a8c3d6e1f4b9"
down_revision: Union[str, None] = "f7a2c5d9e3b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column("source_cv_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "interview_sessions",
        sa.Column("source_job_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_interview_sessions_source_cv_id_cvs",
        "interview_sessions",
        "cvs",
        ["source_cv_id"],
        ["cv_id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_interview_sessions_source_job_id_jobs",
        "interview_sessions",
        "jobs",
        ["source_job_id"],
        ["job_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_interview_sessions_source_cv_id",
        "interview_sessions",
        ["source_cv_id"],
    )
    op.create_index(
        "ix_interview_sessions_source_job_id",
        "interview_sessions",
        ["source_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_interview_sessions_source_job_id",
        table_name="interview_sessions",
    )
    op.drop_index(
        "ix_interview_sessions_source_cv_id",
        table_name="interview_sessions",
    )
    op.drop_constraint(
        "fk_interview_sessions_source_job_id_jobs",
        "interview_sessions",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_interview_sessions_source_cv_id_cvs",
        "interview_sessions",
        type_="foreignkey",
    )
    op.drop_column("interview_sessions", "source_job_id")
    op.drop_column("interview_sessions", "source_cv_id")
