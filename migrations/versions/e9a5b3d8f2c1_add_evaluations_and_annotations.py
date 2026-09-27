"""add candidate evaluations and expert annotations

Revision ID: e9a5b3d8f2c1
Revises: d8f4a2c7e1b9
Create Date: 2026-09-07 01:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e9a5b3d8f2c1"
down_revision: Union[str, None] = "d8f4a2c7e1b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "candidate_evaluations",
        sa.Column("evaluation_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=30), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("overall_explanation", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["interview_sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("evaluation_id"),
    )
    op.create_index("ix_candidate_evaluations_evaluation_id", "candidate_evaluations", ["evaluation_id"])
    op.create_index("ix_candidate_evaluations_model_name", "candidate_evaluations", ["model_name"])
    op.create_index("ix_candidate_evaluations_session_id", "candidate_evaluations", ["session_id"])

    op.create_table(
        "expert_annotations",
        sa.Column("annotation_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("expert_id", sa.String(length=100), nullable=False),
        sa.Column("rubric_version", sa.String(length=30), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["interview_sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("annotation_id"),
        sa.UniqueConstraint(
            "session_id",
            "expert_id",
            "rubric_version",
            name="uq_expert_annotation_session_expert_rubric",
        ),
    )
    op.create_index("ix_expert_annotations_annotation_id", "expert_annotations", ["annotation_id"])
    op.create_index("ix_expert_annotations_expert_id", "expert_annotations", ["expert_id"])
    op.create_index("ix_expert_annotations_session_id", "expert_annotations", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_expert_annotations_session_id", table_name="expert_annotations")
    op.drop_index("ix_expert_annotations_expert_id", table_name="expert_annotations")
    op.drop_index("ix_expert_annotations_annotation_id", table_name="expert_annotations")
    op.drop_table("expert_annotations")
    op.drop_index("ix_candidate_evaluations_session_id", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_model_name", table_name="candidate_evaluations")
    op.drop_index("ix_candidate_evaluations_evaluation_id", table_name="candidate_evaluations")
    op.drop_table("candidate_evaluations")
