"""add question quality benchmark tables

Revision ID: f0b6c4e9a3d2
Revises: e9a5b3d8f2c1
Create Date: 2026-09-14 23:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f0b6c4e9a3d2"
down_revision: Union[str, None] = "e9a5b3d8f2c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "question_benchmark_cases",
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("source_session_id", sa.String(length=36), nullable=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("language", sa.String(length=5), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("max_questions", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("criterion", sa.String(length=60), nullable=False),
        sa.Column("cv_text", sa.Text(), nullable=False),
        sa.Column("jd_text", sa.Text(), nullable=False),
        sa.Column("cv_profile_json", sa.JSON(), nullable=False),
        sa.Column("job_profile_json", sa.JSON(), nullable=False),
        sa.Column("interview_history_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_session_id"],
            ["interview_sessions.session_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("case_id"),
    )
    op.create_index("ix_question_benchmark_cases_case_id", "question_benchmark_cases", ["case_id"])
    op.create_index(
        "ix_question_benchmark_cases_source_session_id",
        "question_benchmark_cases",
        ["source_session_id"],
    )

    op.create_table(
        "question_benchmark_runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("prompt_version", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["question_benchmark_cases.case_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_question_benchmark_runs_case_id", "question_benchmark_runs", ["case_id"])
    op.create_index("ix_question_benchmark_runs_status", "question_benchmark_runs", ["status"])

    op.create_table(
        "question_benchmark_outputs",
        sa.Column("output_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("blind_label", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=True),
        sa.Column("criterion", sa.String(length=60), nullable=True),
        sa.Column("expected_signal", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["question_benchmark_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("output_id"),
        sa.UniqueConstraint("run_id", "blind_label", name="uq_question_benchmark_run_blind_label"),
        sa.UniqueConstraint("run_id", "model_name", name="uq_question_benchmark_run_model"),
    )
    op.create_index(
        "ix_question_benchmark_outputs_blind_label",
        "question_benchmark_outputs",
        ["blind_label"],
    )
    op.create_index("ix_question_benchmark_outputs_model_name", "question_benchmark_outputs", ["model_name"])
    op.create_index("ix_question_benchmark_outputs_run_id", "question_benchmark_outputs", ["run_id"])

    op.create_table(
        "question_quality_annotations",
        sa.Column("annotation_id", sa.Integer(), nullable=False),
        sa.Column("output_id", sa.String(length=36), nullable=False),
        sa.Column("expert_id", sa.String(length=100), nullable=False),
        sa.Column("rubric_version", sa.String(length=30), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["output_id"],
            ["question_benchmark_outputs.output_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("annotation_id"),
        sa.UniqueConstraint(
            "output_id",
            "expert_id",
            "rubric_version",
            name="uq_question_quality_output_expert_rubric",
        ),
    )
    op.create_index(
        "ix_question_quality_annotations_annotation_id",
        "question_quality_annotations",
        ["annotation_id"],
    )
    op.create_index(
        "ix_question_quality_annotations_expert_id",
        "question_quality_annotations",
        ["expert_id"],
    )
    op.create_index(
        "ix_question_quality_annotations_output_id",
        "question_quality_annotations",
        ["output_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_question_quality_annotations_output_id", table_name="question_quality_annotations")
    op.drop_index("ix_question_quality_annotations_expert_id", table_name="question_quality_annotations")
    op.drop_index("ix_question_quality_annotations_annotation_id", table_name="question_quality_annotations")
    op.drop_table("question_quality_annotations")
    op.drop_index("ix_question_benchmark_outputs_run_id", table_name="question_benchmark_outputs")
    op.drop_index("ix_question_benchmark_outputs_model_name", table_name="question_benchmark_outputs")
    op.drop_index("ix_question_benchmark_outputs_blind_label", table_name="question_benchmark_outputs")
    op.drop_table("question_benchmark_outputs")
    op.drop_index("ix_question_benchmark_runs_status", table_name="question_benchmark_runs")
    op.drop_index("ix_question_benchmark_runs_case_id", table_name="question_benchmark_runs")
    op.drop_table("question_benchmark_runs")
    op.drop_index("ix_question_benchmark_cases_source_session_id", table_name="question_benchmark_cases")
    op.drop_index("ix_question_benchmark_cases_case_id", table_name="question_benchmark_cases")
    op.drop_table("question_benchmark_cases")
