"""add multi-llm weak supervision tables

Revision ID: a1c7d5e9f3b4
Revises: f0b6c4e9a3d2
Create Date: 2026-09-16 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1c7d5e9f3b4"
down_revision: Union[str, None] = "f0b6c4e9a3d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "question_ai_judgments",
        sa.Column("judgment_id", sa.Integer(), nullable=False),
        sa.Column("output_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("judge_model_name", sa.String(length=100), nullable=False),
        sa.Column("rubric_version", sa.String(length=30), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("recommendation", sa.String(length=20), nullable=False),
        sa.Column("prompt_version", sa.String(length=30), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["output_id"],
            ["question_benchmark_outputs.output_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("judgment_id"),
        sa.UniqueConstraint(
            "output_id",
            "judge_model_name",
            "rubric_version",
            name="uq_question_ai_judgment_output_model_rubric",
        ),
    )
    op.create_index("ix_question_ai_judgments_judgment_id", "question_ai_judgments", ["judgment_id"])
    op.create_index("ix_question_ai_judgments_output_id", "question_ai_judgments", ["output_id"])
    op.create_index(
        "ix_question_ai_judgments_judge_model_name",
        "question_ai_judgments",
        ["judge_model_name"],
    )

    op.create_table(
        "synthetic_answer_samples",
        sa.Column("sample_id", sa.String(length=36), nullable=False),
        sa.Column("question_output_id", sa.String(length=36), nullable=False),
        sa.Column("blind_label", sa.String(length=20), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("generator_model_name", sa.String(length=100), nullable=False),
        sa.Column("target_quality_level", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("prompt_version", sa.String(length=30), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("generation_latency_ms", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["question_output_id"],
            ["question_benchmark_outputs.output_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("sample_id"),
        sa.UniqueConstraint("blind_label"),
    )
    op.create_index(
        "ix_synthetic_answer_samples_blind_label",
        "synthetic_answer_samples",
        ["blind_label"],
    )
    op.create_index(
        "ix_synthetic_answer_samples_generator_model_name",
        "synthetic_answer_samples",
        ["generator_model_name"],
    )
    op.create_index(
        "ix_synthetic_answer_samples_question_output_id",
        "synthetic_answer_samples",
        ["question_output_id"],
    )

    op.create_table(
        "answer_ai_judgments",
        sa.Column("judgment_id", sa.Integer(), nullable=False),
        sa.Column("sample_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("judge_model_name", sa.String(length=100), nullable=False),
        sa.Column("rubric_version", sa.String(length=30), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("recommendation", sa.String(length=20), nullable=False),
        sa.Column("prompt_version", sa.String(length=30), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["sample_id"],
            ["synthetic_answer_samples.sample_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("judgment_id"),
        sa.UniqueConstraint(
            "sample_id",
            "judge_model_name",
            "rubric_version",
            name="uq_answer_ai_judgment_sample_model_rubric",
        ),
    )
    op.create_index("ix_answer_ai_judgments_judgment_id", "answer_ai_judgments", ["judgment_id"])
    op.create_index("ix_answer_ai_judgments_judge_model_name", "answer_ai_judgments", ["judge_model_name"])
    op.create_index("ix_answer_ai_judgments_sample_id", "answer_ai_judgments", ["sample_id"])

    op.create_table(
        "answer_human_audits",
        sa.Column("audit_id", sa.Integer(), nullable=False),
        sa.Column("sample_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=100), nullable=False),
        sa.Column("rubric_version", sa.String(length=30), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["sample_id"],
            ["synthetic_answer_samples.sample_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("audit_id"),
        sa.UniqueConstraint(
            "sample_id",
            "reviewer_id",
            "rubric_version",
            name="uq_answer_human_audit_sample_reviewer_rubric",
        ),
    )
    op.create_index("ix_answer_human_audits_audit_id", "answer_human_audits", ["audit_id"])
    op.create_index("ix_answer_human_audits_reviewer_id", "answer_human_audits", ["reviewer_id"])
    op.create_index("ix_answer_human_audits_sample_id", "answer_human_audits", ["sample_id"])


def downgrade() -> None:
    op.drop_index("ix_answer_human_audits_sample_id", table_name="answer_human_audits")
    op.drop_index("ix_answer_human_audits_reviewer_id", table_name="answer_human_audits")
    op.drop_index("ix_answer_human_audits_audit_id", table_name="answer_human_audits")
    op.drop_table("answer_human_audits")
    op.drop_index("ix_answer_ai_judgments_sample_id", table_name="answer_ai_judgments")
    op.drop_index("ix_answer_ai_judgments_judge_model_name", table_name="answer_ai_judgments")
    op.drop_index("ix_answer_ai_judgments_judgment_id", table_name="answer_ai_judgments")
    op.drop_table("answer_ai_judgments")
    op.drop_index("ix_synthetic_answer_samples_question_output_id", table_name="synthetic_answer_samples")
    op.drop_index("ix_synthetic_answer_samples_generator_model_name", table_name="synthetic_answer_samples")
    op.drop_index("ix_synthetic_answer_samples_blind_label", table_name="synthetic_answer_samples")
    op.drop_table("synthetic_answer_samples")
    op.drop_index("ix_question_ai_judgments_judge_model_name", table_name="question_ai_judgments")
    op.drop_index("ix_question_ai_judgments_output_id", table_name="question_ai_judgments")
    op.drop_index("ix_question_ai_judgments_judgment_id", table_name="question_ai_judgments")
    op.drop_table("question_ai_judgments")
