from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class QuestionAIJudgment(Base):
    __tablename__ = "question_ai_judgments"
    __table_args__ = (
        UniqueConstraint(
            "output_id",
            "judge_model_name",
            "rubric_version",
            name="uq_question_ai_judgment_output_model_rubric",
        ),
    )

    judgment_id = Column(Integer, primary_key=True, index=True)
    output_id = Column(
        String(36),
        ForeignKey("question_benchmark_outputs.output_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider = Column(String(50), nullable=False, default="google")
    judge_model_name = Column(String(100), nullable=False, index=True)
    rubric_version = Column(String(30), nullable=False)
    scores_json = Column(JSON, nullable=False)
    evidence_json = Column(JSON, nullable=False)
    recommendation = Column(String(20), nullable=False)
    prompt_version = Column(String(30), nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    latency_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    output = relationship("QuestionBenchmarkOutput", back_populates="ai_judgments")


class SyntheticAnswerSample(Base):
    __tablename__ = "synthetic_answer_samples"

    sample_id = Column(String(36), primary_key=True)
    question_output_id = Column(
        String(36),
        ForeignKey("question_benchmark_outputs.output_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    blind_label = Column(String(20), nullable=False, unique=True, index=True)
    provider = Column(String(50), nullable=False, default="google")
    generator_model_name = Column(String(100), nullable=False, index=True)
    target_quality_level = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)
    answer_text = Column(Text, nullable=True)
    prompt_version = Column(String(30), nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    generation_latency_ms = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    question_output = relationship("QuestionBenchmarkOutput", back_populates="synthetic_answers")
    ai_judgments = relationship(
        "AnswerAIJudgment",
        back_populates="sample",
        cascade="all, delete-orphan",
    )
    human_audits = relationship(
        "AnswerHumanAudit",
        back_populates="sample",
        cascade="all, delete-orphan",
    )


class AnswerAIJudgment(Base):
    __tablename__ = "answer_ai_judgments"
    __table_args__ = (
        UniqueConstraint(
            "sample_id",
            "judge_model_name",
            "rubric_version",
            name="uq_answer_ai_judgment_sample_model_rubric",
        ),
    )

    judgment_id = Column(Integer, primary_key=True, index=True)
    sample_id = Column(
        String(36),
        ForeignKey("synthetic_answer_samples.sample_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider = Column(String(50), nullable=False, default="google")
    judge_model_name = Column(String(100), nullable=False, index=True)
    rubric_version = Column(String(30), nullable=False)
    scores_json = Column(JSON, nullable=False)
    evidence_json = Column(JSON, nullable=False)
    recommendation = Column(String(20), nullable=False)
    prompt_version = Column(String(30), nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    latency_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    sample = relationship("SyntheticAnswerSample", back_populates="ai_judgments")


class AnswerHumanAudit(Base):
    __tablename__ = "answer_human_audits"
    __table_args__ = (
        UniqueConstraint(
            "sample_id",
            "reviewer_id",
            "rubric_version",
            name="uq_answer_human_audit_sample_reviewer_rubric",
        ),
    )

    audit_id = Column(Integer, primary_key=True, index=True)
    sample_id = Column(
        String(36),
        ForeignKey("synthetic_answer_samples.sample_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reviewer_id = Column(String(100), nullable=False, index=True)
    rubric_version = Column(String(30), nullable=False)
    scores_json = Column(JSON, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    sample = relationship("SyntheticAnswerSample", back_populates="human_audits")
