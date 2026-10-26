from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class QuestionBenchmarkCase(Base):
    __tablename__ = "question_benchmark_cases"

    case_id = Column(Integer, primary_key=True, index=True)
    source_session_id = Column(
        String(36),
        ForeignKey("interview_sessions.session_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title = Column(String(200), nullable=False)
    language = Column(String(5), nullable=False)
    turn_index = Column(Integer, nullable=False)
    max_questions = Column(Integer, nullable=False)
    stage = Column(String(40), nullable=False)
    criterion = Column(String(60), nullable=False)
    cv_text = Column(Text, nullable=False)
    jd_text = Column(Text, nullable=False)
    cv_profile_json = Column(JSON, nullable=False)
    job_profile_json = Column(JSON, nullable=False)
    interview_history_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    runs = relationship(
        "QuestionBenchmarkRun",
        back_populates="case",
        cascade="all, delete-orphan",
    )


class QuestionBenchmarkRun(Base):
    __tablename__ = "question_benchmark_runs"

    run_id = Column(String(36), primary_key=True)
    case_id = Column(
        Integer,
        ForeignKey("question_benchmark_cases.case_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String(30), nullable=False, default="running", index=True)
    prompt_version = Column(String(30), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("QuestionBenchmarkCase", back_populates="runs")
    outputs = relationship(
        "QuestionBenchmarkOutput",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="QuestionBenchmarkOutput.blind_label",
    )


class QuestionBenchmarkOutput(Base):
    __tablename__ = "question_benchmark_outputs"
    __table_args__ = (
        UniqueConstraint("run_id", "model_name", name="uq_question_benchmark_run_model"),
        UniqueConstraint("run_id", "blind_label", name="uq_question_benchmark_run_blind_label"),
    )

    output_id = Column(String(36), primary_key=True)
    run_id = Column(
        String(36),
        ForeignKey("question_benchmark_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider = Column(String(50), nullable=False, default="google")
    logical_slot = Column(String(30), nullable=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    blind_label = Column(String(20), nullable=False, index=True)
    status = Column(String(20), nullable=False)
    question_text = Column(Text, nullable=True)
    criterion = Column(String(60), nullable=True)
    expected_signal = Column(Text, nullable=True)
    generation_metadata_json = Column(JSON, nullable=True)
    latency_ms = Column(Integer, nullable=False, default=0)
    prompt_hash = Column(String(64), nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    run = relationship("QuestionBenchmarkRun", back_populates="outputs")
    annotations = relationship(
        "QuestionQualityAnnotation",
        back_populates="output",
        cascade="all, delete-orphan",
    )
    ai_judgments = relationship(
        "QuestionAIJudgment",
        back_populates="output",
        cascade="all, delete-orphan",
    )
    synthetic_answers = relationship(
        "SyntheticAnswerSample",
        back_populates="question_output",
        cascade="all, delete-orphan",
    )


class QuestionQualityAnnotation(Base):
    __tablename__ = "question_quality_annotations"
    __table_args__ = (
        UniqueConstraint(
            "output_id",
            "expert_id",
            "rubric_version",
            name="uq_question_quality_output_expert_rubric",
        ),
    )

    annotation_id = Column(Integer, primary_key=True, index=True)
    output_id = Column(
        String(36),
        ForeignKey("question_benchmark_outputs.output_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    expert_id = Column(String(100), nullable=False, index=True)
    rubric_version = Column(String(30), nullable=False, default="question-quality-v1")
    scores_json = Column(JSON, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    output = relationship("QuestionBenchmarkOutput", back_populates="annotations")
