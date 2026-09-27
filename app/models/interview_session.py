from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    session_id = Column(String(36), primary_key=True)
    user_id = Column(Integer, nullable=True, index=True)
    status = Column(String(20), nullable=False, default="active", index=True)
    language = Column(String(5), nullable=False, default="vi")
    max_questions = Column(Integer, nullable=False, default=8)
    current_turn_index = Column(Integer, nullable=False, default=0)
    current_stage = Column(String(40), nullable=False, default="introduction")
    allow_external_ai = Column(Boolean, nullable=False, default=False)
    research_consent = Column(Boolean, nullable=False, default=False)

    cv_text = Column(Text, nullable=False)
    jd_text = Column(Text, nullable=False)
    cv_profile_json = Column(JSON, nullable=False)
    job_profile_json = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    turns = relationship(
        "InterviewTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="InterviewTurn.turn_index",
    )
    evaluations = relationship(
        "CandidateEvaluation",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    expert_annotations = relationship(
        "ExpertAnnotation",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    benchmark_cases = relationship(
        "QuestionBenchmarkCase",
        passive_deletes=True,
    )
