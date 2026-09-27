from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class CandidateEvaluation(Base):
    __tablename__ = "candidate_evaluations"

    evaluation_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        String(36),
        ForeignKey("interview_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False, index=True)
    prompt_version = Column(String(30), nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    scores_json = Column(JSON, nullable=False)
    evidence_json = Column(JSON, nullable=False)
    overall_explanation = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    session = relationship("InterviewSession", back_populates="evaluations")

