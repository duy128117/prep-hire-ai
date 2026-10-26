from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint

from app.db.base_class import Base


class InterviewEvidenceSummary(Base):
    __tablename__ = "interview_evidence_summaries"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "aggregator_version",
            name="uq_interview_evidence_session_version",
        ),
    )

    summary_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        String(36),
        ForeignKey("interview_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    aggregator_version = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    competency_scores_json = Column(JSON, nullable=False)
    evidence_json = Column(JSON, nullable=False)
    flags_json = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
