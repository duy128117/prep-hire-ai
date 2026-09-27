from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class ExpertAnnotation(Base):
    __tablename__ = "expert_annotations"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "expert_id",
            "rubric_version",
            name="uq_expert_annotation_session_expert_rubric",
        ),
    )

    annotation_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        String(36),
        ForeignKey("interview_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    expert_id = Column(String(100), nullable=False, index=True)
    rubric_version = Column(String(30), nullable=False, default="candidate-evaluation-v1")
    scores_json = Column(JSON, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    session = relationship("InterviewSession", back_populates="expert_annotations")

