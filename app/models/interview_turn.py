from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class InterviewTurn(Base):
    __tablename__ = "interview_turns"
    __table_args__ = (UniqueConstraint("session_id", "turn_index", name="uq_interview_turn_session_index"),)

    turn_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        String(36),
        ForeignKey("interview_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_index = Column(Integer, nullable=False)
    stage = Column(String(40), nullable=False)
    criterion = Column(String(60), nullable=False)
    question_text = Column(Text, nullable=False)
    expected_signal = Column(Text, nullable=True)
    question_source = Column(String(20), nullable=False)
    model_name = Column(String(100), nullable=True)
    prompt_version = Column(String(30), nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    generation_latency_ms = Column(Integer, nullable=False, default=0)

    answer_text = Column(Text, nullable=True)
    answer_word_count = Column(Integer, nullable=True)
    answered_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    session = relationship("InterviewSession", back_populates="turns")

