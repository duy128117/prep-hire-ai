from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.schemas.cv import ExtractedCvProfile, JobContext


class InterviewContextRequest(BaseModel):
    cv_text: str = Field(..., min_length=30)
    jd_text: str = Field(..., min_length=30)
    allow_external_ai: bool = False


class ContextExtractionMetadata(BaseModel):
    cv_character_count: int = Field(..., ge=0)
    jd_character_count: int = Field(..., ge=0)
    extraction_version: str = "v1"


class InterviewContextResponse(BaseModel):
    cv_profile: ExtractedCvProfile
    job_profile: JobContext
    metadata: ContextExtractionMetadata


class InterviewSessionCreateRequest(InterviewContextRequest):
    user_id: int | None = Field(default=None, ge=1)
    language: str = "vi"
    max_questions: int = Field(default=8, ge=5, le=15)
    research_consent: bool = False

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"vi", "en"}:
            raise ValueError("language must be 'vi' or 'en'")
        return normalized


class InterviewAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=2, max_length=10000)


class InterviewQuestionResponse(BaseModel):
    turn_index: int
    stage: str
    question: str


class InterviewTurnResponse(InterviewQuestionResponse):
    answer: str | None = None
    question_source: str
    model_name: str | None = None
    generation_latency_ms: int
    created_at: datetime
    answered_at: datetime | None = None


class InterviewSessionResponse(BaseModel):
    session_id: str
    status: str
    language: str
    max_questions: int
    current_stage: str
    allow_external_ai: bool
    research_consent: bool
    current_question: InterviewQuestionResponse | None = None
    turns: list[InterviewTurnResponse] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None


class CandidateRubricScores(BaseModel):
    technical_knowledge: float = Field(..., ge=1, le=5)
    problem_solving: float = Field(..., ge=1, le=5)
    experience_evidence: float = Field(..., ge=1, le=5)
    communication: float = Field(..., ge=1, le=5)
    role_relevance: float = Field(..., ge=1, le=5)
    learning_adaptability: float = Field(..., ge=1, le=5)
    overall_readiness: float = Field(..., ge=1, le=5)


class CandidateEvaluationRequest(BaseModel):
    model_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]+$",
    )


class CandidateEvaluationResponse(BaseModel):
    evaluation_id: int
    session_id: str
    provider: str
    model_name: str
    prompt_version: str
    scores: CandidateRubricScores
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    overall_explanation: str | None = None
    latency_ms: int
    created_at: datetime


class ExpertAnnotationRequest(BaseModel):
    expert_id: str = Field(..., min_length=2, max_length=100)
    scores: CandidateRubricScores
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("expert_id")
    @classmethod
    def normalize_expert_id(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("expert_id must contain at least 2 non-space characters")
        return normalized


class ExpertAnnotationResponse(BaseModel):
    annotation_id: int
    session_id: str
    expert_id: str
    rubric_version: str
    scores: CandidateRubricScores
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class ConsensusCriterionStats(BaseModel):
    median: float
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


class ExpertConsensusResponse(BaseModel):
    session_id: str
    status: str
    expert_count: int
    minimum_experts: int
    rubric_version: str
    reference_scores: CandidateRubricScores | None = None
    criterion_stats: dict[str, ConsensusCriterionStats] = Field(default_factory=dict)
    inter_rater_icc_2_k: float | None = None
