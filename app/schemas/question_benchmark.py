from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class QuestionQualityScores(BaseModel):
    context_relevance: float = Field(..., ge=1, le=5)
    technical_depth: float = Field(..., ge=1, le=5)
    clarity: float = Field(..., ge=1, le=5)
    evidence_elicitation: float = Field(..., ge=1, le=5)
    adaptivity: float = Field(..., ge=1, le=5)
    non_redundancy: float = Field(..., ge=1, le=5)
    fairness: float = Field(..., ge=1, le=5)


class BenchmarkCaseFromSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=36, max_length=36)
    next_turn_index: int = Field(..., ge=0)
    title: str | None = Field(default=None, min_length=3, max_length=200)


class BenchmarkCaseResponse(BaseModel):
    case_id: int
    source_session_id: str | None
    title: str
    language: str
    turn_index: int
    max_questions: int
    stage: str
    criterion: str
    cv_text: str
    jd_text: str
    cv_profile: dict
    job_profile: dict
    interview_history: list[dict]
    created_at: datetime


class BenchmarkRunRequest(BaseModel):
    case_id: int = Field(..., ge=1)
    model_names: list[str] = Field(..., min_length=2, max_length=8)

    @field_validator("model_names")
    @classmethod
    def validate_models(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 100 for value in normalized):
            raise ValueError("Each model name must contain 1-100 characters")
        if any(not value[0].isalnum() or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-" for ch in value) for value in normalized):
            raise ValueError("Model names may only contain letters, numbers, '.', '_', '/', and '-'")
        if len({value.lower() for value in normalized}) != len(normalized):
            raise ValueError("model_names must be unique")
        return normalized


class BenchmarkRunResponse(BaseModel):
    run_id: str
    case_id: int
    status: str
    requested_model_count: int
    successful_output_count: int
    failed_output_count: int
    created_at: datetime


class BlindQuestionOutput(BaseModel):
    output_id: str
    blind_label: str
    question: str


class BlindBenchmarkResponse(BaseModel):
    run_id: str
    case_id: int
    title: str
    language: str
    stage: str
    target_criterion: str
    cv_text: str
    jd_text: str
    cv_profile: dict
    job_profile: dict
    interview_history: list[dict]
    rubric_version: str
    outputs: list[BlindQuestionOutput]


class QuestionQualityAnnotationRequest(BaseModel):
    expert_id: str = Field(..., min_length=2, max_length=100)
    scores: QuestionQualityScores
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("expert_id")
    @classmethod
    def normalize_expert_id(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("expert_id must contain at least 2 non-space characters")
        return normalized


class QuestionQualityAnnotationResponse(BaseModel):
    annotation_id: int
    output_id: str
    blind_label: str
    expert_id: str
    rubric_version: str
    scores: QuestionQualityScores
    notes: str | None
    created_at: datetime


class QuestionHumanAuditRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=2, max_length=100)
    scores: QuestionQualityScores
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("reviewer_id")
    @classmethod
    def normalize_reviewer_id(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("reviewer_id must contain at least 2 non-space characters")
        return normalized


class QuestionHumanAuditResponse(BaseModel):
    audit_id: int
    output_id: str
    blind_label: str
    reviewer_id: str
    rubric_version: str
    scores: QuestionQualityScores
    notes: str | None
    created_at: datetime


class AIJudgeRequest(BaseModel):
    judge_model_names: list[str] = Field(..., min_length=3, max_length=8)

    @field_validator("judge_model_names")
    @classmethod
    def validate_judge_models(cls, values: list[str]) -> list[str]:
        return BenchmarkRunRequest.validate_models(values)


class AIJudgeFailure(BaseModel):
    blind_label: str | None = None
    judge_model: str
    error: str


class AIJudgeBatchResponse(BaseModel):
    run_id: str | None = None
    sample_id: str | None = None
    created: int
    skipped: int
    failures: list[AIJudgeFailure] = Field(default_factory=list)


class AnswerQualityScores(BaseModel):
    answer_relevance: float = Field(..., ge=1, le=5)
    technical_correctness: float = Field(..., ge=1, le=5)
    reasoning_depth: float = Field(..., ge=1, le=5)
    evidence_specificity: float = Field(..., ge=1, le=5)
    clarity: float = Field(..., ge=1, le=5)
    consistency: float = Field(..., ge=1, le=5)
    criterion_score: float = Field(..., ge=1, le=5)


class SyntheticAnswerGenerateRequest(BaseModel):
    generator_model_name: str = Field(..., min_length=2, max_length=100)
    target_quality_levels: list[int] = Field(default=[1, 2, 3, 4, 5], min_length=1, max_length=5)

    @field_validator("generator_model_name")
    @classmethod
    def validate_generator_model(cls, value: str) -> str:
        return BenchmarkRunRequest.validate_models([value])[0]

    @field_validator("target_quality_levels")
    @classmethod
    def validate_quality_levels(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 5 for value in values):
            raise ValueError("target_quality_levels must be between 1 and 5")
        if len(set(values)) != len(values):
            raise ValueError("target_quality_levels must be unique")
        return values


class SyntheticAnswerResponse(BaseModel):
    sample_id: str
    question_output_id: str
    blind_label: str
    generator_model_name: str
    target_quality_level: int
    status: str
    answer: str | None
    generation_latency_ms: int
    error_message: str | None
    created_at: datetime


class BlindAnswerSampleResponse(BaseModel):
    sample_id: str
    blind_label: str
    language: str
    stage: str
    target_criterion: str
    cv_profile: dict
    job_profile: dict
    interview_history: list[dict]
    question: str
    answer: str
    rubric_version: str


class AnswerHumanAuditRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=2, max_length=100)
    scores: AnswerQualityScores
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("reviewer_id")
    @classmethod
    def normalize_answer_reviewer_id(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("reviewer_id must contain at least 2 non-space characters")
        return normalized


class AnswerHumanAuditResponse(BaseModel):
    audit_id: int
    sample_id: str
    reviewer_id: str
    rubric_version: str
    scores: AnswerQualityScores
    notes: str | None
    created_at: datetime


class PseudoLabelConsensusResponse(BaseModel):
    status: str
    ai_judge_count: int
    pseudo_label_scores: dict[str, float] | None
    criterion_stats: dict[str, "QuestionCriterionStats"] = Field(default_factory=dict)
    mean_quality_score: float | None
    inter_judge_icc_2_k: float | None
    needs_human_review: bool
    decision: str


class AnswerSampleSummaryResponse(PseudoLabelConsensusResponse):
    sample_id: str
    blind_label: str
    human_audit_count: int


class QuestionCriterionStats(BaseModel):
    median: float
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


class BenchmarkModelSummary(BaseModel):
    blind_label: str
    provider: str
    model_name: str
    generation_status: str
    latency_ms: int
    ai_judge_count: int
    human_audit_count: int
    mean_quality_score: float | None = None
    pseudo_label_scores: QuestionQualityScores | None = None
    criterion_stats: dict[str, QuestionCriterionStats] = Field(default_factory=dict)
    inter_judge_icc_2_k: float | None = None
    needs_human_review: bool = True
    decision: str = "review"
    error_message: str | None = None


class BenchmarkSummaryResponse(BaseModel):
    run_id: str
    case_id: int
    status: str
    minimum_ai_judges_per_output: int
    pending_blind_labels: list[str] = Field(default_factory=list)
    model_results: list[BenchmarkModelSummary] = Field(default_factory=list)
