from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.question_benchmark import (
    AIJudgeBatchResponse,
    AIJudgeRequest,
    AnswerHumanAuditRequest,
    AnswerHumanAuditResponse,
    AnswerQualityScores,
    AnswerSampleSummaryResponse,
    BenchmarkCaseFromSessionRequest,
    BenchmarkCaseResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkSummaryResponse,
    BlindBenchmarkResponse,
    BlindAnswerSampleResponse,
    BlindQuestionOutput,
    QuestionQualityAnnotationRequest,
    QuestionQualityAnnotationResponse,
    QuestionQualityScores,
    QuestionHumanAuditRequest,
    QuestionHumanAuditResponse,
    SyntheticAnswerGenerateRequest,
    SyntheticAnswerResponse,
)
from app.services.question_benchmark_service import (
    QUESTION_RUBRIC_VERSION,
    QuestionBenchmarkService,
)
from app.services.weak_supervision_service import (
    ANSWER_AI_RUBRIC_VERSION,
    MINIMUM_AI_JUDGES,
    WeakSupervisionService,
)


router = APIRouter()


def _case_response(benchmark_case) -> BenchmarkCaseResponse:
    return BenchmarkCaseResponse(
        case_id=benchmark_case.case_id,
        source_session_id=benchmark_case.source_session_id,
        title=benchmark_case.title,
        language=benchmark_case.language,
        turn_index=benchmark_case.turn_index,
        max_questions=benchmark_case.max_questions,
        stage=benchmark_case.stage,
        criterion=benchmark_case.criterion,
        cv_text=benchmark_case.cv_text,
        jd_text=benchmark_case.jd_text,
        cv_profile=benchmark_case.cv_profile_json,
        job_profile=benchmark_case.job_profile_json,
        interview_history=benchmark_case.interview_history_json,
        created_at=benchmark_case.created_at,
    )


def _run_response(run) -> BenchmarkRunResponse:
    success_count = sum(output.status == "success" for output in run.outputs)
    return BenchmarkRunResponse(
        run_id=run.run_id,
        case_id=run.case_id,
        status=run.status,
        requested_model_count=len(run.outputs),
        successful_output_count=success_count,
        failed_output_count=len(run.outputs) - success_count,
        created_at=run.created_at,
    )


def _synthetic_answer_response(sample) -> SyntheticAnswerResponse:
    return SyntheticAnswerResponse(
        sample_id=sample.sample_id,
        question_output_id=sample.question_output_id,
        blind_label=sample.blind_label,
        generator_model_name=sample.generator_model_name,
        target_quality_level=sample.target_quality_level,
        status=sample.status,
        answer=sample.answer_text,
        generation_latency_ms=sample.generation_latency_ms,
        error_message=sample.error_message,
        created_at=sample.created_at,
    )


@router.post(
    "/cases/from-session",
    response_model=BenchmarkCaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_benchmark_case(
    payload: BenchmarkCaseFromSessionRequest,
    db: Session = Depends(get_db),
) -> BenchmarkCaseResponse:
    try:
        benchmark_case = QuestionBenchmarkService.create_case_from_session(
            db,
            payload.session_id,
            payload.next_turn_index,
            payload.title,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create benchmark case: {exc}") from exc
    return _case_response(benchmark_case)


@router.get("/cases/{case_id}", response_model=BenchmarkCaseResponse)
def get_benchmark_case(case_id: int, db: Session = Depends(get_db)) -> BenchmarkCaseResponse:
    benchmark_case = QuestionBenchmarkService.get_case(db, case_id)
    if not benchmark_case:
        raise HTTPException(status_code=404, detail="Benchmark case not found")
    return _case_response(benchmark_case)


@router.post(
    "/runs",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def run_benchmark(
    payload: BenchmarkRunRequest,
    db: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    try:
        run = QuestionBenchmarkService.run_models(db, payload.case_id, payload.model_names)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to run benchmark: {exc}") from exc
    return _run_response(run)


@router.get("/runs/{run_id}/blind", response_model=BlindBenchmarkResponse)
def get_blind_benchmark(run_id: str, db: Session = Depends(get_db)) -> BlindBenchmarkResponse:
    run = QuestionBenchmarkService.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    benchmark_case = run.case
    return BlindBenchmarkResponse(
        run_id=run.run_id,
        case_id=run.case_id,
        title=benchmark_case.title,
        language=benchmark_case.language,
        stage=benchmark_case.stage,
        target_criterion=benchmark_case.criterion,
        cv_text=benchmark_case.cv_text,
        jd_text=benchmark_case.jd_text,
        cv_profile=benchmark_case.cv_profile_json,
        job_profile=benchmark_case.job_profile_json,
        interview_history=benchmark_case.interview_history_json,
        rubric_version=QUESTION_RUBRIC_VERSION,
        outputs=[
            BlindQuestionOutput(
                output_id=output.output_id,
                blind_label=output.blind_label,
                question=output.question_text,
            )
            for output in run.outputs
            if output.status == "success" and output.question_text
        ],
    )


@router.post(
    "/outputs/{output_id}/annotations",
    response_model=QuestionQualityAnnotationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_question_annotation(
    output_id: str,
    payload: QuestionQualityAnnotationRequest,
    db: Session = Depends(get_db),
) -> QuestionQualityAnnotationResponse:
    try:
        annotation = QuestionBenchmarkService.create_annotation(
            db,
            output_id,
            payload.expert_id,
            payload.scores.model_dump(),
            payload.notes,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save annotation: {exc}") from exc
    return QuestionQualityAnnotationResponse(
        annotation_id=annotation.annotation_id,
        output_id=annotation.output_id,
        blind_label=annotation.output.blind_label,
        expert_id=annotation.expert_id,
        rubric_version=annotation.rubric_version,
        scores=QuestionQualityScores(**annotation.scores_json),
        notes=annotation.notes,
        created_at=annotation.created_at,
    )


@router.post(
    "/outputs/{output_id}/human-audits",
    response_model=QuestionHumanAuditResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_question_human_audit(
    output_id: str,
    payload: QuestionHumanAuditRequest,
    db: Session = Depends(get_db),
) -> QuestionHumanAuditResponse:
    compatibility_payload = QuestionQualityAnnotationRequest(
        expert_id=payload.reviewer_id,
        scores=payload.scores,
        notes=payload.notes,
    )
    annotation = create_question_annotation(output_id, compatibility_payload, db)
    return QuestionHumanAuditResponse(
        audit_id=annotation.annotation_id,
        output_id=annotation.output_id,
        blind_label=annotation.blind_label,
        reviewer_id=annotation.expert_id,
        rubric_version=annotation.rubric_version,
        scores=annotation.scores,
        notes=annotation.notes,
        created_at=annotation.created_at,
    )


@router.post(
    "/runs/{run_id}/ai-judge",
    response_model=AIJudgeBatchResponse,
)
def judge_questions_with_ai(
    run_id: str,
    payload: AIJudgeRequest,
    db: Session = Depends(get_db),
) -> AIJudgeBatchResponse:
    try:
        result = WeakSupervisionService.judge_question_run(db, run_id, payload.judge_model_names)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AIJudgeBatchResponse(**result)


@router.post(
    "/outputs/{output_id}/synthetic-answers",
    response_model=list[SyntheticAnswerResponse],
    status_code=status.HTTP_201_CREATED,
)
def generate_synthetic_answers(
    output_id: str,
    payload: SyntheticAnswerGenerateRequest,
    db: Session = Depends(get_db),
) -> list[SyntheticAnswerResponse]:
    try:
        samples = WeakSupervisionService.generate_answers(
            db,
            output_id,
            payload.generator_model_name,
            payload.target_quality_levels,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [_synthetic_answer_response(sample) for sample in samples]


@router.get(
    "/answer-samples/{sample_id}/blind",
    response_model=BlindAnswerSampleResponse,
)
def get_blind_answer_sample(
    sample_id: str,
    db: Session = Depends(get_db),
) -> BlindAnswerSampleResponse:
    sample = WeakSupervisionService.get_answer_sample(db, sample_id)
    if not sample or sample.status != "success" or not sample.answer_text:
        raise HTTPException(status_code=404, detail="Successful answer sample not found")
    output = sample.question_output
    benchmark_case = output.run.case
    return BlindAnswerSampleResponse(
        sample_id=sample.sample_id,
        blind_label=sample.blind_label,
        language=benchmark_case.language,
        stage=benchmark_case.stage,
        target_criterion=benchmark_case.criterion,
        cv_profile=benchmark_case.cv_profile_json,
        job_profile=benchmark_case.job_profile_json,
        interview_history=benchmark_case.interview_history_json,
        question=output.question_text,
        answer=sample.answer_text,
        rubric_version=ANSWER_AI_RUBRIC_VERSION,
    )


@router.post(
    "/answer-samples/{sample_id}/ai-judge",
    response_model=AIJudgeBatchResponse,
)
def judge_answer_with_ai(
    sample_id: str,
    payload: AIJudgeRequest,
    db: Session = Depends(get_db),
) -> AIJudgeBatchResponse:
    try:
        result = WeakSupervisionService.judge_answer_sample(db, sample_id, payload.judge_model_names)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AIJudgeBatchResponse(**result)


@router.post(
    "/answer-samples/{sample_id}/human-audits",
    response_model=AnswerHumanAuditResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_answer_human_audit(
    sample_id: str,
    payload: AnswerHumanAuditRequest,
    db: Session = Depends(get_db),
) -> AnswerHumanAuditResponse:
    try:
        audit = WeakSupervisionService.create_answer_human_audit(
            db,
            sample_id,
            payload.reviewer_id,
            payload.scores.model_dump(),
            payload.notes,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnswerHumanAuditResponse(
        audit_id=audit.audit_id,
        sample_id=audit.sample_id,
        reviewer_id=audit.reviewer_id,
        rubric_version=audit.rubric_version,
        scores=AnswerQualityScores(**audit.scores_json),
        notes=audit.notes,
        created_at=audit.created_at,
    )


@router.get(
    "/answer-samples/{sample_id}/summary",
    response_model=AnswerSampleSummaryResponse,
)
def get_answer_sample_summary(
    sample_id: str,
    db: Session = Depends(get_db),
) -> AnswerSampleSummaryResponse:
    sample = WeakSupervisionService.get_answer_sample(db, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Answer sample not found")
    consensus = WeakSupervisionService.answer_consensus(sample)
    return AnswerSampleSummaryResponse(
        sample_id=sample.sample_id,
        blind_label=sample.blind_label,
        human_audit_count=len(sample.human_audits),
        **consensus,
    )


@router.get("/runs/{run_id}/summary", response_model=BenchmarkSummaryResponse)
def get_benchmark_summary(run_id: str, db: Session = Depends(get_db)) -> BenchmarkSummaryResponse:
    run = QuestionBenchmarkService.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    summary = QuestionBenchmarkService.summarize_run(run)
    return BenchmarkSummaryResponse(
        run_id=run.run_id,
        case_id=run.case_id,
        minimum_ai_judges_per_output=MINIMUM_AI_JUDGES,
        **summary,
    )
