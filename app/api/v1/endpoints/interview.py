from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.interview import (
    ContextExtractionMetadata,
    CandidateEvaluationRequest,
    CandidateEvaluationResponse,
    CandidateRubricScores,
    ExpertAnnotationRequest,
    ExpertAnnotationResponse,
    ExpertConsensusResponse,
    InterviewAnswerRequest,
    InterviewContextRequest,
    InterviewContextResponse,
    InterviewQuestionResponse,
    InterviewSessionCreateRequest,
    InterviewSessionResponse,
    InterviewTurnResponse,
)
from app.research.rubrics import CANDIDATE_EVALUATION_RUBRIC, QUESTION_QUALITY_RUBRIC
from app.services.pdf_processor import CvIngestService
from app.services.interview_service import InterviewSessionService
from app.services.interview_service import AdaptiveQuestionEngine
from app.services.evaluation_service import CandidateEvaluationService, ExpertConsensusService


router = APIRouter()


def _session_response(session) -> InterviewSessionResponse:
    turns = [
        InterviewTurnResponse(
            turn_index=turn.turn_index,
            stage=turn.stage,
            question=turn.question_text,
            answer=turn.answer_text,
            question_source=turn.question_source,
            model_name=turn.model_name,
            generation_latency_ms=turn.generation_latency_ms,
            created_at=turn.created_at,
            answered_at=turn.answered_at,
        )
        for turn in session.turns
    ]
    active_turn = next(
        (turn for turn in turns if turn.turn_index == session.current_turn_index and turn.answer is None),
        None,
    )
    current_question = None
    if active_turn:
        current_question = InterviewQuestionResponse(
            turn_index=active_turn.turn_index,
            stage=active_turn.stage,
            question=active_turn.question,
        )
    return InterviewSessionResponse(
        session_id=session.session_id,
        status=session.status,
        language=session.language,
        max_questions=session.max_questions,
        current_stage=session.current_stage,
        allow_external_ai=session.allow_external_ai,
        research_consent=session.research_consent,
        current_question=current_question,
        turns=turns,
        created_at=session.created_at,
        completed_at=session.completed_at,
    )


def _evaluation_response(evaluation) -> CandidateEvaluationResponse:
    return CandidateEvaluationResponse(
        evaluation_id=evaluation.evaluation_id,
        session_id=evaluation.session_id,
        provider=evaluation.provider,
        model_name=evaluation.model_name,
        prompt_version=evaluation.prompt_version,
        scores=CandidateRubricScores(**evaluation.scores_json),
        evidence=evaluation.evidence_json,
        overall_explanation=evaluation.overall_explanation,
        latency_ms=evaluation.latency_ms,
        created_at=evaluation.created_at,
    )


def _annotation_response(annotation) -> ExpertAnnotationResponse:
    return ExpertAnnotationResponse(
        annotation_id=annotation.annotation_id,
        session_id=annotation.session_id,
        expert_id=annotation.expert_id,
        rubric_version=annotation.rubric_version,
        scores=CandidateRubricScores(**annotation.scores_json),
        notes=annotation.notes,
        created_at=annotation.created_at,
        updated_at=annotation.updated_at,
    )


@router.get("/rubrics")
def get_research_rubrics() -> dict:
    return {
        "question_quality": QUESTION_QUALITY_RUBRIC,
        "candidate_evaluation": CANDIDATE_EVALUATION_RUBRIC,
    }


@router.post(
    "/sessions",
    response_model=InterviewSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_interview_session(
    payload: InterviewSessionCreateRequest,
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    try:
        session = InterviewSessionService.create(
            db=db,
            cv_text=payload.cv_text,
            jd_text=payload.jd_text,
            language=payload.language,
            max_questions=payload.max_questions,
            allow_external_ai=payload.allow_external_ai,
            research_consent=payload.research_consent,
            user_id=payload.user_id,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create interview session: {exc}") from exc
    return _session_response(session)


@router.get("/sessions/{session_id}", response_model=InterviewSessionResponse)
def get_interview_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    session = InterviewSessionService.get(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return _session_response(session)


@router.post("/sessions/{session_id}/answers", response_model=InterviewSessionResponse)
def submit_interview_answer(
    session_id: str,
    payload: InterviewAnswerRequest,
    db: Session = Depends(get_db),
) -> InterviewSessionResponse:
    try:
        session = InterviewSessionService.submit_answer(db, session_id, payload.answer)
    except ValueError as exc:
        db.rollback()
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save interview answer: {exc}") from exc
    return _session_response(session)


@router.post(
    "/sessions/{session_id}/evaluations",
    response_model=CandidateEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
)
def evaluate_candidate(
    session_id: str,
    payload: CandidateEvaluationRequest,
    db: Session = Depends(get_db),
) -> CandidateEvaluationResponse:
    try:
        evaluation = CandidateEvaluationService.create(db, session_id, payload.model_name)
    except ValueError as exc:
        db.rollback()
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save candidate evaluation: {exc}") from exc
    return _evaluation_response(evaluation)


@router.get(
    "/sessions/{session_id}/evaluations",
    response_model=list[CandidateEvaluationResponse],
)
def list_candidate_evaluations(
    session_id: str,
    db: Session = Depends(get_db),
) -> list[CandidateEvaluationResponse]:
    if not InterviewSessionService.get(db, session_id):
        raise HTTPException(status_code=404, detail="Interview session not found")
    return [
        _evaluation_response(item)
        for item in CandidateEvaluationService.list_for_session(db, session_id)
    ]


@router.post(
    "/sessions/{session_id}/expert-annotations",
    response_model=ExpertAnnotationResponse,
)
def save_expert_annotation(
    session_id: str,
    payload: ExpertAnnotationRequest,
    db: Session = Depends(get_db),
) -> ExpertAnnotationResponse:
    try:
        annotation = ExpertConsensusService.create_annotation(
            db,
            session_id,
            payload.expert_id,
            payload.scores.model_dump(mode="json"),
            payload.notes,
        )
    except ValueError as exc:
        db.rollback()
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save expert annotation: {exc}") from exc
    return _annotation_response(annotation)


@router.get(
    "/sessions/{session_id}/expert-consensus",
    response_model=ExpertConsensusResponse,
)
def get_expert_consensus(
    session_id: str,
    db: Session = Depends(get_db),
) -> ExpertConsensusResponse:
    if not InterviewSessionService.get(db, session_id):
        raise HTTPException(status_code=404, detail="Interview session not found")
    annotations = ExpertConsensusService.list_annotations(db, session_id)
    result = ExpertConsensusService.consensus_from_annotations(annotations)
    return ExpertConsensusResponse(session_id=session_id, **result)


def _extract_context(
    db: Session,
    cv_text: str,
    jd_text: str,
    jd_filename: str | None = None,
    allow_external_ai: bool = False,
) -> InterviewContextResponse:
    try:
        extraction_cv = AdaptiveQuestionEngine.redact_personal_data(cv_text) if allow_external_ai else cv_text
        extraction_jd = AdaptiveQuestionEngine.redact_personal_data(jd_text) if allow_external_ai else jd_text
        cv_profile = CvIngestService.extract_profile(db, extraction_cv, use_ai=allow_external_ai)
        job_profile = CvIngestService.build_uploaded_job_context(
            db,
            extraction_jd,
            jd_filename,
            use_ai=allow_external_ai,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return InterviewContextResponse(
        cv_profile=cv_profile,
        job_profile=job_profile,
        metadata=ContextExtractionMetadata(
            cv_character_count=len(cv_text),
            jd_character_count=len(jd_text),
        ),
    )


@router.post("/extract-context", response_model=InterviewContextResponse)
def extract_interview_context(
    payload: InterviewContextRequest,
    db: Session = Depends(get_db),
) -> InterviewContextResponse:
    """Extract CV and JD independently without computing a match score."""
    return _extract_context(
        db,
        payload.cv_text,
        payload.jd_text,
        allow_external_ai=payload.allow_external_ai,
    )


@router.post("/extract-context-files", response_model=InterviewContextResponse)
async def extract_interview_context_files(
    cv_file: UploadFile = File(..., description="CV: PDF, DOCX, TXT, or MD"),
    jd_file: UploadFile = File(..., description="JD: PDF, DOCX, TXT, MD, PNG, JPG, or WebP"),
    allow_external_ai: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> InterviewContextResponse:
    """Read uploaded CV/JD files and return structured interview context only."""
    try:
        cv_bytes = await cv_file.read()
        jd_bytes = await jd_file.read()
        cv_text = CvIngestService.extract_text_from_file(
            cv_file.filename or "cv",
            cv_file.content_type,
            cv_bytes,
        )
        jd_text = CvIngestService.extract_text_from_job_file(
            jd_file.filename or "jd",
            jd_file.content_type,
            jd_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded files: {exc}") from exc

    return _extract_context(db, cv_text, jd_text, jd_file.filename, allow_external_ai)
