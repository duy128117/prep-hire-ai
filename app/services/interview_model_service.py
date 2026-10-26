from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import BASE_DIR, settings

if TYPE_CHECKING:
    from app.services.question_ranking_v2_service import PairwiseQuestionRanker
    from app.services.weak_supervision_training_service import CandidateAnswerScorer


class InterviewModelService:
    """Lazy, process-local registry for trained realtime interview models."""

    _lock = threading.RLock()
    _ranker: PairwiseQuestionRanker | None = None
    _ranker_path: Path | None = None
    _ranker_error: str | None = None
    _answer_scorer: CandidateAnswerScorer | None = None
    _answer_scorer_path: Path | None = None
    _answer_scorer_error: str | None = None

    @staticmethod
    def _resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else BASE_DIR / path

    @classmethod
    def reset_cache(cls) -> None:
        with cls._lock:
            cls._ranker = None
            cls._ranker_path = None
            cls._ranker_error = None
            cls._answer_scorer = None
            cls._answer_scorer_path = None
            cls._answer_scorer_error = None

    @classmethod
    def _get_ranker(cls) -> PairwiseQuestionRanker:
        from app.services.question_ranking_v2_service import PairwiseQuestionRanker

        path = cls._resolve_path(settings.INTERVIEW_QUESTION_RANKER_ARTIFACT)
        with cls._lock:
            if cls._ranker is not None and cls._ranker_path == path:
                return cls._ranker
            try:
                cls._ranker = PairwiseQuestionRanker(path)
                cls._ranker_path = path
                cls._ranker_error = None
                return cls._ranker
            except Exception as exc:
                cls._ranker = None
                cls._ranker_path = path
                cls._ranker_error = str(exc)[:500]
                raise

    @classmethod
    def _get_answer_scorer(cls) -> CandidateAnswerScorer:
        from app.services.weak_supervision_training_service import CandidateAnswerScorer

        path = cls._resolve_path(settings.INTERVIEW_ANSWER_SCORER_ARTIFACT)
        with cls._lock:
            if cls._answer_scorer is not None and cls._answer_scorer_path == path:
                return cls._answer_scorer
            try:
                cls._answer_scorer = CandidateAnswerScorer(path)
                cls._answer_scorer_path = path
                cls._answer_scorer_error = None
                return cls._answer_scorer
            except Exception as exc:
                cls._answer_scorer = None
                cls._answer_scorer_path = path
                cls._answer_scorer_error = str(exc)[:500]
                raise

    @classmethod
    def rank_questions(
        cls,
        context_text: str,
        questions: list[str],
        stage: str,
        target_criterion: str,
    ) -> tuple[list[float], str]:
        ranker = cls._get_ranker()
        rows = [
            {
                "context_text": context_text,
                "stage": stage,
                "target_criterion": target_criterion,
                "question_text": question,
            }
            for question in questions
        ]
        scores = ranker.score_rows(rows)
        model_type = str(ranker.artifact.get("model_type") or "unknown")
        return [round(float(score), 6) for score in scores], model_type

    @classmethod
    def score_answer(
        cls,
        context_text: str,
        question: str,
        answer: str,
        stage: str,
        target_criterion: str,
    ) -> tuple[dict[str, float], str]:
        scorer = cls._get_answer_scorer()
        scores = scorer.score(
            context_text,
            question,
            answer,
            stage,
            target_criterion,
        )
        model_type = str(scorer.artifact.get("model_type") or "unknown")
        return scores, model_type

    @classmethod
    def status(cls, load_models: bool = True) -> dict[str, Any]:
        if load_models:
            for loader in (cls._get_ranker, cls._get_answer_scorer):
                try:
                    loader()
                except Exception:
                    pass
        ranker_path = cls._resolve_path(settings.INTERVIEW_QUESTION_RANKER_ARTIFACT)
        scorer_path = cls._resolve_path(settings.INTERVIEW_ANSWER_SCORER_ARTIFACT)
        configured_providers = []
        from app.services.llm_provider_service import LLMProviderService

        for spec in LLMProviderService.pool_specs():
            if LLMProviderService.is_configured(spec):
                configured_providers.append(spec.provider)
        return {
            "multi_provider_enabled": settings.INTERVIEW_MULTI_PROVIDER_ENABLED,
            "question_candidate_count": max(
                1, min(4, settings.INTERVIEW_QUESTION_CANDIDATE_COUNT)
            ),
            "configured_providers": configured_providers,
            "question_ranker": {
                "ready": cls._ranker is not None,
                "model_type": (
                    cls._ranker.artifact.get("model_type") if cls._ranker else None
                ),
                "artifact_path": str(ranker_path),
                "error": cls._ranker_error,
            },
            "answer_scorer": {
                "ready": cls._answer_scorer is not None,
                "model_type": (
                    cls._answer_scorer.artifact.get("model_type")
                    if cls._answer_scorer
                    else None
                ),
                "artifact_path": str(scorer_path),
                "error": cls._answer_scorer_error,
            },
        }
