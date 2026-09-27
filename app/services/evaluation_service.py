from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.models.candidate_evaluation import CandidateEvaluation
from app.models.expert_annotation import ExpertAnnotation
from app.models.interview_session import InterviewSession
from app.research.rubrics import CANDIDATE_EVALUATION_RUBRIC
from app.services.ai_service import AIService
from app.services.interview_service import AdaptiveQuestionEngine, InterviewSessionService


EVALUATION_PROMPT_VERSION = "candidate-evaluation-v1"
RUBRIC_VERSION = "candidate-evaluation-v1"
RUBRIC_CRITERIA = tuple(CANDIDATE_EVALUATION_RUBRIC["criteria"])
MINIMUM_CONSENSUS_EXPERTS = 3


@dataclass(frozen=True)
class GeneratedEvaluation:
    scores: dict[str, float]
    evidence: dict[str, list[str]]
    explanation: str
    model_name: str
    prompt_hash: str
    latency_ms: int


class CandidateEvaluationEngine:
    @staticmethod
    def build_prompt(session: InterviewSession) -> str:
        transcript = [
            {
                "turn_index": turn.turn_index,
                "stage": turn.stage,
                "question": AdaptiveQuestionEngine.redact_personal_data(turn.question_text),
                "answer": AdaptiveQuestionEngine.redact_personal_data(turn.answer_text or ""),
            }
            for turn in session.turns
        ]
        context = {
            "cv_profile": session.cv_profile_json,
            "job_profile": session.job_profile_json,
            "cv_excerpt": AdaptiveQuestionEngine.redact_personal_data(session.cv_text)[:5000],
            "jd_excerpt": AdaptiveQuestionEngine.redact_personal_data(session.jd_text)[:5000],
            "transcript": transcript,
        }
        rubric = CANDIDATE_EVALUATION_RUBRIC["criteria"]
        output_language = "Vietnamese" if session.language == "vi" else "English"
        return (
            "You are evaluating a completed structured job interview. Score only job-related evidence "
            "present in the CV, JD, and transcript. Do not infer competence from identity, writing style, "
            "accent, protected characteristics, or missing personal information. Use the full 1-5 scale. "
            "A CV claim without supporting interview evidence must not receive a high evidence score. "
            "Return valid JSON only, without markdown.\n"
            f"Write all evidence strings and overall_explanation in {output_language}.\n"
            "Required JSON schema:\n"
            '{"scores": {criterion: number}, "evidence": {criterion: [string]}, '
            '"overall_explanation": string}\n'
            f"Every scores key is required and must be 1..5: {', '.join(RUBRIC_CRITERIA)}.\n"
            f"Rubric definitions: {json.dumps(rubric, ensure_ascii=False)}\n"
            f"Evaluation context: {json.dumps(context, ensure_ascii=False)}"
        )

    @staticmethod
    def parse_response(data: dict[str, Any] | None) -> tuple[dict[str, float], dict[str, list[str]], str]:
        if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
            raise ValueError("AI evaluation did not return the required scores object")

        scores: dict[str, float] = {}
        for criterion in RUBRIC_CRITERIA:
            try:
                score = float(data["scores"][criterion])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"AI evaluation is missing a valid score for {criterion}") from exc
            if not 1 <= score <= 5:
                raise ValueError(f"AI score for {criterion} must be between 1 and 5")
            scores[criterion] = round(score, 3)

        raw_evidence = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
        evidence = {
            criterion: [str(item).strip()[:500] for item in raw_evidence.get(criterion, [])[:5] if str(item).strip()]
            if isinstance(raw_evidence.get(criterion), list)
            else []
            for criterion in RUBRIC_CRITERIA
        }
        explanation = str(data.get("overall_explanation") or "").strip()[:3000]
        return scores, evidence, explanation

    @classmethod
    def evaluate(cls, session: InterviewSession, model_name: str | None = None) -> GeneratedEvaluation:
        if session.status != "completed":
            raise ValueError("Candidate evaluation requires a completed interview session")
        if not session.allow_external_ai:
            raise ValueError("External AI evaluation was not enabled for this interview session")

        selected_model = (model_name or AIService.MODEL_NAME).strip()
        prompt = cls.build_prompt(session)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        started_at = time.perf_counter()
        try:
            client = AIService._build_client()
            response = client.models.generate_content(model=selected_model, contents=prompt)
            payload = AIService._extract_json((response.text or "").strip() if response else "")
            scores, evidence, explanation = cls.parse_response(payload)
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"AI candidate evaluation failed: {exc}") from exc
        latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        return GeneratedEvaluation(
            scores=scores,
            evidence=evidence,
            explanation=explanation,
            model_name=selected_model,
            prompt_hash=prompt_hash,
            latency_ms=latency_ms,
        )


class CandidateEvaluationService:
    @staticmethod
    def create(db: Session, session_id: str, model_name: str | None = None) -> CandidateEvaluation:
        session = InterviewSessionService.get(db, session_id)
        if not session:
            raise ValueError("Interview session not found")
        generated = CandidateEvaluationEngine.evaluate(session, model_name)
        evaluation = CandidateEvaluation(
            session_id=session.session_id,
            provider="google",
            model_name=generated.model_name,
            prompt_version=EVALUATION_PROMPT_VERSION,
            prompt_hash=generated.prompt_hash,
            scores_json=generated.scores,
            evidence_json=generated.evidence,
            overall_explanation=generated.explanation,
            latency_ms=generated.latency_ms,
        )
        db.add(evaluation)
        db.commit()
        db.refresh(evaluation)
        return evaluation

    @staticmethod
    def list_for_session(db: Session, session_id: str) -> list[CandidateEvaluation]:
        return (
            db.query(CandidateEvaluation)
            .filter(CandidateEvaluation.session_id == session_id)
            .order_by(CandidateEvaluation.created_at.desc())
            .all()
        )


class ExpertConsensusService:
    @staticmethod
    def create_annotation(
        db: Session,
        session_id: str,
        expert_id: str,
        scores: dict[str, float],
        notes: str | None,
    ) -> ExpertAnnotation:
        session = InterviewSessionService.get(db, session_id)
        if not session:
            raise ValueError("Interview session not found")
        if session.status != "completed":
            raise ValueError("Expert annotation requires a completed interview session")

        annotation = (
            db.query(ExpertAnnotation)
            .filter(
                ExpertAnnotation.session_id == session_id,
                ExpertAnnotation.expert_id == expert_id.strip(),
                ExpertAnnotation.rubric_version == RUBRIC_VERSION,
            )
            .first()
        )
        if annotation:
            raise ValueError("Expert annotation already exists and is locked for blind evaluation")
        annotation = ExpertAnnotation(
            session_id=session_id,
            expert_id=expert_id.strip(),
            rubric_version=RUBRIC_VERSION,
            scores_json=scores,
            notes=notes,
        )
        db.add(annotation)
        db.commit()
        db.refresh(annotation)
        return annotation

    @staticmethod
    def list_annotations(db: Session, session_id: str) -> list[ExpertAnnotation]:
        return (
            db.query(ExpertAnnotation)
            .filter(
                ExpertAnnotation.session_id == session_id,
                ExpertAnnotation.rubric_version == RUBRIC_VERSION,
            )
            .order_by(ExpertAnnotation.expert_id)
            .all()
        )

    @staticmethod
    def _icc_2_k(matrix: np.ndarray) -> float | None:
        subject_count, rater_count = matrix.shape
        if subject_count < 2 or rater_count < 2:
            return None
        grand_mean = matrix.mean()
        subject_means = matrix.mean(axis=1)
        rater_means = matrix.mean(axis=0)
        ms_subject = rater_count * np.square(subject_means - grand_mean).sum() / (subject_count - 1)
        ms_rater = subject_count * np.square(rater_means - grand_mean).sum() / (rater_count - 1)
        residual = matrix - subject_means[:, None] - rater_means[None, :] + grand_mean
        ms_error = np.square(residual).sum() / ((subject_count - 1) * (rater_count - 1))
        denominator = ms_subject + (ms_rater - ms_error) / subject_count
        if denominator == 0:
            return None
        return round(float((ms_subject - ms_error) / denominator), 6)

    @classmethod
    def consensus_from_annotations(cls, annotations: list[ExpertAnnotation]) -> dict[str, Any]:
        expert_count = len(annotations)
        stats: dict[str, dict[str, float]] = {}
        reference_scores: dict[str, float] | None = None
        if annotations:
            for criterion in RUBRIC_CRITERIA:
                values = [float(item.scores_json[criterion]) for item in annotations]
                stats[criterion] = {
                    "median": round(float(statistics.median(values)), 3),
                    "mean": round(float(statistics.mean(values)), 3),
                    "standard_deviation": round(float(statistics.pstdev(values)), 3),
                    "minimum": round(min(values), 3),
                    "maximum": round(max(values), 3),
                }
            if expert_count >= MINIMUM_CONSENSUS_EXPERTS:
                reference_scores = {criterion: stats[criterion]["median"] for criterion in RUBRIC_CRITERIA}

        matrix = np.asarray(
            [[float(annotation.scores_json[criterion]) for annotation in annotations] for criterion in RUBRIC_CRITERIA],
            dtype=float,
        )
        reliability = cls._icc_2_k(matrix) if annotations else None
        is_ready = reference_scores is not None
        return {
            "status": "ready" if is_ready else "insufficient_experts",
            "expert_count": expert_count,
            "minimum_experts": MINIMUM_CONSENSUS_EXPERTS,
            "rubric_version": RUBRIC_VERSION,
            "reference_scores": reference_scores,
            "criterion_stats": stats if is_ready else {},
            "inter_rater_icc_2_k": reliability if is_ready else None,
        }
