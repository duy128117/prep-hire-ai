from __future__ import annotations

import statistics
import uuid
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.models.interview_session import InterviewSession
from app.models.question_benchmark import (
    QuestionBenchmarkCase,
    QuestionBenchmarkOutput,
    QuestionBenchmarkRun,
    QuestionQualityAnnotation,
)
from app.research.rubrics import QUESTION_QUALITY_RUBRIC
from app.services.evaluation_service import ExpertConsensusService
from app.services.interview_service import AdaptiveQuestionEngine, InterviewSessionService, PROMPT_VERSION
from app.services.llm_provider_service import LLMProviderService


QUESTION_RUBRIC_VERSION = "question-quality-v1"
MINIMUM_QUESTION_EXPERTS = 3
QUESTION_CRITERIA = tuple(QUESTION_QUALITY_RUBRIC["criteria"])


class QuestionBenchmarkService:
    @staticmethod
    def _redact_value(value: Any) -> Any:
        if isinstance(value, str):
            return AdaptiveQuestionEngine.redact_personal_data(value)
        if isinstance(value, list):
            return [QuestionBenchmarkService._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: QuestionBenchmarkService._redact_value(item) for key, item in value.items()}
        return value

    @classmethod
    def create_case_from_session(
        cls,
        db: Session,
        session_id: str,
        next_turn_index: int,
        title: str | None = None,
    ) -> QuestionBenchmarkCase:
        session = InterviewSessionService.get(db, session_id)
        if not session:
            raise ValueError("Interview session not found")
        if not session.research_consent:
            raise ValueError("Research consent is required before creating a benchmark case")
        if next_turn_index >= session.max_questions:
            raise ValueError("next_turn_index must be lower than max_questions")

        prior_turns = sorted(
            (turn for turn in session.turns if turn.turn_index < next_turn_index),
            key=lambda turn: turn.turn_index,
        )
        expected_indexes = list(range(next_turn_index))
        if [turn.turn_index for turn in prior_turns] != expected_indexes:
            raise ValueError("The session does not contain every turn before next_turn_index")
        if any(turn.answer_text is None for turn in prior_turns):
            raise ValueError("Every prior benchmark turn must have an answer")

        history = [
            {
                "stage": turn.stage,
                "criterion": turn.criterion,
                "question": turn.question_text,
                "answer": turn.answer_text,
            }
            for turn in prior_turns
        ]
        redacted_history = cls._redact_value(history)
        stage = AdaptiveQuestionEngine.determine_stage(
            next_turn_index,
            session.max_questions,
            redacted_history,
        )
        criterion = AdaptiveQuestionEngine.criterion_for_stage(stage, redacted_history)
        benchmark_case = QuestionBenchmarkCase(
            source_session_id=session.session_id,
            title=(title or f"Session {session.session_id[:8]} turn {next_turn_index}").strip(),
            language=session.language,
            turn_index=next_turn_index,
            max_questions=session.max_questions,
            stage=stage,
            criterion=criterion,
            cv_text=AdaptiveQuestionEngine.redact_personal_data(session.cv_text),
            jd_text=AdaptiveQuestionEngine.redact_personal_data(session.jd_text),
            cv_profile_json=cls._redact_value(session.cv_profile_json),
            job_profile_json=cls._redact_value(session.job_profile_json),
            interview_history_json=redacted_history,
        )
        db.add(benchmark_case)
        db.commit()
        db.refresh(benchmark_case)
        return benchmark_case

    @staticmethod
    def get_case(db: Session, case_id: int) -> QuestionBenchmarkCase | None:
        return db.query(QuestionBenchmarkCase).filter(QuestionBenchmarkCase.case_id == case_id).first()

    @classmethod
    def run_models(
        cls,
        db: Session,
        case_id: int,
        model_names: list[str],
    ) -> QuestionBenchmarkRun:
        benchmark_case = cls.get_case(db, case_id)
        if not benchmark_case:
            raise ValueError("Benchmark case not found")

        run = QuestionBenchmarkRun(
            run_id=str(uuid.uuid4()),
            case_id=benchmark_case.case_id,
            status="running",
            prompt_version=PROMPT_VERSION,
        )
        db.add(run)
        db.flush()

        success_count = 0
        for model_name in model_names:
            try:
                spec = LLMProviderService.resolve(model_name)
                provider = spec.provider
                selected_model = spec.model
            except ValueError:
                provider = "unknown"
                selected_model = model_name
            generated = AdaptiveQuestionEngine.generate(
                turn_index=benchmark_case.turn_index,
                max_questions=benchmark_case.max_questions,
                language=benchmark_case.language,
                cv_text=benchmark_case.cv_text,
                jd_text=benchmark_case.jd_text,
                cv_profile=benchmark_case.cv_profile_json,
                job_profile=benchmark_case.job_profile_json,
                history=benchmark_case.interview_history_json,
                allow_external_ai=True,
                model_name=selected_model,
                provider=None if provider == "unknown" else provider,
            )
            is_success = generated.source == "ai"
            success_count += int(is_success)
            output = QuestionBenchmarkOutput(
                output_id=str(uuid.uuid4()),
                run_id=run.run_id,
                provider=generated.provider or provider,
                model_name=selected_model,
                blind_label=f"Q-{uuid.uuid4().hex[:8].upper()}",
                status="success" if is_success else "failed",
                question_text=generated.question if is_success else None,
                criterion=generated.criterion if is_success else None,
                expected_signal=generated.expected_signal if is_success else None,
                latency_ms=generated.latency_ms,
                prompt_hash=generated.prompt_hash,
                error_message=None if is_success else "Model generation failed or returned an invalid response",
            )
            run.outputs.append(output)

        run.status = "completed" if success_count == len(model_names) else "partial_failed"
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def get_run(db: Session, run_id: str) -> QuestionBenchmarkRun | None:
        return db.query(QuestionBenchmarkRun).filter(QuestionBenchmarkRun.run_id == run_id).first()

    @staticmethod
    def create_annotation(
        db: Session,
        output_id: str,
        expert_id: str,
        scores: dict[str, float],
        notes: str | None,
    ) -> QuestionQualityAnnotation:
        output = (
            db.query(QuestionBenchmarkOutput)
            .filter(QuestionBenchmarkOutput.output_id == output_id)
            .first()
        )
        if not output:
            raise ValueError("Benchmark output not found")
        if output.status != "success" or not output.question_text:
            raise ValueError("Only a successfully generated question can be annotated")

        normalized_expert_id = expert_id.strip()
        existing = (
            db.query(QuestionQualityAnnotation)
            .filter(
                QuestionQualityAnnotation.output_id == output_id,
                QuestionQualityAnnotation.expert_id == normalized_expert_id,
                QuestionQualityAnnotation.rubric_version == QUESTION_RUBRIC_VERSION,
            )
            .first()
        )
        if existing:
            raise ValueError("Question annotation already exists and is locked for blind evaluation")

        annotation = QuestionQualityAnnotation(
            output_id=output_id,
            expert_id=normalized_expert_id,
            rubric_version=QUESTION_RUBRIC_VERSION,
            scores_json=scores,
            notes=notes,
        )
        db.add(annotation)
        db.commit()
        db.refresh(annotation)
        return annotation

    @staticmethod
    def consensus_for_output(output: QuestionBenchmarkOutput) -> dict[str, Any]:
        annotations = [
            item for item in output.annotations if item.rubric_version == QUESTION_RUBRIC_VERSION
        ]
        expert_count = len(annotations)
        if expert_count < MINIMUM_QUESTION_EXPERTS:
            return {
                "expert_count": expert_count,
                "reference_scores": None,
                "criterion_stats": {},
                "mean_quality_score": None,
                "inter_rater_icc_2_k": None,
            }

        stats: dict[str, dict[str, float]] = {}
        for criterion in QUESTION_CRITERIA:
            values = [float(item.scores_json[criterion]) for item in annotations]
            stats[criterion] = {
                "median": round(float(statistics.median(values)), 3),
                "mean": round(float(statistics.mean(values)), 3),
                "standard_deviation": round(float(statistics.pstdev(values)), 3),
                "minimum": round(min(values), 3),
                "maximum": round(max(values), 3),
            }
        reference_scores = {criterion: stats[criterion]["median"] for criterion in QUESTION_CRITERIA}
        matrix = np.asarray(
            [[float(item.scores_json[criterion]) for item in annotations] for criterion in QUESTION_CRITERIA],
            dtype=float,
        )
        return {
            "expert_count": expert_count,
            "reference_scores": reference_scores,
            "criterion_stats": stats,
            "mean_quality_score": round(float(statistics.mean(reference_scores.values())), 3),
            "inter_rater_icc_2_k": ExpertConsensusService._icc_2_k(matrix),
        }

    @classmethod
    def summarize_run(cls, run: QuestionBenchmarkRun) -> dict[str, Any]:
        from app.services.weak_supervision_service import MINIMUM_AI_JUDGES, WeakSupervisionService

        successful = [output for output in run.outputs if output.status == "success"]
        consensus = {
            output.output_id: WeakSupervisionService.question_consensus(output)
            for output in successful
        }
        pending = [
            output.blind_label
            for output in successful
            if consensus[output.output_id]["ai_judge_count"] < MINIMUM_AI_JUDGES
        ]
        if not successful or pending:
            return {
                "status": "insufficient_ai_judgments",
                "pending_blind_labels": pending,
                "model_results": [],
            }

        results = []
        for output in run.outputs:
            values = consensus.get(output.output_id, {})
            results.append(
                {
                    "blind_label": output.blind_label,
                    "provider": output.provider,
                    "model_name": output.model_name,
                    "generation_status": output.status,
                    "latency_ms": output.latency_ms,
                    "ai_judge_count": values.get("ai_judge_count", 0),
                    "human_audit_count": len(output.annotations),
                    "mean_quality_score": values.get("mean_quality_score"),
                    "pseudo_label_scores": values.get("pseudo_label_scores"),
                    "criterion_stats": values.get("criterion_stats", {}),
                    "inter_judge_icc_2_k": values.get("inter_judge_icc_2_k"),
                    "needs_human_review": values.get("needs_human_review", True),
                    "decision": values.get("decision", "review"),
                    "error_message": output.error_message,
                }
            )
        results.sort(key=lambda item: item["mean_quality_score"] or -1, reverse=True)
        return {"status": "ready", "pending_blind_labels": [], "model_results": results}
