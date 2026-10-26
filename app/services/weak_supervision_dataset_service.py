from __future__ import annotations

import json
import statistics
import time
from typing import Any, Callable

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, load_only, selectinload

from app.models.question_benchmark import (
    QuestionBenchmarkCase,
    QuestionBenchmarkOutput,
    QuestionBenchmarkRun,
    QuestionQualityAnnotation,
)
from app.models.weak_supervision import (
    AnswerAIJudgment,
    AnswerHumanAudit,
    QuestionAIJudgment,
    SyntheticAnswerSample,
)
from app.services.weak_supervision_service import (
    ANSWER_CRITERIA,
    QUESTION_CRITERIA,
    TURN_EVIDENCE_CRITERIA,
    WeakSupervisionService,
)


QUESTION_DATASET_COLUMNS = (
    "group_id",
    "case_id",
    "output_id",
    "stage",
    "target_criterion",
    "context_text",
    "question_text",
    "question_type",
    "difficulty",
    "required_signals_json",
    "bonus_signals_json",
    "applicable_answer_criteria_json",
    "contradiction_rules_json",
    "generator_model_name",
    "ai_judge_count",
    "mean_quality_score",
    "decision",
    "needs_human_review",
    "human_audit_count",
    "human_mean_quality_score",
) + tuple(f"pseudo_{criterion}" for criterion in QUESTION_CRITERIA)

ANSWER_DATASET_COLUMNS = (
    "group_id",
    "case_id",
    "sample_id",
    "question_output_id",
    "stage",
    "target_criterion",
    "context_text",
    "question_text",
    "question_type",
    "difficulty",
    "required_signals_json",
    "bonus_signals_json",
    "applicable_answer_criteria_json",
    "contradiction_rules_json",
    "answer_text",
    "generator_model_name",
    "target_quality_level_metadata",
    "ai_judge_count",
    "mean_quality_score",
    "decision",
    "needs_human_review",
    "human_audit_count",
    "human_mean_quality_score",
) + tuple(f"pseudo_{criterion}" for criterion in ANSWER_CRITERIA)

TURN_EVIDENCE_DATASET_COLUMNS = (
    "group_id",
    "case_id",
    "sample_id",
    "question_output_id",
    "stage",
    "target_criterion",
    "context_text",
    "question_text",
    "question_type",
    "difficulty",
    "required_signals_json",
    "bonus_signals_json",
    "applicable_answer_criteria_json",
    "contradiction_rules_json",
    "answer_text",
    "generator_model_name",
    "target_quality_level_metadata",
    "target_length_band_metadata",
    "target_word_range_json",
    "counterfactual_type_metadata",
    "actual_word_count",
    "length_compliance",
    "ai_judge_count",
    "mean_quality_score",
    "decision",
    "needs_human_review",
    "applicability_json",
    "human_audit_count",
    "human_mean_quality_score",
) + tuple(f"pseudo_{criterion}" for criterion in TURN_EVIDENCE_CRITERIA)

EXPORT_BATCH_SIZE = 100
EXPORT_QUERY_ATTEMPTS = 5


class WeakSupervisionDatasetBuilder:
    @staticmethod
    def _query_with_retry(db: Session, query_factory):
        for attempt in range(1, EXPORT_QUERY_ATTEMPTS + 1):
            try:
                return query_factory().all()
            except OperationalError:
                try:
                    db.rollback()
                except OperationalError:
                    db.invalidate()
                if attempt >= EXPORT_QUERY_ATTEMPTS:
                    raise
                time.sleep(attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def _context_text(benchmark_case) -> str:
        context = {
            "language": benchmark_case.language,
            "stage": benchmark_case.stage,
            "target_criterion": benchmark_case.criterion,
            "cv_profile": benchmark_case.cv_profile_json,
            "job_profile": benchmark_case.job_profile_json,
            "interview_history": benchmark_case.interview_history_json,
        }
        return json.dumps(context, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _human_mean(annotations: list[Any], criteria: tuple[str, ...]) -> float | None:
        if not annotations:
            return None
        consensus_scores = []
        for criterion in criteria:
            values = [float(item.scores_json[criterion]) for item in annotations]
            consensus_scores.append(float(statistics.median(values)))
        return round(float(statistics.mean(consensus_scores)), 3)

    @classmethod
    def question_rows(
        cls,
        db: Session,
        prompt_version: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        rows: list[dict[str, Any]] = []
        successful_outputs = 0
        last_output_id: str | None = None
        while True:
            def build_query():
                query = (
                    db.query(QuestionBenchmarkOutput)
                    .options(
                        load_only(
                            QuestionBenchmarkOutput.output_id,
                            QuestionBenchmarkOutput.run_id,
                            QuestionBenchmarkOutput.question_text,
                            QuestionBenchmarkOutput.model_name,
                            QuestionBenchmarkOutput.generation_metadata_json,
                        ),
                        selectinload(QuestionBenchmarkOutput.ai_judgments).load_only(
                            QuestionAIJudgment.rubric_version,
                            QuestionAIJudgment.scores_json,
                        ),
                        selectinload(QuestionBenchmarkOutput.annotations).load_only(
                            QuestionQualityAnnotation.scores_json,
                        ),
                        selectinload(QuestionBenchmarkOutput.run)
                        .load_only(
                            QuestionBenchmarkRun.run_id,
                            QuestionBenchmarkRun.case_id,
                        )
                        .selectinload(QuestionBenchmarkRun.case)
                        .load_only(
                            QuestionBenchmarkCase.case_id,
                            QuestionBenchmarkCase.language,
                            QuestionBenchmarkCase.stage,
                            QuestionBenchmarkCase.criterion,
                            QuestionBenchmarkCase.cv_profile_json,
                            QuestionBenchmarkCase.job_profile_json,
                            QuestionBenchmarkCase.interview_history_json,
                        ),
                    )
                    .filter(QuestionBenchmarkOutput.status == "success")
                )
                if prompt_version:
                    query = query.join(QuestionBenchmarkRun).filter(
                        QuestionBenchmarkRun.prompt_version == prompt_version
                    )
                if last_output_id is not None:
                    query = query.filter(
                        QuestionBenchmarkOutput.output_id > last_output_id
                    )
                return query.order_by(QuestionBenchmarkOutput.output_id).limit(
                    EXPORT_BATCH_SIZE
                )

            outputs = cls._query_with_retry(db, build_query)
            if not outputs:
                break
            successful_outputs += len(outputs)
            if progress_callback:
                progress_callback("question", successful_outputs)
            for output in outputs:
                consensus = WeakSupervisionService.question_consensus(output)
                if consensus["status"] != "pseudo_label_ready":
                    continue
                benchmark_case = output.run.case
                question_metadata = output.generation_metadata_json or {}
                row: dict[str, Any] = {
                    "group_id": f"case-{benchmark_case.case_id}",
                    "case_id": benchmark_case.case_id,
                    "output_id": output.output_id,
                    "stage": benchmark_case.stage,
                    "target_criterion": benchmark_case.criterion,
                    "context_text": cls._context_text(benchmark_case),
                    "question_text": output.question_text,
                    "question_type": question_metadata.get("question_type") or "",
                    "difficulty": question_metadata.get("difficulty") or "",
                    "required_signals_json": json.dumps(
                        question_metadata.get("required_signals") or [],
                        ensure_ascii=False,
                    ),
                    "bonus_signals_json": json.dumps(
                        question_metadata.get("bonus_signals") or [],
                        ensure_ascii=False,
                    ),
                    "applicable_answer_criteria_json": json.dumps(
                        question_metadata.get("applicable_answer_criteria") or [],
                        ensure_ascii=False,
                    ),
                    "contradiction_rules_json": json.dumps(
                        question_metadata.get("contradiction_rules") or [],
                        ensure_ascii=False,
                    ),
                    "generator_model_name": output.model_name,
                    "ai_judge_count": consensus["ai_judge_count"],
                    "mean_quality_score": consensus["mean_quality_score"],
                    "decision": consensus["decision"],
                    "needs_human_review": consensus["needs_human_review"],
                    "human_audit_count": len(output.annotations),
                    "human_mean_quality_score": cls._human_mean(
                        output.annotations,
                        QUESTION_CRITERIA,
                    ),
                }
                row.update({
                    f"pseudo_{criterion}": consensus["pseudo_label_scores"][criterion]
                    for criterion in QUESTION_CRITERIA
                })
                rows.append(row)
            last_output_id = outputs[-1].output_id
        return rows, {
            "successful_outputs": successful_outputs,
            "exported_question_rows": len(rows),
        }

    @classmethod
    def answer_rows(
        cls,
        db: Session,
        prompt_version: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        rows: list[dict[str, Any]] = []
        successful_answer_samples = 0
        last_sample_id: str | None = None
        while True:
            def build_query():
                query = (
                    db.query(SyntheticAnswerSample)
                    .options(
                        load_only(
                            SyntheticAnswerSample.sample_id,
                            SyntheticAnswerSample.question_output_id,
                            SyntheticAnswerSample.answer_text,
                            SyntheticAnswerSample.generator_model_name,
                            SyntheticAnswerSample.target_quality_level,
                            SyntheticAnswerSample.generation_metadata_json,
                        ),
                        selectinload(SyntheticAnswerSample.ai_judgments).load_only(
                            AnswerAIJudgment.rubric_version,
                            AnswerAIJudgment.scores_json,
                        ),
                        selectinload(SyntheticAnswerSample.human_audits).load_only(
                            AnswerHumanAudit.scores_json,
                        ),
                        selectinload(SyntheticAnswerSample.question_output)
                        .load_only(
                            QuestionBenchmarkOutput.output_id,
                            QuestionBenchmarkOutput.run_id,
                            QuestionBenchmarkOutput.question_text,
                            QuestionBenchmarkOutput.generation_metadata_json,
                        )
                        .selectinload(QuestionBenchmarkOutput.run)
                        .load_only(
                            QuestionBenchmarkRun.run_id,
                            QuestionBenchmarkRun.case_id,
                        )
                        .selectinload(QuestionBenchmarkRun.case)
                        .load_only(
                            QuestionBenchmarkCase.case_id,
                            QuestionBenchmarkCase.language,
                            QuestionBenchmarkCase.stage,
                            QuestionBenchmarkCase.criterion,
                            QuestionBenchmarkCase.cv_profile_json,
                            QuestionBenchmarkCase.job_profile_json,
                            QuestionBenchmarkCase.interview_history_json,
                        ),
                    )
                    .filter(SyntheticAnswerSample.status == "success")
                )
                if prompt_version:
                    query = (
                        query.join(QuestionBenchmarkOutput)
                        .join(QuestionBenchmarkRun)
                        .filter(QuestionBenchmarkRun.prompt_version == prompt_version)
                    )
                if last_sample_id is not None:
                    query = query.filter(
                        SyntheticAnswerSample.sample_id > last_sample_id
                    )
                return query.order_by(SyntheticAnswerSample.sample_id).limit(
                    EXPORT_BATCH_SIZE
                )

            samples = cls._query_with_retry(db, build_query)
            if not samples:
                break
            successful_answer_samples += len(samples)
            if progress_callback:
                progress_callback("answer", successful_answer_samples)
            for sample in samples:
                is_v11 = prompt_version == "balanced-weak-v1.1"
                consensus = (
                    WeakSupervisionService.turn_evidence_consensus(sample)
                    if is_v11
                    else WeakSupervisionService.answer_consensus(sample)
                )
                if consensus["status"] != "pseudo_label_ready":
                    continue
                output = sample.question_output
                benchmark_case = output.run.case
                question_metadata = output.generation_metadata_json or {}
                answer_metadata = sample.generation_metadata_json or {}
                row: dict[str, Any] = {
                    "group_id": f"case-{benchmark_case.case_id}",
                    "case_id": benchmark_case.case_id,
                    "sample_id": sample.sample_id,
                    "question_output_id": output.output_id,
                    "stage": benchmark_case.stage,
                    "target_criterion": benchmark_case.criterion,
                    "context_text": cls._context_text(benchmark_case),
                    "question_text": output.question_text,
                    "question_type": question_metadata.get("question_type") or "",
                    "difficulty": question_metadata.get("difficulty") or "",
                    "required_signals_json": json.dumps(
                        question_metadata.get("required_signals") or [],
                        ensure_ascii=False,
                    ),
                    "bonus_signals_json": json.dumps(
                        question_metadata.get("bonus_signals") or [],
                        ensure_ascii=False,
                    ),
                    "applicable_answer_criteria_json": json.dumps(
                        question_metadata.get("applicable_answer_criteria") or [],
                        ensure_ascii=False,
                    ),
                    "contradiction_rules_json": json.dumps(
                        question_metadata.get("contradiction_rules") or [],
                        ensure_ascii=False,
                    ),
                    "answer_text": sample.answer_text,
                    "generator_model_name": sample.generator_model_name,
                    "target_quality_level_metadata": sample.target_quality_level,
                    "ai_judge_count": consensus["ai_judge_count"],
                    "mean_quality_score": consensus["mean_quality_score"],
                    "decision": consensus["decision"],
                    "needs_human_review": consensus["needs_human_review"],
                    "human_audit_count": len(sample.human_audits),
                    "human_mean_quality_score": cls._human_mean(
                        sample.human_audits,
                        TURN_EVIDENCE_CRITERIA if is_v11 else ANSWER_CRITERIA,
                    ),
                }
                if is_v11:
                    row.update({
                        "target_length_band_metadata": answer_metadata.get(
                            "target_length_band"
                        ) or "",
                        "target_word_range_json": json.dumps(
                            answer_metadata.get("target_word_range") or []
                        ),
                        "counterfactual_type_metadata": answer_metadata.get(
                            "counterfactual_type"
                        ) or "",
                        "actual_word_count": answer_metadata.get("actual_word_count") or 0,
                        "length_compliance": bool(
                            answer_metadata.get("length_compliance")
                        ),
                        "applicability_json": json.dumps(
                            consensus["applicability"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    })
                    row.update({
                        f"pseudo_{criterion}": (
                            consensus["pseudo_label_scores"].get(criterion)
                            if consensus["pseudo_label_scores"] is not None
                            else None
                        )
                        for criterion in TURN_EVIDENCE_CRITERIA
                    })
                else:
                    row.update({
                        f"pseudo_{criterion}": consensus["pseudo_label_scores"][criterion]
                        for criterion in ANSWER_CRITERIA
                    })
                rows.append(row)
            last_sample_id = samples[-1].sample_id
        return rows, {
            "successful_answer_samples": successful_answer_samples,
            "exported_answer_rows": len(rows),
        }
