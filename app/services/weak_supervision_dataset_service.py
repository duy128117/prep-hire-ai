from __future__ import annotations

import json
import statistics
from typing import Any

from sqlalchemy.orm import Session

from app.models.question_benchmark import QuestionBenchmarkOutput
from app.models.weak_supervision import SyntheticAnswerSample
from app.services.weak_supervision_service import (
    ANSWER_CRITERIA,
    QUESTION_CRITERIA,
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


class WeakSupervisionDatasetBuilder:
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
    def question_rows(cls, db: Session) -> tuple[list[dict[str, Any]], dict[str, int]]:
        rows: list[dict[str, Any]] = []
        outputs = db.query(QuestionBenchmarkOutput).filter(QuestionBenchmarkOutput.status == "success").all()
        for output in outputs:
            consensus = WeakSupervisionService.question_consensus(output)
            if consensus["status"] != "pseudo_label_ready":
                continue
            benchmark_case = output.run.case
            row: dict[str, Any] = {
                "group_id": f"case-{benchmark_case.case_id}",
                "case_id": benchmark_case.case_id,
                "output_id": output.output_id,
                "stage": benchmark_case.stage,
                "target_criterion": benchmark_case.criterion,
                "context_text": cls._context_text(benchmark_case),
                "question_text": output.question_text,
                "generator_model_name": output.model_name,
                "ai_judge_count": consensus["ai_judge_count"],
                "mean_quality_score": consensus["mean_quality_score"],
                "decision": consensus["decision"],
                "needs_human_review": consensus["needs_human_review"],
                "human_audit_count": len(output.annotations),
                "human_mean_quality_score": cls._human_mean(output.annotations, QUESTION_CRITERIA),
            }
            row.update({
                f"pseudo_{criterion}": consensus["pseudo_label_scores"][criterion]
                for criterion in QUESTION_CRITERIA
            })
            rows.append(row)
        return rows, {"successful_outputs": len(outputs), "exported_question_rows": len(rows)}

    @classmethod
    def answer_rows(cls, db: Session) -> tuple[list[dict[str, Any]], dict[str, int]]:
        rows: list[dict[str, Any]] = []
        samples = db.query(SyntheticAnswerSample).filter(SyntheticAnswerSample.status == "success").all()
        for sample in samples:
            consensus = WeakSupervisionService.answer_consensus(sample)
            if consensus["status"] != "pseudo_label_ready":
                continue
            output = sample.question_output
            benchmark_case = output.run.case
            row: dict[str, Any] = {
                "group_id": f"case-{benchmark_case.case_id}",
                "case_id": benchmark_case.case_id,
                "sample_id": sample.sample_id,
                "question_output_id": output.output_id,
                "stage": benchmark_case.stage,
                "target_criterion": benchmark_case.criterion,
                "context_text": cls._context_text(benchmark_case),
                "question_text": output.question_text,
                "answer_text": sample.answer_text,
                "generator_model_name": sample.generator_model_name,
                "target_quality_level_metadata": sample.target_quality_level,
                "ai_judge_count": consensus["ai_judge_count"],
                "mean_quality_score": consensus["mean_quality_score"],
                "decision": consensus["decision"],
                "needs_human_review": consensus["needs_human_review"],
                "human_audit_count": len(sample.human_audits),
                "human_mean_quality_score": cls._human_mean(sample.human_audits, ANSWER_CRITERIA),
            }
            row.update({
                f"pseudo_{criterion}": consensus["pseudo_label_scores"][criterion]
                for criterion in ANSWER_CRITERIA
            })
            rows.append(row)
        return rows, {"successful_answer_samples": len(samples), "exported_answer_rows": len(rows)}
