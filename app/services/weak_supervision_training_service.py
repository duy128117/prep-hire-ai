from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from app.services.weak_supervision_service import ANSWER_CRITERIA, TURN_EVIDENCE_CRITERIA


ANSWER_TARGET_COLUMNS = tuple(f"pseudo_{criterion}" for criterion in ANSWER_CRITERIA)


@dataclass(frozen=True)
class WeakSupervisionTrainingResult:
    artifact: dict[str, Any]
    report: dict[str, Any]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _regression_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | None]:
    reference_flat = reference.reshape(-1)
    estimate_flat = estimate.reshape(-1)
    error = estimate_flat - reference_flat
    correlation = None
    if reference_flat.size >= 2 and not np.all(reference_flat == reference_flat[0]) and not np.all(estimate_flat == estimate_flat[0]):
        value = spearmanr(reference_flat, estimate_flat).statistic
        correlation = None if np.isnan(value) else round(float(value), 6)
    return {
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 6),
        "spearman": correlation,
    }


def _group_split(groups: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(groups).size < 5:
        raise ValueError("Training needs at least 5 independent context groups")
    indices = np.arange(groups.size)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    return next(splitter.split(indices, groups=groups))


def _text_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=1)),
        ("regressor", Ridge(alpha=alpha, solver="lsqr")),
    ])


def _require_columns(
    records: list[dict[str, Any]],
    required: set[str],
    dataset_name: str,
) -> None:
    if not records:
        raise ValueError(f"{dataset_name} partition must contain rows")
    missing = required.difference(records[0])
    if missing:
        raise ValueError(
            f"{dataset_name} dataset is missing columns: {', '.join(sorted(missing))}"
        )


def _validate_disjoint_partitions(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, int]:
    groups = {
        "train": {str(row["group_id"]) for row in train_records},
        "validation": {str(row["group_id"]) for row in validation_records},
        "test": {str(row["group_id"]) for row in test_records},
    }
    if groups["train"] & groups["validation"]:
        raise ValueError("Train and validation context groups overlap")
    if groups["train"] & groups["test"]:
        raise ValueError("Train and test context groups overlap")
    if groups["validation"] & groups["test"]:
        raise ValueError("Validation and test context groups overlap")
    return {name: len(values) for name, values in groups.items()}


def _save_result(
    result: WeakSupervisionTrainingResult,
    artifact_path: str | Path,
    report_path: str | Path,
) -> None:
    artifact = Path(artifact_path)
    report = Path(report_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.artifact, artifact)
    report.write_text(json.dumps(result.report, indent=2), encoding="utf-8")


class QuestionQualityRankerTrainer:
    def __init__(self, random_state: int = 42, minimum_rows: int = 20):
        self.random_state = random_state
        self.minimum_rows = minimum_rows

    @staticmethod
    def _combined_text(row: dict[str, Any]) -> str:
        return (
            f"CONTEXT: {row['context_text']}\n"
            f"STAGE: {row.get('stage', '')}\n"
            f"TARGET: {row.get('target_criterion', '')}\n"
            f"QUESTION: {row['question_text']}"
        )

    def train(self, rows: Iterable[dict[str, Any]]) -> WeakSupervisionTrainingResult:
        records = list(rows)
        if len(records) < self.minimum_rows:
            raise ValueError(f"Question ranker needs at least {self.minimum_rows} pseudo-labeled rows")
        required = {"group_id", "context_text", "question_text", "mean_quality_score"}
        missing = required.difference(records[0])
        if missing:
            raise ValueError(f"Question dataset is missing columns: {', '.join(sorted(missing))}")

        texts = np.asarray([self._combined_text(row) for row in records], dtype=object)
        targets = np.asarray([float(row["mean_quality_score"]) for row in records], dtype=float)
        groups = np.asarray([str(row["group_id"]) for row in records], dtype=object)
        if np.any((targets < 1) | (targets > 5)):
            raise ValueError("Question quality targets must be between 1 and 5")
        train_indices, test_indices = _group_split(groups, self.random_state)

        model = _text_pipeline()
        model.fit(texts[train_indices].tolist(), targets[train_indices])
        predictions = np.clip(model.predict(texts[test_indices].tolist()), 1.0, 5.0)
        reference = targets[test_indices]
        test_groups = groups[test_indices]

        ndcg_values: list[float] = []
        top_one_matches: list[float] = []
        for group in np.unique(test_groups):
            mask = test_groups == group
            if int(mask.sum()) < 2:
                continue
            group_reference = reference[mask]
            group_predictions = predictions[mask]
            ndcg_values.append(float(ndcg_score([group_reference], [group_predictions])))
            selected_reference = group_reference[int(np.argmax(group_predictions))]
            top_one_matches.append(
                float(np.isclose(selected_reference, np.max(group_reference)))
            )

        human_reference: list[float] = []
        human_estimate: list[float] = []
        for index, prediction in zip(test_indices, predictions):
            value = str(records[int(index)].get("human_mean_quality_score") or "").strip()
            if value:
                human_reference.append(float(value))
                human_estimate.append(float(prediction))

        report = {
            "model": "TF-IDF + Ridge pointwise question quality ranker",
            "label_source": "multi_llm_consensus_pseudo_labels",
            "dataset_rows": len(records),
            "context_groups": int(np.unique(groups).size),
            "train_rows": int(train_indices.size),
            "test_rows": int(test_indices.size),
            "split_strategy": "group_by_benchmark_case",
            "pseudo_label_metrics": _regression_metrics(reference, predictions),
            "ndcg": round(float(np.mean(ndcg_values)), 6) if ndcg_values else None,
            "top_1_accuracy": round(float(np.mean(top_one_matches)), 6) if top_one_matches else None,
            "top_1_definition": "tie_aware_any_maximum_reference_score",
            "human_audit_test_rows": len(human_reference),
            "human_audit_metrics": (
                _regression_metrics(np.asarray(human_reference), np.asarray(human_estimate))
                if len(human_reference) >= 2 else None
            ),
        }
        artifact = {
            "model": model,
            "model_type": "question_quality_ranker",
            "score_range": (1.0, 5.0),
            "training_report": report,
        }
        return WeakSupervisionTrainingResult(artifact, report)

    @classmethod
    def _evaluate_partition(
        cls,
        model: Pipeline,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        texts = [cls._combined_text(row) for row in records]
        reference = np.asarray(
            [float(row["mean_quality_score"]) for row in records],
            dtype=float,
        )
        predictions = np.clip(model.predict(texts), 1.0, 5.0)
        groups = np.asarray([str(row["group_id"]) for row in records], dtype=object)
        ndcg_values: list[float] = []
        top_one_matches: list[float] = []
        for group in np.unique(groups):
            mask = groups == group
            if int(mask.sum()) < 2:
                continue
            group_reference = reference[mask]
            group_predictions = predictions[mask]
            ndcg_values.append(float(ndcg_score([group_reference], [group_predictions])))
            selected_reference = group_reference[int(np.argmax(group_predictions))]
            top_one_matches.append(
                float(np.isclose(selected_reference, np.max(group_reference)))
            )

        human_reference: list[float] = []
        human_estimate: list[float] = []
        for row, prediction in zip(records, predictions):
            value = str(row.get("human_mean_quality_score") or "").strip()
            if value:
                human_reference.append(float(value))
                human_estimate.append(float(prediction))

        return {
            "rows": len(records),
            "context_groups": int(np.unique(groups).size),
            "pseudo_label_metrics": _regression_metrics(reference, predictions),
            "ndcg": round(float(np.mean(ndcg_values)), 6) if ndcg_values else None,
            "top_1_accuracy": (
                round(float(np.mean(top_one_matches)), 6)
                if top_one_matches else None
            ),
            "top_1_definition": "tie_aware_any_maximum_reference_score",
            "human_audit_rows": len(human_reference),
            "human_audit_metrics": (
                _regression_metrics(
                    np.asarray(human_reference),
                    np.asarray(human_estimate),
                )
                if len(human_reference) >= 2 else None
            ),
        }

    def train_partitions(
        self,
        train_rows: Iterable[dict[str, Any]],
        validation_rows: Iterable[dict[str, Any]],
        test_rows: Iterable[dict[str, Any]],
    ) -> WeakSupervisionTrainingResult:
        train_records = list(train_rows)
        validation_records = list(validation_rows)
        test_records = list(test_rows)
        if len(train_records) < self.minimum_rows:
            raise ValueError(
                f"Question ranker needs at least {self.minimum_rows} training rows"
            )
        required = {"group_id", "context_text", "question_text", "mean_quality_score"}
        for name, records in (
            ("Question train", train_records),
            ("Question validation", validation_records),
            ("Question test", test_records),
        ):
            _require_columns(records, required, name)
        group_counts = _validate_disjoint_partitions(
            train_records,
            validation_records,
            test_records,
        )
        all_targets = np.asarray([
            float(row["mean_quality_score"])
            for row in train_records + validation_records + test_records
        ])
        if np.any((all_targets < 1) | (all_targets > 5)):
            raise ValueError("Question quality targets must be between 1 and 5")

        validation_model = _text_pipeline()
        validation_model.fit(
            [self._combined_text(row) for row in train_records],
            [float(row["mean_quality_score"]) for row in train_records],
        )
        validation_metrics = self._evaluate_partition(
            validation_model,
            validation_records,
        )

        fit_records = train_records + validation_records
        final_model = _text_pipeline()
        final_model.fit(
            [self._combined_text(row) for row in fit_records],
            [float(row["mean_quality_score"]) for row in fit_records],
        )
        test_metrics = self._evaluate_partition(final_model, test_records)
        report = {
            "model": "TF-IDF + Ridge pointwise question quality ranker",
            "label_source": "multi_llm_consensus_pseudo_labels",
            "dataset_rows": len(train_records) + len(validation_records) + len(test_records),
            "context_groups": sum(group_counts.values()),
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
            "test_rows": len(test_records),
            "train_context_groups": group_counts["train"],
            "validation_context_groups": group_counts["validation"],
            "test_context_groups": group_counts["test"],
            "split_strategy": (
                "precomputed_group_by_benchmark_case_"
                "stratified_by_stage_and_criterion"
            ),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "pseudo_label_metrics": test_metrics["pseudo_label_metrics"],
            "ndcg": test_metrics["ndcg"],
            "top_1_accuracy": test_metrics["top_1_accuracy"],
            "top_1_definition": test_metrics["top_1_definition"],
            "human_audit_test_rows": test_metrics["human_audit_rows"],
            "human_audit_metrics": test_metrics["human_audit_metrics"],
        }
        artifact = {
            "model": final_model,
            "model_type": "question_quality_ranker",
            "score_range": (1.0, 5.0),
            "training_report": report,
        }
        return WeakSupervisionTrainingResult(artifact, report)

    def train_csv(self, dataset_path: str | Path, artifact_path: str | Path, report_path: str | Path) -> WeakSupervisionTrainingResult:
        result = self.train(_read_csv(dataset_path))
        _save_result(result, artifact_path, report_path)
        return result

    def train_partition_csv(
        self,
        train_path: str | Path,
        validation_path: str | Path,
        test_path: str | Path,
        artifact_path: str | Path,
        report_path: str | Path,
    ) -> WeakSupervisionTrainingResult:
        result = self.train_partitions(
            _read_csv(train_path),
            _read_csv(validation_path),
            _read_csv(test_path),
        )
        _save_result(result, artifact_path, report_path)
        return result


class QuestionQualityRanker:
    def __init__(self, artifact_path: str | Path):
        self.artifact = joblib.load(artifact_path)

    def rank(self, context_text: str, questions: list[str], stage: str = "", target_criterion: str = "") -> list[dict[str, Any]]:
        if not questions:
            raise ValueError("At least one question is required")
        rows = [
            {
                "context_text": context_text,
                "stage": stage,
                "target_criterion": target_criterion,
                "question_text": question,
            }
            for question in questions
        ]
        texts = [QuestionQualityRankerTrainer._combined_text(row) for row in rows]
        lower, upper = self.artifact["score_range"]
        scores = np.clip(self.artifact["model"].predict(texts), lower, upper)
        ranked = [
            {"question": question, "predicted_quality_score": round(float(score), 3)}
            for question, score in zip(questions, scores)
        ]
        return sorted(ranked, key=lambda item: item["predicted_quality_score"], reverse=True)


class CandidateAnswerScorerTrainer:
    def __init__(
        self,
        random_state: int = 42,
        minimum_rows: int = 20,
        criteria: tuple[str, ...] = ANSWER_CRITERIA,
    ):
        if not criteria or len(criteria) != len(set(criteria)):
            raise ValueError("Answer scoring criteria must be non-empty and unique")
        self.random_state = random_state
        self.minimum_rows = minimum_rows
        self.criteria = tuple(criteria)
        self.target_columns = tuple(
            f"pseudo_{criterion}" for criterion in self.criteria
        )

    @staticmethod
    def _combined_text(row: dict[str, Any]) -> str:
        return (
            f"CONTEXT: {row['context_text']}\n"
            f"STAGE: {row.get('stage', '')}\n"
            f"TARGET: {row.get('target_criterion', '')}\n"
            f"QUESTION: {row['question_text']}\n"
            f"ANSWER: {row['answer_text']}"
        )

    @staticmethod
    def _target(row: dict[str, Any], column: str) -> float | None:
        raw_value = row.get(column)
        if raw_value is None or str(raw_value).strip() == "":
            return None
        value = float(raw_value)
        if value < 1 or value > 5:
            raise ValueError("Answer quality targets must be between 1 and 5")
        return value

    def _required_columns(self) -> set[str]:
        return {
            "group_id",
            "context_text",
            "question_text",
            "answer_text",
            *self.target_columns,
        }

    def _fit_models(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, Pipeline], dict[str, int]]:
        models: dict[str, Pipeline] = {}
        labeled_rows: dict[str, int] = {}
        for criterion, column in zip(self.criteria, self.target_columns):
            examples = [
                (self._combined_text(row), target)
                for row in records
                if (target := self._target(row, column)) is not None
            ]
            if len(examples) < 2:
                raise ValueError(
                    f"Answer criterion {criterion} needs at least 2 labeled rows"
                )
            model = _text_pipeline()
            model.fit(
                [text for text, _target_value in examples],
                [target_value for _text, target_value in examples],
            )
            models[criterion] = model
            labeled_rows[criterion] = len(examples)
        return models, labeled_rows

    def _evaluate_partition(
        self,
        models: dict[str, Pipeline],
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        all_reference: list[float] = []
        all_predictions: list[float] = []
        by_criterion: dict[str, dict[str, float | int | None]] = {}
        for criterion, column in zip(self.criteria, self.target_columns):
            examples = [
                (self._combined_text(row), target)
                for row in records
                if (target := self._target(row, column)) is not None
            ]
            if not examples:
                by_criterion[criterion] = {
                    "labeled_rows": 0,
                    "mae": None,
                    "rmse": None,
                    "spearman": None,
                }
                continue
            reference = np.asarray(
                [target for _text, target in examples],
                dtype=float,
            )
            predictions = np.clip(
                models[criterion].predict([text for text, _target in examples]),
                1.0,
                5.0,
            )
            metrics = _regression_metrics(reference, predictions)
            by_criterion[criterion] = {
                "labeled_rows": len(examples),
                **metrics,
            }
            all_reference.extend(reference.tolist())
            all_predictions.extend(predictions.tolist())

        overall_metrics = _regression_metrics(
            np.asarray(all_reference, dtype=float),
            np.asarray(all_predictions, dtype=float),
        )
        return {
            "rows": len(records),
            "context_groups": len({str(row["group_id"]) for row in records}),
            "labeled_scores": len(all_reference),
            "overall_pseudo_label_metrics": overall_metrics,
            "by_criterion": by_criterion,
        }

    def _artifact(
        self,
        models: dict[str, Pipeline],
        report: dict[str, Any],
    ) -> dict[str, Any]:
        model_type = (
            "candidate_turn_evidence_scorer"
            if self.criteria == TURN_EVIDENCE_CRITERIA
            else "candidate_answer_scorer"
        )
        return {
            "models": models,
            "model_type": model_type,
            "criteria": self.criteria,
            "score_range": (1.0, 5.0),
            "training_report": report,
        }

    def train(self, rows: Iterable[dict[str, Any]]) -> WeakSupervisionTrainingResult:
        records = list(rows)
        if len(records) < self.minimum_rows:
            raise ValueError(
                f"Answer scorer needs at least {self.minimum_rows} pseudo-labeled rows"
            )
        _require_columns(records, self._required_columns(), "Answer")
        groups = np.asarray([str(row["group_id"]) for row in records], dtype=object)
        train_indices, test_indices = _group_split(groups, self.random_state)
        train_records = [records[int(index)] for index in train_indices]
        test_records = [records[int(index)] for index in test_indices]
        models, labeled_rows = self._fit_models(train_records)
        test_metrics = self._evaluate_partition(models, test_records)
        report = {
            "model": "Independent TF-IDF + Ridge regressors per answer criterion",
            "label_source": "multi_llm_consensus_pseudo_labels",
            "criteria": list(self.criteria),
            "dataset_rows": len(records),
            "context_groups": int(np.unique(groups).size),
            "train_rows": len(train_records),
            "test_rows": len(test_records),
            "train_labeled_rows_by_criterion": labeled_rows,
            "split_strategy": "group_by_benchmark_case",
            "overall_pseudo_label_metrics": test_metrics[
                "overall_pseudo_label_metrics"
            ],
            "by_criterion": test_metrics["by_criterion"],
        }
        artifact = self._artifact(models, report)
        return WeakSupervisionTrainingResult(artifact, report)

    def train_partitions(
        self,
        train_rows: Iterable[dict[str, Any]],
        validation_rows: Iterable[dict[str, Any]],
        test_rows: Iterable[dict[str, Any]],
    ) -> WeakSupervisionTrainingResult:
        train_records = list(train_rows)
        validation_records = list(validation_rows)
        test_records = list(test_rows)
        if len(train_records) < self.minimum_rows:
            raise ValueError(
                f"Answer scorer needs at least {self.minimum_rows} training rows"
            )
        required = self._required_columns()
        for name, records in (
            ("Answer train", train_records),
            ("Answer validation", validation_records),
            ("Answer test", test_records),
        ):
            _require_columns(records, required, name)
        group_counts = _validate_disjoint_partitions(
            train_records,
            validation_records,
            test_records,
        )

        validation_models, train_labeled_rows = self._fit_models(train_records)
        validation_metrics = self._evaluate_partition(
            validation_models,
            validation_records,
        )

        fit_records = train_records + validation_records
        final_models, fit_labeled_rows = self._fit_models(fit_records)
        test_metrics = self._evaluate_partition(final_models, test_records)
        report = {
            "model": "Independent TF-IDF + Ridge regressors per answer criterion",
            "label_source": "multi_llm_consensus_pseudo_labels",
            "criteria": list(self.criteria),
            "dataset_rows": len(train_records) + len(validation_records) + len(test_records),
            "context_groups": sum(group_counts.values()),
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
            "test_rows": len(test_records),
            "train_context_groups": group_counts["train"],
            "validation_context_groups": group_counts["validation"],
            "test_context_groups": group_counts["test"],
            "train_labeled_rows_by_criterion": train_labeled_rows,
            "fit_labeled_rows_by_criterion": fit_labeled_rows,
            "split_strategy": (
                "precomputed_group_by_benchmark_case_"
                "stratified_by_stage_and_criterion"
            ),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "overall_pseudo_label_metrics": test_metrics[
                "overall_pseudo_label_metrics"
            ],
            "by_criterion": test_metrics["by_criterion"],
        }
        artifact = self._artifact(final_models, report)
        return WeakSupervisionTrainingResult(artifact, report)

    def train_csv(
        self,
        dataset_path: str | Path,
        artifact_path: str | Path,
        report_path: str | Path,
    ) -> WeakSupervisionTrainingResult:
        result = self.train(_read_csv(dataset_path))
        _save_result(result, artifact_path, report_path)
        return result

    def train_partition_csv(
        self,
        train_path: str | Path,
        validation_path: str | Path,
        test_path: str | Path,
        artifact_path: str | Path,
        report_path: str | Path,
    ) -> WeakSupervisionTrainingResult:
        result = self.train_partitions(
            _read_csv(train_path),
            _read_csv(validation_path),
            _read_csv(test_path),
        )
        _save_result(result, artifact_path, report_path)
        return result


class CandidateAnswerScorer:
    def __init__(self, artifact_path: str | Path):
        self.artifact = joblib.load(artifact_path)

    def score(self, context_text: str, question: str, answer: str, stage: str = "", target_criterion: str = "") -> dict[str, float]:
        row = {
            "context_text": context_text,
            "stage": stage,
            "target_criterion": target_criterion,
            "question_text": question,
            "answer_text": answer,
        }
        text = CandidateAnswerScorerTrainer._combined_text(row)
        lower, upper = self.artifact["score_range"]
        models = self.artifact.get("models")
        if models:
            return {
                criterion: round(
                    float(np.clip(models[criterion].predict([text])[0], lower, upper)),
                    3,
                )
                for criterion in self.artifact["criteria"]
            }
        values = np.clip(self.artifact["model"].predict([text])[0], lower, upper)
        return {
            criterion: round(float(values[index]), 3)
            for index, criterion in enumerate(self.artifact["criteria"])
        }
