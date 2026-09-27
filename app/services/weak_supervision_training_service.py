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

from app.services.weak_supervision_service import ANSWER_CRITERIA


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


def _text_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=1)),
        ("regressor", Ridge(alpha=1.0, solver="lsqr")),
    ])


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
            top_one_matches.append(float(np.argmax(group_reference) == np.argmax(group_predictions)))

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

    def train_csv(self, dataset_path: str | Path, artifact_path: str | Path, report_path: str | Path) -> WeakSupervisionTrainingResult:
        result = self.train(_read_csv(dataset_path))
        artifact = Path(artifact_path)
        report = Path(report_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.artifact, artifact)
        report.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
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
    def __init__(self, random_state: int = 42, minimum_rows: int = 20):
        self.random_state = random_state
        self.minimum_rows = minimum_rows

    @staticmethod
    def _combined_text(row: dict[str, Any]) -> str:
        return (
            f"CONTEXT: {row['context_text']}\n"
            f"STAGE: {row.get('stage', '')}\n"
            f"TARGET: {row.get('target_criterion', '')}\n"
            f"QUESTION: {row['question_text']}\n"
            f"ANSWER: {row['answer_text']}"
        )

    def train(self, rows: Iterable[dict[str, Any]]) -> WeakSupervisionTrainingResult:
        records = list(rows)
        if len(records) < self.minimum_rows:
            raise ValueError(f"Answer scorer needs at least {self.minimum_rows} pseudo-labeled rows")
        required = {"group_id", "context_text", "question_text", "answer_text", *ANSWER_TARGET_COLUMNS}
        missing = required.difference(records[0])
        if missing:
            raise ValueError(f"Answer dataset is missing columns: {', '.join(sorted(missing))}")

        texts = np.asarray([self._combined_text(row) for row in records], dtype=object)
        targets = np.asarray(
            [[float(row[column]) for column in ANSWER_TARGET_COLUMNS] for row in records],
            dtype=float,
        )
        groups = np.asarray([str(row["group_id"]) for row in records], dtype=object)
        if np.any((targets < 1) | (targets > 5)):
            raise ValueError("Answer quality targets must be between 1 and 5")
        train_indices, test_indices = _group_split(groups, self.random_state)

        model = _text_pipeline()
        model.fit(texts[train_indices].tolist(), targets[train_indices])
        predictions = np.clip(model.predict(texts[test_indices].tolist()), 1.0, 5.0)
        reference = targets[test_indices]
        report = {
            "model": "TF-IDF + Ridge multi-output candidate answer scorer",
            "label_source": "multi_llm_consensus_pseudo_labels",
            "dataset_rows": len(records),
            "context_groups": int(np.unique(groups).size),
            "train_rows": int(train_indices.size),
            "test_rows": int(test_indices.size),
            "split_strategy": "group_by_benchmark_case",
            "overall_pseudo_label_metrics": _regression_metrics(reference, predictions),
            "by_criterion": {
                criterion: _regression_metrics(reference[:, index], predictions[:, index])
                for index, criterion in enumerate(ANSWER_CRITERIA)
            },
        }
        artifact = {
            "model": model,
            "model_type": "candidate_answer_scorer",
            "criteria": ANSWER_CRITERIA,
            "score_range": (1.0, 5.0),
            "training_report": report,
        }
        return WeakSupervisionTrainingResult(artifact, report)

    def train_csv(self, dataset_path: str | Path, artifact_path: str | Path, report_path: str | Path) -> WeakSupervisionTrainingResult:
        result = self.train(_read_csv(dataset_path))
        artifact = Path(artifact_path)
        report = Path(report_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.artifact, artifact)
        report.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
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
        values = np.clip(self.artifact["model"].predict([text])[0], lower, upper)
        return {
            criterion: round(float(values[index]), 3)
            for index, criterion in enumerate(self.artifact["criteria"])
        }
