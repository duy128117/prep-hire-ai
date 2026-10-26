from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
from sklearn.pipeline import FeatureUnion

from app.services.weak_supervision_training_service import (
    WeakSupervisionTrainingResult,
    _validate_disjoint_partitions,
)


PAIRWISE_REQUIRED_COLUMNS = {
    "group_id",
    "context_text",
    "stage",
    "target_criterion",
    "question_text",
    "mean_quality_score",
}
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def read_question_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pairwise_ranking_metrics(
    records: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, float | int | None]:
    if len(records) != len(scores):
        raise ValueError("Records and scores must have equal length")
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped_indices[str(row["group_id"])].append(index)

    ndcg_values: list[float] = []
    top_one_matches: list[float] = []
    reciprocal_ranks: list[float] = []
    pairwise_matches: list[float] = []
    tied_top_groups = 0
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        reference = np.asarray(
            [float(records[index]["mean_quality_score"]) for index in indices],
            dtype=float,
        )
        estimate = np.asarray([float(scores[index]) for index in indices], dtype=float)
        ndcg_values.append(float(ndcg_score([reference], [estimate])))

        maximum = float(np.max(reference))
        true_top = np.flatnonzero(np.isclose(reference, maximum))
        tied_top_groups += int(len(true_top) > 1)
        predicted_order = np.argsort(-estimate, kind="stable")
        top_one_matches.append(float(int(predicted_order[0]) in set(true_top.tolist())))
        first_relevant_rank = min(
            rank
            for rank, candidate_index in enumerate(predicted_order, start=1)
            if int(candidate_index) in set(true_top.tolist())
        )
        reciprocal_ranks.append(1.0 / first_relevant_rank)

        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                reference_delta = reference[left] - reference[right]
                if np.isclose(reference_delta, 0.0):
                    continue
                estimate_delta = estimate[left] - estimate[right]
                pairwise_matches.append(
                    float(np.sign(reference_delta) == np.sign(estimate_delta))
                )

    return {
        "rows": len(records),
        "context_groups": len(grouped_indices),
        "evaluated_ranking_groups": len(ndcg_values),
        "tied_top_groups": tied_top_groups,
        "ndcg": round(float(np.mean(ndcg_values)), 6) if ndcg_values else None,
        "top_1_accuracy_tie_aware": (
            round(float(np.mean(top_one_matches)), 6)
            if top_one_matches
            else None
        ),
        "mrr": (
            round(float(np.mean(reciprocal_ranks)), 6)
            if reciprocal_ranks
            else None
        ),
        "pairwise_accuracy": (
            round(float(np.mean(pairwise_matches)), 6)
            if pairwise_matches
            else None
        ),
        "evaluated_pairs": len(pairwise_matches),
    }


class PairwiseQuestionRankerTrainer:
    def __init__(
        self,
        random_state: int = 42,
        minimum_rows: int = 20,
        c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
        pair_margins: tuple[float, ...] = (0.0, 0.1, 0.25),
    ) -> None:
        self.random_state = random_state
        self.minimum_rows = minimum_rows
        self.c_values = c_values
        self.pair_margins = pair_margins

    @staticmethod
    def _feature_text(row: dict[str, Any]) -> str:
        question = str(row["question_text"])
        context = str(row["context_text"])
        question_tokens = {
            token.casefold()
            for token in TOKEN_PATTERN.findall(question)
            if len(token) >= 3
        }
        context_tokens = {
            token.casefold()
            for token in TOKEN_PATTERN.findall(context)
            if len(token) >= 3
        }
        overlap = sorted(question_tokens & context_tokens)
        overlap_ratio = len(overlap) / max(1, len(question_tokens))
        overlap_bin = min(10, int(overlap_ratio * 10))
        length_bin = min(12, max(1, len(question.split()) // 10 + 1))
        overlap_features = " ".join(f"MATCH_{token}" for token in overlap[:30])
        return (
            f"STAGE_{row.get('stage', '')} "
            f"TARGET_{row.get('target_criterion', '')} "
            f"OVERLAP_BIN_{overlap_bin} LENGTH_BIN_{length_bin} "
            f"{overlap_features}\nQUESTION: {question}"
        )

    @staticmethod
    def _vectorizer() -> FeatureUnion:
        return FeatureUnion([
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ])

    @staticmethod
    def _validate_records(records: list[dict[str, Any]], name: str) -> None:
        if not records:
            raise ValueError(f"{name} partition must contain rows")
        missing = PAIRWISE_REQUIRED_COLUMNS.difference(records[0])
        if missing:
            raise ValueError(
                f"{name} partition is missing columns: {', '.join(sorted(missing))}"
            )
        scores = np.asarray(
            [float(row["mean_quality_score"]) for row in records],
            dtype=float,
        )
        if np.any((scores < 1) | (scores > 5)):
            raise ValueError("Question quality targets must be between 1 and 5")

    @staticmethod
    def _pairwise_examples(
        matrix: sparse.spmatrix,
        records: list[dict[str, Any]],
        margin: float,
    ) -> tuple[sparse.csr_matrix, np.ndarray, int]:
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(records):
            grouped_indices[str(row["group_id"])].append(index)

        differences: list[sparse.spmatrix] = []
        labels: list[int] = []
        skipped_ties = 0
        for indices in grouped_indices.values():
            for left_position in range(len(indices)):
                for right_position in range(left_position + 1, len(indices)):
                    left = indices[left_position]
                    right = indices[right_position]
                    delta = (
                        float(records[left]["mean_quality_score"])
                        - float(records[right]["mean_quality_score"])
                    )
                    if abs(delta) <= margin:
                        skipped_ties += 1
                        continue
                    difference = matrix[left] - matrix[right]
                    label = int(delta > 0)
                    differences.extend((difference, -difference))
                    labels.extend((label, 1 - label))
        if not differences:
            raise ValueError("No non-tied question pairs are available for training")
        return sparse.vstack(differences).tocsr(), np.asarray(labels), skipped_ties

    def _fit(
        self,
        records: list[dict[str, Any]],
        *,
        c_value: float,
        margin: float,
    ) -> tuple[FeatureUnion, LogisticRegression, dict[str, int]]:
        vectorizer = self._vectorizer()
        matrix = vectorizer.fit_transform(
            [self._feature_text(row) for row in records]
        )
        pair_matrix, labels, skipped_ties = self._pairwise_examples(
            matrix,
            records,
            margin,
        )
        classifier = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=2000,
            random_state=self.random_state,
        )
        classifier.fit(pair_matrix, labels)
        return vectorizer, classifier, {
            "training_pairs": int(labels.size // 2),
            "symmetric_training_examples": int(labels.size),
            "skipped_tied_or_near_tied_pairs": skipped_ties,
        }

    @classmethod
    def _scores(
        cls,
        vectorizer: FeatureUnion,
        classifier: LogisticRegression,
        records: list[dict[str, Any]],
    ) -> np.ndarray:
        matrix = vectorizer.transform([cls._feature_text(row) for row in records])
        return np.asarray(classifier.decision_function(matrix), dtype=float)

    @staticmethod
    def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(metrics["top_1_accuracy_tie_aware"] or 0),
            float(metrics["mrr"] or 0),
            float(metrics["pairwise_accuracy"] or 0),
            float(metrics["ndcg"] or 0),
        )

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
                f"Pairwise ranker needs at least {self.minimum_rows} training rows"
            )
        for name, records in (
            ("Question train", train_records),
            ("Question validation", validation_records),
            ("Question test", test_records),
        ):
            self._validate_records(records, name)
        group_counts = _validate_disjoint_partitions(
            train_records,
            validation_records,
            test_records,
        )

        trials: list[dict[str, Any]] = []
        best_trial: dict[str, Any] | None = None
        for margin in self.pair_margins:
            for c_value in self.c_values:
                vectorizer, classifier, pair_stats = self._fit(
                    train_records,
                    c_value=c_value,
                    margin=margin,
                )
                validation_scores = self._scores(
                    vectorizer,
                    classifier,
                    validation_records,
                )
                metrics = pairwise_ranking_metrics(
                    validation_records,
                    validation_scores,
                )
                trial = {
                    "c": c_value,
                    "pair_margin": margin,
                    **pair_stats,
                    "validation_metrics": metrics,
                }
                trials.append(trial)
                if best_trial is None or self._selection_key(metrics) > self._selection_key(
                    best_trial["validation_metrics"]
                ):
                    best_trial = trial
        if best_trial is None:
            raise RuntimeError("Pairwise hyperparameter search produced no model")

        fit_records = train_records + validation_records
        vectorizer, classifier, final_pair_stats = self._fit(
            fit_records,
            c_value=float(best_trial["c"]),
            margin=float(best_trial["pair_margin"]),
        )
        test_scores = self._scores(vectorizer, classifier, test_records)
        test_metrics = pairwise_ranking_metrics(test_records, test_scores)
        report = {
            "model": "Pairwise word+character TF-IDF logistic learning-to-rank",
            "label_source": "multi_llm_consensus_pseudo_labels",
            "dataset_rows": len(train_records) + len(validation_records) + len(test_records),
            "context_groups": sum(group_counts.values()),
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
            "test_rows": len(test_records),
            "train_context_groups": group_counts["train"],
            "validation_context_groups": group_counts["validation"],
            "test_context_groups": group_counts["test"],
            "split_strategy": "precomputed_group_by_benchmark_case",
            "selection_metric_order": [
                "top_1_accuracy_tie_aware",
                "mrr",
                "pairwise_accuracy",
                "ndcg",
            ],
            "hyperparameter_trials": trials,
            "selected_hyperparameters": {
                "c": best_trial["c"],
                "pair_margin": best_trial["pair_margin"],
            },
            "selected_validation_metrics": best_trial["validation_metrics"],
            "final_pair_stats": final_pair_stats,
            "test_metrics": test_metrics,
        }
        artifact = {
            "model_type": "pairwise_question_quality_ranker_v2",
            "vectorizer": vectorizer,
            "classifier": classifier,
            "training_report": report,
        }
        return WeakSupervisionTrainingResult(artifact, report)

    def train_partition_csv(
        self,
        train_path: str | Path,
        validation_path: str | Path,
        test_path: str | Path,
        artifact_path: str | Path,
        report_path: str | Path,
    ) -> WeakSupervisionTrainingResult:
        result = self.train_partitions(
            read_question_csv(train_path),
            read_question_csv(validation_path),
            read_question_csv(test_path),
        )
        artifact = Path(artifact_path)
        report = Path(report_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.artifact, artifact)
        report.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
        return result


class PairwiseQuestionRanker:
    def __init__(self, artifact_path: str | Path):
        self.artifact = joblib.load(artifact_path)
        if self.artifact.get("model_type") != "pairwise_question_quality_ranker_v2":
            raise ValueError("Artifact is not a V2 pairwise question ranker")

    def score_rows(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return PairwiseQuestionRankerTrainer._scores(
            self.artifact["vectorizer"],
            self.artifact["classifier"],
            rows,
        )

    def rank(
        self,
        context_text: str,
        questions: list[str],
        stage: str = "",
        target_criterion: str = "",
    ) -> list[dict[str, Any]]:
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
        scores = self.score_rows(rows)
        ranked = [
            {
                "question": question,
                "pairwise_utility": round(float(score), 6),
            }
            for question, score in zip(questions, scores)
        ]
        return sorted(ranked, key=lambda item: item["pairwise_utility"], reverse=True)
