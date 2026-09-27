from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.research.rubrics import CANDIDATE_EVALUATION_RUBRIC


RUBRIC_CRITERIA = tuple(CANDIDATE_EVALUATION_RUBRIC["criteria"])
CONTEXT_FEATURE_COLUMNS = (
    "cv_years_experience",
    "jd_years_required",
    "cv_skill_count",
    "jd_skill_count",
    "interview_answer_count",
    "interview_total_words",
)
LLM_SCORE_COLUMNS = tuple(f"llm_{criterion}" for criterion in RUBRIC_CRITERIA)
TARGET_COLUMNS = tuple(f"expert_consensus_{criterion}" for criterion in RUBRIC_CRITERIA)
FEATURE_COLUMNS = CONTEXT_FEATURE_COLUMNS + LLM_SCORE_COLUMNS
REQUIRED_COLUMNS = ("candidate_id", "job_id") + FEATURE_COLUMNS + TARGET_COLUMNS


@dataclass(frozen=True)
class CalibrationTrainingResult:
    artifact: dict[str, Any]
    report: dict[str, Any]


def _safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = pearsonr(left, right).statistic if method == "pearson" else spearmanr(left, right).statistic
    return None if np.isnan(value) else round(float(value), 6)


def _icc_consistency(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    """Calculate ICC(3,1) consistency for two score sources."""
    ratings = np.column_stack((reference, estimate)).astype(float)
    row_count, rater_count = ratings.shape
    if row_count < 2:
        return None

    row_means = ratings.mean(axis=1)
    column_means = ratings.mean(axis=0)
    grand_mean = ratings.mean()
    mean_square_rows = rater_count * np.square(row_means - grand_mean).sum() / (row_count - 1)
    residual = ratings - row_means[:, None] - column_means[None, :] + grand_mean
    mean_square_error = np.square(residual).sum() / ((row_count - 1) * (rater_count - 1))
    denominator = mean_square_rows + (rater_count - 1) * mean_square_error
    if denominator == 0:
        return None
    return round(float((mean_square_rows - mean_square_error) / denominator), 6)


def _metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | None]:
    reference_flat = reference.reshape(-1)
    estimate_flat = estimate.reshape(-1)
    errors = estimate_flat - reference_flat
    return {
        "mae": round(float(np.mean(np.abs(errors))), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 6),
        "pearson": _safe_correlation(reference_flat, estimate_flat, "pearson"),
        "spearman": _safe_correlation(reference_flat, estimate_flat, "spearman"),
        "icc_3_1": _icc_consistency(reference_flat, estimate_flat),
    }


class CandidateCalibrationTrainer:
    def __init__(self, random_state: int = 42, minimum_rows: int = 20):
        self.random_state = random_state
        self.minimum_rows = minimum_rows

    @staticmethod
    def read_csv(dataset_path: str | Path) -> list[dict[str, str]]:
        with Path(dataset_path).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _validate_and_convert(
        self,
        rows: Iterable[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        records = list(rows)
        if len(records) < self.minimum_rows:
            raise ValueError(
                f"Calibration training needs at least {self.minimum_rows} expert-annotated rows; "
                f"received {len(records)}."
            )

        missing = [column for column in REQUIRED_COLUMNS if column not in records[0]]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

        features: list[list[float]] = []
        targets: list[list[float]] = []
        groups: list[str] = []
        for row_number, row in enumerate(records, start=2):
            if not str(row.get("candidate_id", "")).strip():
                raise ValueError(f"candidate_id is required at CSV row {row_number}")
            try:
                feature_row = [float(row[column]) if str(row[column]).strip() else np.nan for column in FEATURE_COLUMNS]
                target_row = [float(row[column]) for column in TARGET_COLUMNS]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid numeric value at CSV row {row_number}: {exc}") from exc

            llm_scores = feature_row[-len(LLM_SCORE_COLUMNS) :]
            if any(np.isnan(score) for score in llm_scores):
                raise ValueError(f"LLM rubric scores are required at CSV row {row_number}")

            context_features = feature_row[: len(CONTEXT_FEATURE_COLUMNS)]
            if any(not np.isnan(value) and value < 0 for value in context_features):
                raise ValueError(f"Context features must be non-negative at CSV row {row_number}")

            invalid_scores = [score for score in target_row if not 1 <= score <= 5]
            invalid_scores.extend(score for score in llm_scores if not 1 <= score <= 5)
            if invalid_scores:
                raise ValueError(f"Rubric scores must be between 1 and 5 at CSV row {row_number}")

            features.append(feature_row)
            targets.append(target_row)
            groups.append(str(row["candidate_id"]).strip())

        return np.asarray(features, dtype=float), np.asarray(targets, dtype=float), np.asarray(groups)

    def train(self, rows: Iterable[dict[str, Any]]) -> CalibrationTrainingResult:
        features, targets, groups = self._validate_and_convert(rows)
        indices = np.arange(features.shape[0])

        if np.unique(groups).size >= 5:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
            train_indices, test_indices = next(splitter.split(indices, groups=groups))
            split_strategy = "group_by_candidate"
        else:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=0.2,
                random_state=self.random_state,
            )
            split_strategy = "random_rows_fallback"

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", RidgeCV(alphas=np.logspace(-3, 3, 13))),
            ]
        )
        model.fit(features[train_indices], targets[train_indices])

        calibrated = np.clip(model.predict(features[test_indices]), 1.0, 5.0)
        reference = targets[test_indices]
        raw_llm = features[test_indices, -len(LLM_SCORE_COLUMNS) :]

        criterion_metrics = {
            criterion: {
                "raw_llm": _metrics(reference[:, position], raw_llm[:, position]),
                "calibrated": _metrics(reference[:, position], calibrated[:, position]),
            }
            for position, criterion in enumerate(RUBRIC_CRITERIA)
        }
        report = {
            "model": "RidgeCV multi-output calibration",
            "dataset_rows": int(features.shape[0]),
            "train_rows": int(train_indices.size),
            "test_rows": int(test_indices.size),
            "split_strategy": split_strategy,
            "rubric_scale": [1, 5],
            "overall": {
                "raw_llm": _metrics(reference, raw_llm),
                "calibrated": _metrics(reference, calibrated),
            },
            "by_criterion": criterion_metrics,
        }
        artifact = {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "rubric_criteria": RUBRIC_CRITERIA,
            "score_range": (1.0, 5.0),
            "training_report": report,
        }
        return CalibrationTrainingResult(artifact=artifact, report=report)

    def train_csv(
        self,
        dataset_path: str | Path,
        model_output_path: str | Path,
        report_output_path: str | Path,
    ) -> CalibrationTrainingResult:
        result = self.train(self.read_csv(dataset_path))
        model_path = Path(model_output_path)
        report_path = Path(report_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.artifact, model_path)
        report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
        return result


class CandidateScoreCalibrator:
    def __init__(self, artifact_path: str | Path):
        self.artifact = joblib.load(artifact_path)

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        missing = [column for column in self.artifact["feature_columns"] if column not in features]
        if missing:
            raise ValueError(f"Missing calibration features: {', '.join(missing)}")

        values = np.asarray([[float(features[column]) for column in self.artifact["feature_columns"]]])
        lower, upper = self.artifact["score_range"]
        prediction = np.clip(self.artifact["model"].predict(values)[0], lower, upper)
        return {
            criterion: round(float(prediction[position]), 3)
            for position, criterion in enumerate(self.artifact["rubric_criteria"])
        }
