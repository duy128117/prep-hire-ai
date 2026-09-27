from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from app.services.calibration_service import (
    CONTEXT_FEATURE_COLUMNS,
    LLM_SCORE_COLUMNS,
    RUBRIC_CRITERIA,
    TARGET_COLUMNS,
    CandidateCalibrationTrainer,
    CandidateScoreCalibrator,
)


def _synthetic_rows(row_count: int = 80) -> list[dict[str, str]]:
    rng = np.random.default_rng(7)
    rows: list[dict[str, str]] = []
    for index in range(row_count):
        expert_scores = np.clip(rng.normal(3.0, 0.8, len(RUBRIC_CRITERIA)), 1, 5)
        # This fixture models a consistent LLM bias that calibration should learn.
        llm_scores = np.clip(expert_scores + 0.65 + rng.normal(0, 0.08, len(RUBRIC_CRITERIA)), 1, 5)
        row = {
            "candidate_id": f"candidate-{index // 2}",
            "job_id": f"job-{index % 8}",
            "cv_years_experience": str(1 + index % 6),
            "jd_years_required": str(2 + index % 4),
            "cv_skill_count": str(5 + index % 10),
            "jd_skill_count": str(6 + index % 8),
            "interview_answer_count": str(8 + index % 5),
            "interview_total_words": str(450 + index * 3),
        }
        row.update({column: str(llm_scores[pos]) for pos, column in enumerate(LLM_SCORE_COLUMNS)})
        row.update({column: str(expert_scores[pos]) for pos, column in enumerate(TARGET_COLUMNS)})
        rows.append(row)
    return rows


class CandidateCalibrationTrainerTests(unittest.TestCase):
    def test_training_improves_biased_llm_scores(self) -> None:
        result = CandidateCalibrationTrainer().train(_synthetic_rows())

        raw_mae = result.report["overall"]["raw_llm"]["mae"]
        calibrated_mae = result.report["overall"]["calibrated"]["mae"]
        self.assertEqual(result.report["split_strategy"], "group_by_candidate")
        self.assertLess(calibrated_mae, raw_mae)

    def test_artifact_can_be_loaded_for_prediction(self) -> None:
        result = CandidateCalibrationTrainer().train(_synthetic_rows())
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "calibrator.joblib"
            joblib.dump(result.artifact, artifact_path)
            calibrator = CandidateScoreCalibrator(artifact_path)
            features = {column: 3.0 for column in CONTEXT_FEATURE_COLUMNS}
            features.update({column: 4.0 for column in LLM_SCORE_COLUMNS})

            prediction = calibrator.predict(features)

        self.assertEqual(set(prediction), set(RUBRIC_CRITERIA))
        self.assertTrue(all(1 <= score <= 5 for score in prediction.values()))

    def test_rejects_dataset_without_enough_expert_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 20"):
            CandidateCalibrationTrainer().train(_synthetic_rows(5))


if __name__ == "__main__":
    unittest.main()

