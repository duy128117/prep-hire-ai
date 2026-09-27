from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib

from app.services.weak_supervision_service import ANSWER_CRITERIA
from app.services.weak_supervision_training_service import (
    CandidateAnswerScorer,
    CandidateAnswerScorerTrainer,
    QuestionQualityRanker,
    QuestionQualityRankerTrainer,
)


def _question_rows() -> list[dict[str, str]]:
    rows = []
    for group_index in range(10):
        for quality in range(1, 5):
            rows.append({
                "group_id": f"case-{group_index}",
                "context_text": f"Backend context {group_index} Python API SQL",
                "stage": "technical",
                "target_criterion": "technical_knowledge",
                "question_text": f"quality{quality} technical question with depth level {quality}",
                "mean_quality_score": str(float(quality)),
                "human_mean_quality_score": str(float(quality)) if group_index % 2 == 0 else "",
            })
    return rows


def _answer_rows() -> list[dict[str, str]]:
    rows = []
    for group_index in range(10):
        for quality in range(1, 5):
            row = {
                "group_id": f"case-{group_index}",
                "context_text": f"Backend context {group_index} Python API SQL",
                "stage": "technical",
                "target_criterion": "technical_knowledge",
                "question_text": "Explain transaction handling.",
                "answer_text": f"quality{quality} answer reasoning evidence level {quality}",
            }
            row.update({f"pseudo_{criterion}": str(float(quality)) for criterion in ANSWER_CRITERIA})
            rows.append(row)
    return rows


class WeakSupervisionTrainingTests(unittest.TestCase):
    def test_question_ranker_uses_group_split_and_can_rank(self) -> None:
        result = QuestionQualityRankerTrainer().train(_question_rows())
        self.assertEqual(result.report["split_strategy"], "group_by_benchmark_case")
        self.assertEqual(result.report["dataset_rows"], 40)

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "question-ranker.joblib"
            joblib.dump(result.artifact, artifact_path)
            ranker = QuestionQualityRanker(artifact_path)
            ranked = ranker.rank(
                "Backend Python API SQL",
                ["quality1 shallow question", "quality4 technical depth question"],
                "technical",
                "technical_knowledge",
            )
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0]["predicted_quality_score"], ranked[1]["predicted_quality_score"])

    def test_answer_scorer_predicts_all_answer_criteria(self) -> None:
        result = CandidateAnswerScorerTrainer().train(_answer_rows())
        self.assertEqual(result.report["split_strategy"], "group_by_benchmark_case")
        self.assertEqual(set(result.report["by_criterion"]), set(ANSWER_CRITERIA))

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "answer-scorer.joblib"
            joblib.dump(result.artifact, artifact_path)
            scorer = CandidateAnswerScorer(artifact_path)
            scores = scorer.score(
                "Backend Python API SQL",
                "Explain transaction handling.",
                "I use transactions and rollback failed operations.",
                "technical",
                "technical_knowledge",
            )
        self.assertEqual(set(scores), set(ANSWER_CRITERIA))
        self.assertTrue(all(1 <= score <= 5 for score in scores.values()))

    def test_training_rejects_too_few_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 20"):
            QuestionQualityRankerTrainer().train(_question_rows()[:5])


if __name__ == "__main__":
    unittest.main()
