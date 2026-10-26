from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib

from app.services.weak_supervision_service import ANSWER_CRITERIA, TURN_EVIDENCE_CRITERIA
from app.services.weak_supervision_training_service import (
    CandidateAnswerScorer,
    CandidateAnswerScorerTrainer,
    QuestionQualityRanker,
    QuestionQualityRankerTrainer,
)
from app.services.weak_supervision_split_service import split_weak_supervision_rows


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


def _answer_rows(
    criteria: tuple[str, ...] = ANSWER_CRITERIA,
) -> list[dict[str, str]]:
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
            row.update({f"pseudo_{criterion}": str(float(quality)) for criterion in criteria})
            rows.append(row)
    return rows


class WeakSupervisionTrainingTests(unittest.TestCase):
    @staticmethod
    def _partitions(rows: list[dict[str, str]]):
        return (
            [row for row in rows if int(row["group_id"].split("-")[-1]) < 6],
            [row for row in rows if int(row["group_id"].split("-")[-1]) in {6, 7}],
            [row for row in rows if int(row["group_id"].split("-")[-1]) >= 8],
        )

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

    def test_turn_evidence_scorer_supports_partially_applicable_labels(self) -> None:
        rows = _answer_rows(TURN_EVIDENCE_CRITERIA)
        for index, row in enumerate(rows):
            if index % 4 == 0:
                row["pseudo_technical_correctness"] = ""

        result = CandidateAnswerScorerTrainer(
            criteria=TURN_EVIDENCE_CRITERIA,
        ).train(rows)

        self.assertEqual(
            set(result.report["by_criterion"]),
            set(TURN_EVIDENCE_CRITERIA),
        )
        self.assertLess(
            result.report["train_labeled_rows_by_criterion"][
                "technical_correctness"
            ],
            result.report["train_rows"],
        )

    def test_training_rejects_too_few_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 20"):
            QuestionQualityRankerTrainer().train(_question_rows()[:5])

    def test_precomputed_partitions_are_reported_separately(self) -> None:
        question_result = QuestionQualityRankerTrainer().train_partitions(
            *self._partitions(_question_rows())
        )
        answer_result = CandidateAnswerScorerTrainer().train_partitions(
            *self._partitions(_answer_rows())
        )

        self.assertEqual(question_result.report["train_rows"], 24)
        self.assertEqual(question_result.report["validation_rows"], 8)
        self.assertEqual(question_result.report["test_rows"], 8)
        self.assertIn("test_metrics", question_result.report)
        self.assertEqual(answer_result.report["test_context_groups"], 2)
        self.assertIn("validation_metrics", answer_result.report)

    def test_dataset_split_is_deterministic_and_group_disjoint(self) -> None:
        questions = _question_rows()
        answers = _answer_rows()
        first = split_weak_supervision_rows(questions, answers, random_state=42)
        second = split_weak_supervision_rows(questions, answers, random_state=42)

        self.assertEqual(first.group_ids, second.group_ids)
        train_groups = set(first.group_ids["train"])
        validation_groups = set(first.group_ids["validation"])
        test_groups = set(first.group_ids["test"])
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)
        self.assertEqual(
            {row["group_id"] for row in first.question_rows["test"]},
            {row["group_id"] for row in first.answer_rows["test"]},
        )


if __name__ == "__main__":
    unittest.main()
