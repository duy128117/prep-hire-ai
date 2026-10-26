from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from app.services.question_ranking_v2_service import (
    PairwiseQuestionRanker,
    PairwiseQuestionRankerTrainer,
    pairwise_ranking_metrics,
)
from scripts.export_question_pairwise_datasets import build_pairwise_rows


def _rows(group_start: int, group_end: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_index in range(group_start, group_end):
        for quality in range(1, 5):
            rows.append({
                "group_id": f"case-{group_index}",
                "context_text": f"Python API context {group_index}",
                "stage": "technical",
                "target_criterion": "technical_knowledge",
                "question_text": (
                    f"quality{quality} question reasoning evidence detail {quality}"
                ),
                "mean_quality_score": str(float(quality)),
            })
    return rows


class PairwiseQuestionRankerTests(unittest.TestCase):
    def test_pairwise_export_skips_ties_and_keeps_preferred_question(self) -> None:
        rows = _rows(0, 1)
        rows[1]["mean_quality_score"] = rows[0]["mean_quality_score"]

        pairs, skipped = build_pairwise_rows(rows, margin=0.0)

        self.assertEqual(skipped, 1)
        self.assertTrue(all(
            pair["preferred_quality_score"] > pair["rejected_quality_score"]
            for pair in pairs
        ))

    def test_tie_aware_top_one_accepts_any_reference_winner(self) -> None:
        rows = [
            {"group_id": "case-1", "mean_quality_score": "5"},
            {"group_id": "case-1", "mean_quality_score": "5"},
            {"group_id": "case-1", "mean_quality_score": "2"},
        ]

        metrics = pairwise_ranking_metrics(rows, np.asarray([0.1, 0.9, 0.2]))

        self.assertEqual(metrics["top_1_accuracy_tie_aware"], 1.0)
        self.assertEqual(metrics["tied_top_groups"], 1)

    def test_pairwise_ranker_trains_and_ranks(self) -> None:
        trainer = PairwiseQuestionRankerTrainer(
            minimum_rows=8,
            c_values=(1.0,),
            pair_margins=(0.0,),
        )
        result = trainer.train_partitions(
            _rows(0, 5),
            _rows(5, 8),
            _rows(8, 11),
        )

        self.assertEqual(
            result.artifact["model_type"],
            "pairwise_question_quality_ranker_v2",
        )
        self.assertIn("mrr", result.report["test_metrics"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranker.joblib"
            joblib.dump(result.artifact, path)
            ranker = PairwiseQuestionRanker(path)
            ranked = ranker.rank(
                "Python API context",
                ["quality1 question", "quality4 reasoning evidence detail"],
                "technical",
                "technical_knowledge",
            )

        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(
            ranked[0]["pairwise_utility"],
            ranked[1]["pairwise_utility"],
        )


if __name__ == "__main__":
    unittest.main()
