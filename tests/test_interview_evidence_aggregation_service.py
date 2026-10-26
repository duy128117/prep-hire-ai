from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.services.interview_evidence_aggregation_service import (
    InterviewEvidenceAggregator,
)


SCORES = {
    "answer_relevance": 4.0,
    "technical_correctness": 4.2,
    "expected_signal_coverage": 3.8,
    "reasoning_quality": 4.1,
    "evidence_quality": 3.7,
    "clarity_conciseness": 4.3,
    "consistency": 4.0,
}


def turn(index: int, criterion: str, answer: str, scores: dict | None = None):
    values = scores or SCORES
    return SimpleNamespace(
        turn_index=index,
        stage="technical",
        criterion=criterion,
        answer_text=answer,
        answer_scores_json={
            "status": "success",
            "scores": values,
            "applicable_criteria": list(values),
        },
    )


def session(turns, status="completed"):
    return SimpleNamespace(turns=turns, status=status)


class InterviewEvidenceAggregatorTests(unittest.TestCase):
    def test_answer_length_does_not_change_aggregation(self) -> None:
        short = InterviewEvidenceAggregator.aggregate(
            session([turn(0, "technical_knowledge", "Short answer.")])
        )
        long = InterviewEvidenceAggregator.aggregate(
            session(
                [
                    turn(
                        0,
                        "technical_knowledge",
                        "Long answer text " * 100,
                    )
                ]
            )
        )

        self.assertEqual(
            short["competencies"]["technical_knowledge"]["score"],
            long["competencies"]["technical_knowledge"]["score"],
        )

    def test_missing_direct_evidence_has_no_score(self) -> None:
        result = InterviewEvidenceAggregator.aggregate(
            session([turn(0, "communication", "Clear introduction.")])
        )

        technical = result["competencies"]["technical_knowledge"]
        self.assertIsNone(technical["score"])
        self.assertEqual(technical["status"], "insufficient_evidence")

    def test_low_consistency_and_divergence_create_flags(self) -> None:
        low = dict(SCORES, technical_correctness=1.2, consistency=2.0)
        high = dict(SCORES, technical_correctness=5.0, consistency=4.5)
        result = InterviewEvidenceAggregator.aggregate(
            session(
                [
                    turn(0, "technical_knowledge", "First", low),
                    turn(1, "technical_knowledge", "Second", high),
                ]
            )
        )
        flag_types = {item["type"] for item in result["flags"]}

        self.assertIn("possible_inconsistency", flag_types)
        self.assertIn("score_divergence", flag_types)

    def test_active_session_is_provisional(self) -> None:
        result = InterviewEvidenceAggregator.aggregate(
            session([turn(0, "communication", "Introduction")], status="active")
        )
        self.assertEqual(result["status"], "provisional")

    def test_overall_remains_limited_when_component_evidence_is_limited(self) -> None:
        result = InterviewEvidenceAggregator.aggregate(
            session(
                [
                    turn(0, "communication", "Introduction"),
                    turn(1, "experience_evidence", "Project"),
                    turn(2, "technical_knowledge", "Technical"),
                    turn(3, "problem_solving", "Problem"),
                    turn(4, "overall_readiness", "Closing"),
                ]
            )
        )

        overall = result["competencies"]["overall_readiness"]
        self.assertEqual(overall["status"], "limited")
        self.assertLess(overall["sufficient_competency_count"], 4)


if __name__ == "__main__":
    unittest.main()
