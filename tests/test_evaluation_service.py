from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.models.interview_session import InterviewSession
from app.research.rubrics import CANDIDATE_EVALUATION_RUBRIC
from app.services.evaluation_service import (
    CandidateEvaluationEngine,
    ExpertConsensusService,
    RUBRIC_CRITERIA,
)
from app.services.research_dataset_service import CalibrationDatasetBuilder


def _scores(offset: float = 0.0) -> dict[str, float]:
    return {
        criterion: max(1.0, min(5.0, 3.0 + offset + position * 0.1))
        for position, criterion in enumerate(RUBRIC_CRITERIA)
    }


class CandidateEvaluationEngineTests(unittest.TestCase):
    def test_evaluation_prompt_uses_session_language(self) -> None:
        session = SimpleNamespace(
            language="vi",
            cv_profile_json={},
            job_profile_json={},
            cv_text="Backend intern profile",
            jd_text="Backend intern requirements",
            turns=[],
        )
        prompt = CandidateEvaluationEngine.build_prompt(session)
        self.assertIn("overall_explanation in Vietnamese", prompt)

    def test_parses_complete_rubric_response(self) -> None:
        payload = {
            "scores": _scores(),
            "evidence": {criterion: ["Observed interview evidence"] for criterion in RUBRIC_CRITERIA},
            "overall_explanation": "The scores are based on the completed interview.",
        }
        scores, evidence, explanation = CandidateEvaluationEngine.parse_response(payload)
        self.assertEqual(set(scores), set(CANDIDATE_EVALUATION_RUBRIC["criteria"]))
        self.assertEqual(len(evidence), 7)
        self.assertTrue(explanation)

    def test_rejects_incomplete_ai_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing a valid score"):
            CandidateEvaluationEngine.parse_response({"scores": {"communication": 4}})

    def test_evaluation_requires_external_ai_opt_in(self) -> None:
        session = InterviewSession(
            session_id="no-opt-in",
            status="completed",
            allow_external_ai=False,
        )
        with self.assertRaisesRegex(ValueError, "not enabled"):
            CandidateEvaluationEngine.evaluate(session)


class ExpertConsensusServiceTests(unittest.TestCase):
    def test_three_experts_produce_median_reference(self) -> None:
        annotations = [
            SimpleNamespace(scores_json=_scores(-0.5)),
            SimpleNamespace(scores_json=_scores(0.0)),
            SimpleNamespace(scores_json=_scores(0.5)),
        ]
        result = ExpertConsensusService.consensus_from_annotations(annotations)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["expert_count"], 3)
        self.assertEqual(result["reference_scores"]["technical_knowledge"], 3.0)
        self.assertIsNotNone(result["inter_rater_icc_2_k"])

    def test_two_experts_are_not_a_reference_standard(self) -> None:
        annotations = [
            SimpleNamespace(scores_json=_scores(-0.2)),
            SimpleNamespace(scores_json=_scores(0.2)),
        ]
        result = ExpertConsensusService.consensus_from_annotations(annotations)
        self.assertEqual(result["status"], "insufficient_experts")
        self.assertIsNone(result["reference_scores"])
        self.assertEqual(result["criterion_stats"], {})
        self.assertIsNone(result["inter_rater_icc_2_k"])


class CalibrationDatasetBuilderTests(unittest.TestCase):
    def test_training_row_uses_stable_pseudonym_instead_of_user_id(self) -> None:
        session = SimpleNamespace(
            session_id="session-abc",
            user_id=42,
            cv_profile_json={"cv_years_experience": 2, "cv_skills": [{"name": "Python"}]},
            job_profile_json={"job_id": 7, "job_years_required": 1, "job_skills": []},
            turns=[SimpleNamespace(answer_text="A detailed answer", answer_word_count=3)],
        )
        evaluation = SimpleNamespace(scores_json=_scores())
        consensus = {
            "reference_scores": _scores(-0.2),
            "expert_count": 3,
        }

        first = CalibrationDatasetBuilder.build_row(session, evaluation, consensus, "test-secret-123456")
        second = CalibrationDatasetBuilder.build_row(session, evaluation, consensus, "test-secret-123456")

        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertNotEqual(first["candidate_id"], "user-42")
        self.assertTrue(first["candidate_id"].startswith("candidate-"))
        self.assertEqual(first["expert_count"], 3)


if __name__ == "__main__":
    unittest.main()
