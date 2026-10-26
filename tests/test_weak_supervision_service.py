from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm_provider_service import LLMProviderError, LLMJsonResult, TokenUsage
from app.services.weak_supervision_service import (
    TurnEvidenceAIJudgeEngine,
    WeakSupervisionEngine,
)


class WeakSupervisionEngineTests(unittest.TestCase):
    def test_turn_judge_prompt_hides_answer_generation_targets(self) -> None:
        benchmark_case = SimpleNamespace(
            cv_profile_json={"skills": ["Python"]},
            job_profile_json={"title": "Backend Developer"},
            interview_history_json=[],
            stage="technical",
            criterion="technical_knowledge",
        )
        output = SimpleNamespace(
            run=SimpleNamespace(case=benchmark_case),
            question_text="Explain a transaction rollback strategy.",
            generation_metadata_json={
                "question_type": "technical_scenario",
                "difficulty": 3,
                "required_signals": ["atomicity", "rollback"],
                "bonus_signals": ["savepoint"],
                "applicable_answer_criteria": [
                    "answer_relevance",
                    "technical_correctness",
                ],
                "contradiction_rules": [],
            },
        )
        sample = SimpleNamespace(
            question_output=output,
            answer_text="I use a transaction and roll back when an operation fails.",
            generation_metadata_json={
                "target_quality_level": 5,
                "target_length_band": "short",
            },
        )

        prompt = TurnEvidenceAIJudgeEngine.build_prompt(sample)

        self.assertNotIn("target_quality_level", prompt)
        self.assertNotIn("target_length_band", prompt)
        self.assertIn("required_signals", prompt)

    @patch("app.services.weak_supervision_service.settings.CV_AI_ENRICHMENT_ENABLED", True)
    @patch("app.services.weak_supervision_service.LLMProviderService.generate_json")
    def test_call_json_retries_invalid_json_response(self, generate_json) -> None:
        generate_json.side_effect = [
            LLMProviderError(
                "inception",
                "inception returned invalid JSON",
                error_code="invalid_json",
            ),
            LLMJsonResult(
                "inception",
                "mercury-2.5",
                {"answer": "Valid answer"},
                TokenUsage(10, 5, 15),
                9,
            ),
        ]

        data, prompt_hash, latency_ms = WeakSupervisionEngine.call_json(
            "inception",
            "generate an answer",
            "answer_generation",
        )

        self.assertEqual(data, {"answer": "Valid answer"})
        self.assertEqual(len(prompt_hash), 64)
        self.assertEqual(latency_ms, 9)
        self.assertEqual(generate_json.call_count, 2)
        retry_prompt = generate_json.call_args.args[1]
        self.assertIn("previous response was not valid JSON", retry_prompt)

    def test_judgment_retries_out_of_range_score_and_accepts_numeric_string(self) -> None:
        bad = ({"scores": {"adaptivity": 0}}, "bad-hash", 5)
        corrected = (
            {
                "scores": {"adaptivity": "4"},
                "evidence": {"adaptivity": "Fits the interview stage."},
                "recommendation": "accept",
            },
            "good-hash",
            7,
        )

        with patch.object(
            WeakSupervisionEngine,
            "call_json",
            side_effect=[bad, corrected],
        ) as call_json:
            result = WeakSupervisionEngine.call_judgment(
                "example-model",
                "judge this",
                "question_judgment",
                ("adaptivity",),
            )

        self.assertEqual(result.scores, {"adaptivity": 4.0})
        self.assertEqual(result.prompt_hash, "good-hash")
        self.assertEqual(result.latency_ms, 12)
        self.assertEqual(call_json.call_count, 2)
        self.assertIn("violated the rubric validation", call_json.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
