from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base_class import Base
from app.models.interview_session import InterviewSession
from app.models.interview_turn import InterviewTurn
from app.models.interview_evidence_summary import InterviewEvidenceSummary
from app.services.interview_service import AdaptiveQuestionEngine, InterviewSessionService
from app.services.interview_service import GeneratedQuestion
from app.services.llm_provider_service import LLMProviderSpec
from app.services.interview_model_service import InterviewModelService
from app.services.llm_provider_service import (
    LLMJsonResult,
    LLMProviderError,
    LLMProviderService,
    TokenUsage,
)
from app.services.ai_service import AIService
from app.services.pdf_processor import CvIngestService
from app.api.v1.endpoints.interview import _session_response


CV_PROFILE = {
    "cv_level": "junior",
    "cv_years_experience": 1,
    "preferred_locations": [],
    "cv_certifications": [],
    "cv_skills": [{"name": "Python", "proficiency": 0.7, "years_of_experience": 1}],
}
JOB_PROFILE = {
    "job_id": 0,
    "title": "Backend Developer",
    "source_url": "uploaded://jd",
    "job_level": "junior",
    "job_years_required": 1,
    "job_location": None,
    "job_is_remote": False,
    "job_skills": [{"name": "FastAPI", "importance": 0.8, "required_proficiency": 0.7}],
}


class AdaptiveQuestionEngineTests(unittest.TestCase):
    def test_multi_provider_questions_are_ranked(self) -> None:
        specs = tuple(
            LLMProviderSpec(f"provider-{index}", f"model-{index}")
            for index in range(4)
        )

        def generated_for_provider(*args, **kwargs):
            provider = kwargs["provider"]
            index = int(provider.rsplit("-", 1)[1])
            return GeneratedQuestion(
                stage="technical",
                criterion="technical_knowledge",
                question=f"Candidate question {index}",
                expected_signal="Technical evidence",
                source="ai",
                model_name=f"model-{index}",
                prompt_hash=f"hash-{index}",
                latency_ms=10,
                provider=provider,
                applicable_answer_criteria=("answer_relevance",),
            )

        with (
            patch.object(LLMProviderService, "pool_specs", return_value=specs),
            patch.object(LLMProviderService, "is_configured", return_value=True),
            patch.object(AdaptiveQuestionEngine, "generate", side_effect=generated_for_provider),
            patch.object(
                InterviewModelService,
                "rank_questions",
                return_value=([0.1, 0.9, 0.3, 0.2], "pairwise_question_quality_ranker_v2"),
            ),
        ):
            selection = AdaptiveQuestionEngine.generate_ranked(
                1,
                5,
                "en",
                "Synthetic CV with Python project evidence.",
                "Synthetic JD requiring Python and API design.",
                CV_PROFILE,
                JOB_PROFILE,
                [],
            )

        self.assertEqual(selection.selected.question, "Candidate question 2")
        self.assertEqual(selection.selected_score, 0.9)
        self.assertEqual(
            len([item for item in selection.candidates if item.get("question")]),
            4,
        )

    @patch("app.services.interview_service.settings.CV_AI_ENRICHMENT_ENABLED", True)
    @patch("app.services.interview_service.LLMProviderService.generate_json")
    def test_question_generation_retries_invalid_json(self, generate_json) -> None:
        invalid = LLMProviderError(
            "inception",
            "inception returned invalid JSON",
            error_code="invalid_json",
        )
        generate_json.side_effect = [
            invalid,
            LLMJsonResult(
                "inception",
                "mercury-2.5",
                {
                    "question": "How did you validate your API design?",
                    "criterion": "technical_knowledge",
                    "expected_signal": "A concrete validation approach.",
                },
                TokenUsage(10, 5, 15),
                8,
            ),
        ]

        generated = AdaptiveQuestionEngine.generate(
            turn_index=1,
            max_questions=5,
            language="en",
            cv_text="Python backend developer with API project experience.",
            jd_text="Backend role requiring Python, FastAPI, SQL, and testing.",
            cv_profile=CV_PROFILE,
            job_profile=JOB_PROFILE,
            history=[],
            allow_external_ai=True,
            model_name="mercury-2.5",
            provider="inception",
            strict_external_ai=True,
        )

        self.assertEqual(generated.source, "ai")
        self.assertEqual(generate_json.call_count, 2)
        self.assertIn(
            "previous response was not valid JSON",
            generate_json.call_args.args[1],
        )

    @patch("app.services.interview_service.settings.CV_AI_ENRICHMENT_ENABLED", True)
    @patch("app.services.interview_service.LLMProviderService.generate_json")
    def test_question_generation_retries_missing_question(self, generate_json) -> None:
        invalid_content = LLMJsonResult(
            "inception",
            "mercury-2.5",
            {"criterion": "technical_knowledge"},
            TokenUsage(10, 5, 15),
            8,
        )
        valid_content = LLMJsonResult(
            "inception",
            "mercury-2.5",
            {
                "question": "How did you validate your API design?",
                "criterion": "technical_knowledge",
                "expected_signal": "A concrete validation approach.",
            },
            TokenUsage(10, 5, 15),
            8,
        )
        generate_json.side_effect = [invalid_content, valid_content]

        generated = AdaptiveQuestionEngine.generate(
            turn_index=1,
            max_questions=5,
            language="en",
            cv_text="Python backend developer with API project experience.",
            jd_text="Backend role requiring Python, FastAPI, SQL, and testing.",
            cv_profile=CV_PROFILE,
            job_profile=JOB_PROFILE,
            history=[],
            allow_external_ai=True,
            model_name="mercury-2.5",
            provider="inception",
            strict_external_ai=True,
        )

        self.assertEqual(generated.source, "ai")
        self.assertEqual(generate_json.call_count, 2)
        self.assertIn(
            "missing, empty, or oversized question field",
            generate_json.call_args.args[1],
        )

    @patch("app.services.interview_service.settings.CV_AI_ENRICHMENT_ENABLED", True)
    @patch("app.services.interview_service.LLMProviderService.generate_json")
    def test_strict_generation_rejects_repeated_missing_question(
        self,
        generate_json,
    ) -> None:
        generate_json.return_value = LLMJsonResult(
            "inception",
            "mercury-2.5",
            {"criterion": "technical_knowledge"},
            TokenUsage(10, 5, 15),
            8,
        )

        with self.assertRaises(LLMProviderError) as raised:
            AdaptiveQuestionEngine.generate(
                turn_index=1,
                max_questions=5,
                language="en",
                cv_text="Python backend developer.",
                jd_text="Backend role requiring API design.",
                cv_profile=CV_PROFILE,
                job_profile=JOB_PROFILE,
                history=[],
                allow_external_ai=True,
                provider="inception",
                strict_external_ai=True,
            )

        self.assertEqual(raised.exception.error_code, "invalid_content")
        self.assertEqual(generate_json.call_count, 3)

    def test_redacts_contact_data_before_external_prompt(self) -> None:
        prompt = AdaptiveQuestionEngine.build_prompt(
            stage="technical",
            criterion="technical_knowledge",
            language="vi",
            cv_text="Email person@example.com phone +84 912 345 678 https://portfolio.example",
            jd_text="Backend role using Python and FastAPI for production services.",
            cv_profile=CV_PROFILE,
            job_profile=JOB_PROFILE,
            history=[],
        )

        self.assertNotIn("person@example.com", prompt)
        self.assertNotIn("912 345 678", prompt)
        self.assertNotIn("portfolio.example", prompt)
        self.assertIn("[REDACTED_EMAIL]", prompt)

    @patch("app.services.interview_service.settings.CV_AI_ENRICHMENT_ENABLED", True)
    @patch("app.services.interview_service.LLMProviderService.generate_json")
    def test_dataset_controls_lock_difficulty_and_required_applicability(
        self,
        generate_json,
    ) -> None:
        generate_json.return_value = LLMJsonResult(
            "groq",
            "openai/gpt-oss-120b",
            {
                "question": "How would you diagnose a slow API?",
                "criterion": "problem_solving",
                "expected_signal": "A structured diagnostic process.",
                "question_type": "problem_solving",
                "difficulty": 5,
                "required_signals": ["measure", "isolate"],
                "bonus_signals": ["distributed tracing"],
                "applicable_answer_criteria": ["reasoning_quality"],
                "contradiction_rules": [],
            },
            TokenUsage(10, 5, 15),
            8,
        )

        generated = AdaptiveQuestionEngine.generate(
            turn_index=3,
            max_questions=7,
            language="en",
            cv_text="Python backend developer.",
            jd_text="Backend role requiring production diagnostics.",
            cv_profile=CV_PROFILE,
            job_profile=JOB_PROFILE,
            history=[],
            allow_external_ai=True,
            provider="groq",
            strict_external_ai=True,
            generation_controls={"difficulty": 2},
        )

        self.assertEqual(generated.difficulty, 2)
        self.assertIn("reasoning_quality", generated.applicable_answer_criteria)
        self.assertIn("answer_relevance", generated.applicable_answer_criteria)
        self.assertIn("expected_signal_coverage", generated.applicable_answer_criteria)
        self.assertIn("clarity_conciseness", generated.applicable_answer_criteria)
        self.assertIn("consistency", generated.applicable_answer_criteria)

    def test_short_answer_triggers_follow_up(self) -> None:
        history = [
            {
                "stage": "technical",
                "criterion": "technical_knowledge",
                "question": "Explain dependency injection.",
                "answer": "I used it before.",
            }
        ]
        self.assertEqual(AdaptiveQuestionEngine.determine_stage(4, 8, history), "follow_up")

    def test_external_ai_is_opt_in(self) -> None:
        generated = AdaptiveQuestionEngine.generate(
            turn_index=0,
            max_questions=5,
            language="en",
            cv_text="Python backend developer with one year of project experience.",
            jd_text="Backend role requiring Python, APIs, SQL, testing, and teamwork.",
            cv_profile=CV_PROFILE,
            job_profile=JOB_PROFILE,
            history=[],
            allow_external_ai=False,
        )
        self.assertEqual(generated.source, "fallback")
        self.assertIsNone(generated.model_name)

    def test_context_extraction_does_not_call_ai_without_opt_in(self) -> None:
        cv_text = "Python backend developer with API, SQL, testing, and project experience."
        with (
            patch.object(CvIngestService, "_collect_skill_candidates", return_value=[]),
            patch.object(CvIngestService, "_collect_skill_alias_lookup", return_value={}),
            patch.object(AIService, "extract_cv_profile") as ai_extract,
        ):
            CvIngestService.extract_profile(object(), cv_text, use_ai=False)

        ai_extract.assert_not_called()

    def test_session_response_contains_current_question(self) -> None:
        now = datetime.now(timezone.utc)
        turn = SimpleNamespace(
            turn_index=0,
            stage="introduction",
            question_text="Please introduce yourself.",
            answer_text=None,
            question_source="fallback",
            model_name=None,
            generation_latency_ms=0,
            question_candidates_json=[
                {
                    "provider": "groq",
                    "model_name": "model-a",
                    "question": "Please introduce yourself.",
                    "criterion": "role_relevance",
                    "question_type": "behavioral",
                    "difficulty": 2,
                    "pairwise_utility": 0.42,
                    "selected": True,
                    "generation_latency_ms": 120,
                },
                {
                    "provider": "cohere",
                    "model_name": "model-b",
                    "question": "What project best represents your experience?",
                    "pairwise_utility": 0.18,
                    "selected": False,
                    "generation_latency_ms": 150,
                },
            ],
            created_at=now,
            answered_at=None,
        )
        session = SimpleNamespace(
            session_id="response-test",
            status="active",
            language="en",
            max_questions=5,
            current_turn_index=0,
            current_stage="introduction",
            allow_external_ai=False,
            research_consent=False,
            turns=[turn],
            created_at=now,
            completed_at=None,
        )

        response = _session_response(session)

        self.assertEqual(response.session_id, "response-test")
        self.assertIsNotNone(response.current_question)
        self.assertEqual(response.current_question.turn_index, 0)
        self.assertEqual(len(response.turns[0].question_candidates), 2)
        self.assertEqual(
            response.turns[0].question_candidates[0].provider_name,
            "groq",
        )
        self.assertTrue(response.turns[0].question_candidates[0].selected)
        self.assertEqual(response.turns[0].question_candidates[0].difficulty, 2)


class InterviewSessionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(
            self.engine,
            tables=[
                InterviewSession.__table__,
                InterviewTurn.__table__,
                InterviewEvidenceSummary.__table__,
            ],
        )
        self.db = sessionmaker(bind=self.engine)()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_answer_advances_and_completes_session(self) -> None:
        session = InterviewSession(
            session_id="test-session",
            status="active",
            language="vi",
            max_questions=2,
            current_turn_index=0,
            current_stage="introduction",
            allow_external_ai=False,
            research_consent=False,
            cv_text="Python backend developer with project experience.",
            jd_text="Backend role requiring Python and API development.",
            cv_profile_json=CV_PROFILE,
            job_profile_json=JOB_PROFILE,
        )
        session.turns.append(InterviewSessionService.new_turn(session, 0))
        self.db.add(session)
        self.db.commit()

        advanced = InterviewSessionService.submit_answer(
            self.db,
            session.session_id,
            "I built an API and was responsible for implementation and testing.",
        )
        self.assertEqual(advanced.current_turn_index, 1)
        self.assertEqual(len(advanced.turns), 2)

        completed = InterviewSessionService.submit_answer(
            self.db,
            session.session_id,
            "I would add measurable evidence about reliability and delivery outcomes.",
        )
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.current_stage, "completed")
        self.assertIsNotNone(completed.completed_at)
        summary = (
            self.db.query(InterviewEvidenceSummary)
            .filter(InterviewEvidenceSummary.session_id == completed.session_id)
            .one()
        )
        self.assertEqual(summary.status, "final")
        self.assertIn("overall_readiness", summary.competency_scores_json)


if __name__ == "__main__":
    unittest.main()
