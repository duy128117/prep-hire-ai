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
from app.services.interview_service import AdaptiveQuestionEngine, InterviewSessionService
from app.services.llm_provider_service import (
    LLMJsonResult,
    LLMProviderError,
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


class InterviewSessionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(
            self.engine,
            tables=[InterviewSession.__table__, InterviewTurn.__table__],
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


if __name__ == "__main__":
    unittest.main()
