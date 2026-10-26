from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models.cv import CV
from app.models.job import Job
from app.services.interview_service import InterviewSessionService
from app.services.stored_interview_demo_service import StoredInterviewDemoService


class StoredInterviewDemoServiceTests(unittest.TestCase):
    def test_only_approved_https_storage_hosts_are_allowed(self) -> None:
        self.assertTrue(
            StoredInterviewDemoService._allowed_cv_url(
                "https://bucket.r2.cloudflarestorage.com/path/cv.pdf"
            )
        )
        self.assertTrue(
            StoredInterviewDemoService._allowed_cv_url(
                "https://project.supabase.co/storage/v1/object/cv.pdf"
            )
        )
        self.assertFalse(
            StoredInterviewDemoService._allowed_cv_url(
                "http://project.supabase.co/storage/v1/object/cv.pdf"
            )
        )
        self.assertFalse(
            StoredInterviewDemoService._allowed_cv_url(
                "https://example.com/cv.pdf"
            )
        )

    def test_build_job_text_combines_structured_sections(self) -> None:
        job = SimpleNamespace(
            title="Backend Developer",
            company=SimpleNamespace(name="Example Co"),
            level=SimpleNamespace(value="junior"),
            location="Da Nang",
            description_clean="Build APIs and maintain backend services.",
            description_raw="",
            role_responsibilities="Own API implementation and automated tests.",
            skills_qualifications="Python, FastAPI, PostgreSQL.",
            benefits=None,
        )

        text = StoredInterviewDemoService.build_job_text(job)

        self.assertIn("Backend Developer", text)
        self.assertIn("Python, FastAPI, PostgreSQL", text)
        self.assertIn("Da Nang", text)

    @patch.object(InterviewSessionService, "create")
    @patch.object(
        StoredInterviewDemoService,
        "load_cv_text",
        return_value=("Stored CV text with enough project evidence.", "analysis_fallback"),
    )
    def test_create_session_preserves_selected_record_ids(
        self,
        load_cv_text,
        create_interview,
    ) -> None:
        cv = SimpleNamespace(cv_id=22, user_user_id=7)
        job = SimpleNamespace(job_id=8)
        cv_query = MagicMock()
        cv_query.filter.return_value.first.return_value = cv
        job_query = MagicMock()
        job_query.filter.return_value.first.return_value = job
        db = MagicMock()
        db.query.side_effect = lambda model: cv_query if model is CV else job_query
        create_interview.return_value = SimpleNamespace(session_id="demo")

        with patch.object(
            StoredInterviewDemoService,
            "build_job_text",
            return_value="Stored job description with enough technical requirements.",
        ):
            result = StoredInterviewDemoService.create_session(
                db,
                cv_id=22,
                job_id=8,
                language="vi",
                max_questions=8,
                allow_external_ai=True,
            )

        self.assertEqual(result.session_id, "demo")
        load_cv_text.assert_called_once_with(db, cv)
        self.assertEqual(create_interview.call_args.kwargs["source_cv_id"], 22)
        self.assertEqual(create_interview.call_args.kwargs["source_job_id"], 8)
        self.assertEqual(
            create_interview.call_args.kwargs["context_source"],
            "stored_cv_job",
        )


if __name__ == "__main__":
    unittest.main()
