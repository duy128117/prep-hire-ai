from __future__ import annotations

import unittest

from app.services.synthetic_interview_context_service import (
    SyntheticInterviewContextService,
)


class SyntheticInterviewContextServiceTests(unittest.TestCase):
    def test_context_is_deterministic_and_clearly_labeled(self) -> None:
        first = SyntheticInterviewContextService.generate(42, "backend", "junior")
        second = SyntheticInterviewContextService.generate(42, "backend", "junior")

        self.assertEqual(first, second)
        self.assertIn("SYNTHETIC PROFILE", first.cv_text)
        self.assertIn("SYNTHETIC JOB DESCRIPTION", first.jd_text)
        self.assertIn("Backend Developer", first.jd_text)

    def test_rejects_unknown_role(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticInterviewContextService.generate(1, "unknown", "junior")


if __name__ == "__main__":
    unittest.main()
