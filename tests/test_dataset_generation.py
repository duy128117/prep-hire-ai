from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy.exc import OperationalError

from app.services.llm_provider_service import LLMProviderError, LLMProviderService
from scripts.generate_weak_supervision_dataset import (
    GenerationLedger,
    InsufficientProviderPoolError,
    failover_worthy,
    generation_status,
    generation_status_with_retry,
    grouped_case_index,
    rollback_or_invalidate,
)


class BalancedDatasetGenerationTests(unittest.TestCase):
    @patch("scripts.generate_weak_supervision_dataset._complete_outputs", return_value=[])
    def test_generation_status_returns_structured_report(self, _complete_outputs) -> None:
        db = unittest.mock.Mock()
        db.query.return_value.scalar.return_value = 0
        db.query.return_value.filter.return_value.scalar.return_value = 0

        status = generation_status(db)

        self.assertIsInstance(status, dict)
        self.assertEqual(status["complete_sets"], 0)
        self.assertEqual(len(status["logical_slot_counts"]), 4)
        self.assertEqual(len(status["provider_pool"]), 6)
        self.assertEqual(status["required_active_providers"], 4)

    @patch("scripts.generate_weak_supervision_dataset.time.sleep")
    @patch("scripts.generate_weak_supervision_dataset.rollback_or_invalidate")
    @patch("scripts.generate_weak_supervision_dataset.generation_status")
    def test_status_retries_transient_database_error(
        self,
        generation_status,
        rollback_or_invalidate,
        sleep,
    ) -> None:
        generation_status.side_effect = [
            OperationalError("SELECT", {}, RuntimeError("temporary DNS failure")),
            {"complete_sets": 56},
        ]
        db = unittest.mock.Mock()

        result = generation_status_with_retry(db)

        self.assertEqual(result, {"complete_sets": 56})
        rollback_or_invalidate.assert_called_once_with(db)
        sleep.assert_called_once_with(1)

    def test_failed_rollback_invalidates_stale_database_connection(self) -> None:
        db = unittest.mock.Mock()
        db.rollback.side_effect = RuntimeError("connection closed")

        error = rollback_or_invalidate(db)

        self.assertEqual(error, "connection closed")
        db.invalidate.assert_called_once_with()

    def test_all_generators_in_a_round_share_one_context(self) -> None:
        start_set = 36

        first_round_case = grouped_case_index(36, 4, 240, start_set)
        second_round_case = grouped_case_index(40, 4, 240, start_set)

        self.assertEqual(first_round_case, 36)
        self.assertEqual(second_round_case, 37)

    def test_grouped_generation_rejects_partial_round(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete provider round"):
            grouped_case_index(38, 4, 240, 36)

    def test_failed_worker_is_replaced_from_equal_provider_pool(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = GenerationLedger(Path(directory) / "checkpoint.json", 1000, 0)
            with patch.object(LLMProviderService, "is_configured", return_value=True):
                replacement = ledger.failover_provider("groq", "daily quota")

            self.assertIsNotNone(replacement)
            self.assertEqual(replacement.provider, "ollama_cloud")
            self.assertEqual(
                [spec.provider for spec in ledger.active_roster()],
                ["ollama_cloud", "cloudflare", "inception", "opentyphoon"],
            )
            self.assertEqual(len(ledger.active_roster()), 4)

    def test_generation_stops_only_when_pool_has_fewer_than_four_providers(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = GenerationLedger(Path(directory) / "checkpoint.json", 1000, 0)
            with patch.object(LLMProviderService, "is_configured", return_value=True):
                first = ledger.failover_provider("groq", "daily quota")
                second = ledger.failover_provider("cloudflare", "daily quota")
                third = ledger.failover_provider("inception", "daily quota")

                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                self.assertEqual(second.provider, "nvidia_nemotron")
                self.assertIsNone(third)
                self.assertEqual(ledger.pool_status()["available_count"], 3)
                with self.assertRaises(InsufficientProviderPoolError):
                    ledger.active_roster()

    def test_unconfigured_provider_is_not_selected_from_pool(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = GenerationLedger(Path(directory) / "checkpoint.json", 1000, 0)
            with patch.object(LLMProviderService, "is_configured", return_value=False):
                replacement = ledger.failover_provider("opentyphoon", "HTTP 402")

            self.assertIsNone(replacement)

    def test_round_robin_rotates_all_pool_providers(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.object(LLMProviderService, "is_configured", return_value=True):
                ledger = GenerationLedger(Path(directory) / "checkpoint.json", 1000, 0)

                first = [spec.provider for spec in ledger.active_roster()]
                ledger.rotate_active_roster()
                second = [spec.provider for spec in ledger.active_roster()]
                ledger.rotate_active_roster()
                third = [spec.provider for spec in ledger.active_roster()]

                self.assertEqual(
                    first,
                    ["groq", "cloudflare", "inception", "opentyphoon"],
                )
                self.assertEqual(
                    second,
                    ["cloudflare", "inception", "opentyphoon", "ollama_cloud"],
                )
                self.assertEqual(
                    third,
                    [
                        "inception",
                        "opentyphoon",
                        "ollama_cloud",
                        "nvidia_nemotron",
                    ],
                )

    def test_quota_and_repeated_invalid_json_are_failover_worthy(self) -> None:
        quota_error = LLMProviderError(
            "inception",
            "HTTP 402",
            status_code=402,
            quota_exhausted=True,
        )
        json_error = LLMProviderError(
            "groq",
            "invalid JSON",
            error_code="invalid_json",
        )

        self.assertTrue(failover_worthy(quota_error))
        self.assertTrue(failover_worthy(json_error))

    def test_rate_limit_replaces_provider_after_internal_retries_fail(self) -> None:
        rate_limit = LLMProviderError(
            "groq",
            "Rate limit reached; retry shortly",
            status_code=429,
            error_code="quota_exhausted",
            quota_exhausted=True,
        )

        self.assertTrue(failover_worthy(rate_limit))

    def test_daily_token_limit_is_failover_worthy(self) -> None:
        daily_limit = LLMProviderError(
            "groq",
            "Rate limit reached on tokens per day (TPD)",
            status_code=429,
            error_code="quota_exhausted",
            quota_exhausted=True,
        )

        self.assertTrue(failover_worthy(daily_limit))

if __name__ == "__main__":
    unittest.main()
