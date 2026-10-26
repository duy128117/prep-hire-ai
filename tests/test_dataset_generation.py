from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy.exc import OperationalError

from app.services.llm_provider_service import (
    LLMCallRecord,
    LLMProviderError,
    LLMProviderService,
)
from scripts.generate_weak_supervision_dataset import (
    GenerationLedger,
    InsufficientProviderPoolError,
    answer_generation_plan,
    failover_worthy,
    generation_status,
    generation_status_with_retry,
    grouped_case_index,
    rollback_or_invalidate,
)


class BalancedDatasetGenerationTests(unittest.TestCase):
    def test_v11_answer_plan_balances_quality_and_length(self) -> None:
        plans = [answer_generation_plan(index) for index in range(60)]
        combinations = {
            (plan["target_quality_level"], plan["target_length_band"])
            for plan in plans
        }

        self.assertEqual(len(combinations), 15)
        for quality_level in range(1, 6):
            for length_band in ("short", "medium", "long"):
                count = sum(
                    plan["target_quality_level"] == quality_level
                    and plan["target_length_band"] == length_band
                    for plan in plans
                )
                self.assertEqual(count, 4)

    def test_v11_plan_includes_length_counterfactuals(self) -> None:
        plans = [answer_generation_plan(index) for index in range(15)]

        self.assertTrue(any(
            plan["counterfactual_type"] == "concise_strong" for plan in plans
        ))
        self.assertTrue(any(
            plan["counterfactual_type"] == "verbose_weak" for plan in plans
        ))

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

    @patch("scripts.generate_weak_supervision_dataset.time.sleep")
    def test_checkpoint_save_retries_temporary_windows_file_lock(self, sleep) -> None:
        with TemporaryDirectory() as directory:
            ledger = GenerationLedger(Path(directory) / "checkpoint.json", 1000, 0)
            original_replace = Path.replace
            attempts = 0

            def replace_after_two_file_locks(source, target):
                nonlocal attempts
                attempts += 1
                if attempts <= 2:
                    raise PermissionError(5, "Access is denied")
                return original_replace(source, target)

            with patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=replace_after_two_file_locks,
            ):
                ledger.save()

            self.assertEqual(attempts, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertTrue(ledger.path.exists())

    @patch("scripts.generate_weak_supervision_dataset.time.sleep")
    def test_checkpoint_save_falls_back_when_atomic_replace_stays_locked(
        self, sleep
    ) -> None:
        with TemporaryDirectory() as directory:
            ledger = GenerationLedger(Path(directory) / "checkpoint.json", 1000, 0)
            ledger.state["status"] = "fallback-test"

            with patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=PermissionError(5, "Access is denied"),
            ):
                ledger.save()

            saved = ledger._load()
            self.assertEqual(saved["status"], "fallback-test")
            self.assertEqual(sleep.call_count, 4)

    def test_successful_retry_clears_matching_transient_error(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = GenerationLedger(Path(directory) / "checkpoint.json", 1000, 0)
            failed = LLMCallRecord(
                provider="inception",
                model="mercury-2.5",
                operation="answer_generation",
                status="failed",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                latency_ms=20,
                error_code="invalid_json",
            )
            succeeded = LLMCallRecord(
                provider="inception",
                model="mercury-2.5",
                operation="answer_generation",
                status="success",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                latency_ms=20,
            )

            ledger.record(failed)
            self.assertEqual(ledger.state["last_error"]["error_code"], "invalid_json")

            ledger.record(succeeded)
            self.assertIsNone(ledger.state["last_error"])

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

    def test_provider_generation_failure_is_failover_worthy(self) -> None:
        generation_error = LLMProviderError(
            "groq",
            "Failed to generate JSON",
            status_code=400,
            error_code="generation_failed",
        )

        self.assertTrue(failover_worthy(generation_error))

    def test_invalid_question_content_is_failover_worthy(self) -> None:
        invalid_content = LLMProviderError(
            "inception",
            "did not generate a valid question",
            error_code="invalid_content",
        )

        self.assertTrue(failover_worthy(invalid_content))

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
