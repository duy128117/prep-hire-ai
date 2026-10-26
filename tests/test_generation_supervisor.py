from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.supervise_weak_supervision_generation import (
    _last_error_text,
    _read_checkpoint,
    _retry_delay,
)


class GenerationSupervisorTests(unittest.TestCase):
    def test_reads_valid_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            checkpoint.write_text(
                '{"status": "running", "completed_sets": 44}',
                encoding="utf-8",
            )

            state = _read_checkpoint(checkpoint)

            self.assertEqual(state["completed_sets"], 44)

    def test_corrupt_checkpoint_does_not_crash_supervisor(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            checkpoint.write_text("{", encoding="utf-8")

            self.assertEqual(_read_checkpoint(checkpoint, attempts=1), {})

    def test_provider_pool_failure_uses_longer_retry_delay(self) -> None:
        state = {
            "last_error": {
                "type": "InsufficientProviderPoolError",
                "message": "Only 3 configured providers remain",
            }
        }

        self.assertEqual(_retry_delay(state, 60, 900), 900)

    def test_transient_generation_failure_uses_normal_retry_delay(self) -> None:
        state = {
            "last_error": {
                "type": "LLMProviderError",
                "message": "Failed to generate JSON",
            }
        }

        self.assertEqual(_retry_delay(state, 60, 900), 60)
        self.assertIn("Failed to generate JSON", _last_error_text(state))


if __name__ == "__main__":
    unittest.main()
