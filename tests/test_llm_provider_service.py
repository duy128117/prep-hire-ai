from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app.services.llm_provider_service import (
    LLMProviderError,
    LLMProviderService,
    LLMProviderSpec,
)


class LLMProviderServiceTests(unittest.TestCase):
    @patch("app.services.llm_provider_service.time.sleep")
    @patch("app.services.llm_provider_service.requests.post")
    def test_post_json_retries_temporary_rate_limit(self, post, sleep) -> None:
        rate_limited = Mock(ok=False, status_code=429)
        rate_limited.json.return_value = {"error": {"message": "Rate limit reached"}}
        success = Mock(ok=True, status_code=200)
        success.json.return_value = {"ok": True}
        post.side_effect = [rate_limited, success]

        result = LLMProviderService._post_json(
            "groq",
            "https://example.test",
            "secret",
            {"model": "example"},
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("app.services.llm_provider_service.time.sleep")
    @patch("app.services.llm_provider_service.requests.post")
    def test_post_json_honors_provider_retry_delay(self, post, sleep) -> None:
        rate_limited = Mock(ok=False, status_code=429, headers={})
        rate_limited.json.return_value = {
            "error": {"message": "Rate limit reached. Please try again in 6.75s."}
        }
        success = Mock(ok=True, status_code=200)
        success.json.return_value = {"ok": True}
        post.side_effect = [rate_limited, success]

        result = LLMProviderService._post_json(
            "groq",
            "https://example.test",
            "secret",
            {"model": "example"},
        )

        self.assertEqual(result, {"ok": True})
        sleep.assert_called_once_with(7.25)

    @patch("app.services.llm_provider_service.time.sleep")
    @patch("app.services.llm_provider_service.requests.post")
    def test_post_json_does_not_retry_daily_token_quota(self, post, sleep) -> None:
        exhausted = Mock(ok=False, status_code=429, headers={})
        exhausted.json.return_value = {
            "error": {"message": "Tokens per day (TPD): Limit 200000"}
        }
        post.return_value = exhausted

        with self.assertRaises(LLMProviderError) as raised:
            LLMProviderService._post_json(
                "groq",
                "https://example.test",
                "secret",
                {"model": "example"},
            )

        self.assertTrue(raised.exception.quota_exhausted)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_insufficient_balance_is_treated_as_quota_exhaustion(self) -> None:
        self.assertTrue(
            LLMProviderService._is_quota_error(
                400,
                "Sorry, your account balance is insufficient",
            )
        )

    def test_research_provider_pool_contains_all_equal_workers(self) -> None:
        providers = [spec.provider for spec in LLMProviderService.specs()]

        self.assertEqual(
            providers,
            [
                "groq",
                "cloudflare",
                "inception",
                "opentyphoon",
                "ollama_cloud",
                "nvidia_nemotron",
            ],
        )
        self.assertNotIn("gemini", providers)
        self.assertNotIn("huggingface", providers)
        self.assertNotIn("siliconflow", providers)

    def test_provider_pool_has_no_core_or_backup_classification(self) -> None:
        self.assertEqual(LLMProviderService.specs(), LLMProviderService.pool_specs())
        self.assertEqual(LLMProviderService.all_specs(), LLMProviderService.pool_specs())

    @patch.object(LLMProviderService, "_post_json")
    def test_inception_uses_mercury_chat_completions(self, post_json) -> None:
        post_json.return_value = {
            "choices": [{"message": {"content": '{"question": "Example?"}'}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        }
        spec = LLMProviderSpec("inception", "mercury-2.5")

        text, usage = LLMProviderService._call_inception(spec, "prompt", 128, 0.1)

        self.assertEqual(text, '{"question": "Example?"}')
        self.assertEqual(usage.total_tokens, 12)
        args = post_json.call_args.args
        self.assertEqual(args[0], "inception")
        self.assertEqual(args[1], "https://api.inceptionlabs.ai/v1/chat/completions")
        self.assertEqual(args[3]["model"], "mercury-2.5")
        self.assertEqual(args[3]["reasoning_effort"], "low")

    @patch.object(LLMProviderService, "_post_json")
    def test_opentyphoon_uses_openai_compatible_chat(self, post_json) -> None:
        post_json.return_value = {
            "choices": [{"message": {"content": '{"question": "Example?"}'}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
        spec = LLMProviderSpec("opentyphoon", "typhoon-v2.5-30b-a3b-instruct")

        text, usage = LLMProviderService._call_opentyphoon(spec, "prompt", 128, 0.1)

        self.assertEqual(text, '{"question": "Example?"}')
        self.assertEqual(usage.total_tokens, 12)
        args = post_json.call_args.args
        self.assertEqual(args[0], "opentyphoon")
        self.assertEqual(args[1], "https://api.opentyphoon.ai/v1/chat/completions")
        self.assertEqual(args[3]["model"], "typhoon-v2.5-30b-a3b-instruct")
        self.assertFalse(args[3]["stream"])

    @patch.object(LLMProviderService, "_post_json")
    def test_ollama_cloud_uses_native_chat_as_pool_worker(self, post_json) -> None:
        post_json.return_value = {
            "message": {"role": "assistant", "content": '{"question": "Example?"}'},
            "prompt_eval_count": 11,
            "eval_count": 6,
        }
        spec = LLMProviderSpec("ollama_cloud", "gemma4:31b")

        text, usage = LLMProviderService._call_ollama_cloud(
            spec,
            "prompt",
            128,
            0.1,
        )

        self.assertEqual(text, '{"question": "Example?"}')
        self.assertEqual(usage.total_tokens, 17)
        self.assertEqual(post_json.call_args.args[0], "ollama_cloud")

    @patch.object(LLMProviderService, "_post_json")
    def test_nvidia_nemotron_uses_openai_compatible_chat(self, post_json) -> None:
        post_json.return_value = {
            "choices": [{"message": {"content": '{"question": "Example?"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        spec = LLMProviderSpec(
            "nvidia_nemotron",
            "nvidia/nemotron-3-ultra-550b-a55b",
        )

        text, usage = LLMProviderService._call_nvidia_nemotron(
            spec,
            "prompt",
            128,
            0.1,
        )

        self.assertEqual(text, '{"question": "Example?"}')
        self.assertEqual(usage.total_tokens, 15)
        args = post_json.call_args.args
        self.assertEqual(args[0], "nvidia_nemotron")
        self.assertEqual(args[1], "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertEqual(args[3]["reasoning_effort"], "none")
        self.assertFalse(args[3]["stream"])

if __name__ == "__main__":
    unittest.main()
