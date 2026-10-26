from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.llm_provider_service import (
    LLMProviderError,
    LLMProviderService,
    LLMProviderSpec,
)


class LLMProviderServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        LLMProviderService._key_cursors.clear()

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

    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY_4", "")
    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY_3", "")
    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY_2", "primary")
    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY", "primary")
    @patch.object(LLMProviderService, "MAX_CREDENTIAL_SLOTS", 4)
    def test_groq_key_pool_removes_duplicate_keys(self) -> None:
        self.assertEqual(LLMProviderService.groq_api_keys(), ("primary",))

    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY_4", "")
    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY_3", "")
    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY_2", "backup")
    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY", "primary")
    @patch.object(LLMProviderService, "MAX_CREDENTIAL_SLOTS", 4)
    @patch.object(LLMProviderService, "_post_json")
    def test_groq_switches_to_backup_key_and_keeps_using_it(self, post_json) -> None:
        exhausted = LLMProviderError(
            "groq",
            "daily quota exhausted",
            status_code=429,
            error_code="quota_exhausted",
            quota_exhausted=True,
        )
        success = {
            "choices": [{"message": {"content": '{"question": "Example?"}'}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        }
        post_json.side_effect = [exhausted, success, success]
        spec = LLMProviderSpec("groq", "openai/gpt-oss-120b")

        first_text, first_usage = LLMProviderService._call_groq(
            spec, "prompt", 128, 0.1
        )
        second_text, second_usage = LLMProviderService._call_groq(
            spec, "prompt", 128, 0.1
        )

        self.assertEqual(first_text, '{"question": "Example?"}')
        self.assertEqual(second_text, first_text)
        self.assertEqual(first_usage.total_tokens, 12)
        self.assertEqual(second_usage.total_tokens, 12)
        used_keys = [call.args[2] for call in post_json.call_args_list]
        self.assertEqual(used_keys, ["primary", "backup", "backup"])

    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY_2", "backup")
    @patch("app.services.llm_provider_service.settings.GROQ_API_KEY", "primary")
    @patch.object(LLMProviderService, "MAX_CREDENTIAL_SLOTS", 2)
    @patch.object(LLMProviderService, "_post_json")
    def test_groq_switches_key_after_generation_failure(self, post_json) -> None:
        generation_failed = LLMProviderError(
            "groq",
            "Failed to generate JSON",
            status_code=400,
            error_code="generation_failed",
        )
        success = {
            "choices": [{"message": {"content": '{"question": "Example?"}'}}],
            "usage": {"total_tokens": 12},
        }
        post_json.side_effect = [generation_failed, success]
        spec = LLMProviderSpec("groq", "openai/gpt-oss-120b")

        text, _usage = LLMProviderService._call_groq(
            spec, "prompt", 128, 0.1
        )

        self.assertEqual(text, '{"question": "Example?"}')
        used_keys = [call.args[2] for call in post_json.call_args_list]
        self.assertEqual(used_keys, ["primary", "backup"])

    @patch("app.services.llm_provider_service.time.sleep")
    @patch("app.services.llm_provider_service.requests.post")
    def test_post_json_classifies_repeated_generation_failure(self, post, sleep) -> None:
        failed = Mock(ok=False, status_code=400, headers={})
        failed.json.return_value = {
            "error": {"message": "Failed to generate JSON: failed_generation"}
        }
        post.return_value = failed

        with self.assertRaises(LLMProviderError) as raised:
            LLMProviderService._post_json(
                "groq",
                "https://example.test",
                "secret",
                {"model": "example"},
            )

        self.assertEqual(raised.exception.error_code, "generation_failed")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_numbered_key_pool_supports_thirteen_slots(self) -> None:
        values = {"GROQ_API_KEY": "key-1"}
        values.update({f"GROQ_API_KEY_{slot}": "" for slot in range(2, 14)})
        values["GROQ_API_KEY_13"] = "key-13"

        with patch(
            "app.services.llm_provider_service.settings",
            SimpleNamespace(**values),
        ):
            self.assertEqual(
                LLMProviderService.groq_api_keys(),
                ("key-1", "key-13"),
            )

    def test_provider_pool_has_no_core_or_backup_classification(self) -> None:
        self.assertEqual(LLMProviderService.specs(), LLMProviderService.pool_specs())
        self.assertEqual(LLMProviderService.all_specs(), LLMProviderService.pool_specs())

    @patch.object(
        LLMProviderService,
        "inception_api_keys",
        return_value=("test-inception-key",),
    )
    @patch.object(LLMProviderService, "_post_json")
    def test_inception_uses_mercury_chat_completions(
        self, post_json, _api_keys
    ) -> None:
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
        self.assertEqual(args[3]["response_format"], {"type": "json_object"})

    @patch.object(
        LLMProviderService,
        "opentyphoon_api_keys",
        return_value=("test-opentyphoon-key",),
    )
    @patch.object(LLMProviderService, "_post_json")
    def test_opentyphoon_uses_openai_compatible_chat(
        self, post_json, _api_keys
    ) -> None:
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

    @patch.object(
        LLMProviderService,
        "ollama_api_keys",
        return_value=("test-ollama-key",),
    )
    @patch.object(LLMProviderService, "_post_json")
    def test_ollama_cloud_uses_native_chat_as_pool_worker(
        self, post_json, _api_keys
    ) -> None:
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

    @patch.object(
        LLMProviderService,
        "nvidia_api_keys",
        return_value=("test-nvidia-key",),
    )
    @patch.object(LLMProviderService, "_post_json")
    def test_nvidia_nemotron_uses_openai_compatible_chat(
        self, post_json, _api_keys
    ) -> None:
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

    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_API_TOKEN_4", "")
    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_ACCOUNT_ID_4", "")
    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_API_TOKEN_3", "")
    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_ACCOUNT_ID_3", "")
    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_API_TOKEN_2", "token-2")
    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_ACCOUNT_ID_2", "account-2")
    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_API_TOKEN", "token-1")
    @patch("app.services.llm_provider_service.settings.CLOUDFLARE_ACCOUNT_ID", "account-1")
    @patch.object(LLMProviderService, "MAX_CREDENTIAL_SLOTS", 4)
    def test_cloudflare_credentials_keep_account_and_token_pairs(self) -> None:
        self.assertEqual(
            LLMProviderService.cloudflare_credentials(),
            (("account-1", "token-1"), ("account-2", "token-2")),
        )

    def test_cloudflare_credentials_support_thirteen_slots(self) -> None:
        values: dict[str, str] = {}
        for slot in range(1, 14):
            suffix = "" if slot == 1 else f"_{slot}"
            values[f"CLOUDFLARE_ACCOUNT_ID{suffix}"] = ""
            values[f"CLOUDFLARE_API_TOKEN{suffix}"] = ""
        values["CLOUDFLARE_ACCOUNT_ID_13"] = "account-13"
        values["CLOUDFLARE_API_TOKEN_13"] = "token-13"

        with patch(
            "app.services.llm_provider_service.settings",
            SimpleNamespace(**values),
        ):
            self.assertEqual(
                LLMProviderService.cloudflare_credentials(),
                (("account-13", "token-13"),),
            )

    @patch.object(LLMProviderService, "_post_json")
    def test_generic_pool_switches_on_authentication_error(self, post_json) -> None:
        unauthorized = LLMProviderError(
            "nvidia_nemotron",
            "invalid API key",
            status_code=401,
            error_code="http_error",
        )
        success = {"ok": True}
        credentials = (
            ("https://example.test", "primary"),
            ("https://example.test", "backup"),
        )
        post_json.side_effect = [unauthorized, success, success]

        first = LLMProviderService._post_json_with_credential_pool(
            "nvidia_nemotron", credentials, {"model": "example"}
        )
        second = LLMProviderService._post_json_with_credential_pool(
            "nvidia_nemotron", credentials, {"model": "example"}
        )

        self.assertEqual(first, success)
        self.assertEqual(second, success)
        used_keys = [call.args[2] for call in post_json.call_args_list]
        self.assertEqual(used_keys, ["primary", "backup", "backup"])

if __name__ == "__main__":
    unittest.main()
