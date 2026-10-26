from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

import requests

from app.core.config import settings


@dataclass(frozen=True)
class LLMProviderSpec:
    provider: str
    model: str


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMCallRecord:
    provider: str
    model: str
    operation: str
    status: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    error_code: str | None = None
    quota_exhausted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMJsonResult:
    provider: str
    model: str
    data: dict[str, Any]
    usage: TokenUsage
    latency_ms: int


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str = "provider_error",
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.quota_exhausted = quota_exhausted


UsageSink = Callable[[LLMCallRecord], None]
_usage_sink: ContextVar[UsageSink | None] = ContextVar("llm_usage_sink", default=None)


@contextmanager
def capture_llm_usage(sink: UsageSink) -> Iterator[None]:
    token = _usage_sink.set(sink)
    try:
        yield
    finally:
        _usage_sink.reset(token)


class LLMProviderService:
    REQUEST_TIMEOUT_SECONDS = 60
    MAX_NETWORK_ATTEMPTS = 3
    MAX_CREDENTIAL_SLOTS = 13
    _key_cursors: dict[str, int] = {}

    @staticmethod
    def _unique_keys(*values: str) -> tuple[str, ...]:
        keys: list[str] = []
        for value in values:
            key = (value or "").strip()
            if key and key not in keys:
                keys.append(key)
        return tuple(keys)

    @classmethod
    def _numbered_setting_values(cls, prefix: str) -> tuple[str, ...]:
        values = [getattr(settings, prefix, "")]
        values.extend(
            getattr(settings, f"{prefix}_{slot}", "")
            for slot in range(2, cls.MAX_CREDENTIAL_SLOTS + 1)
        )
        return cls._unique_keys(*values)

    @classmethod
    def groq_api_keys(cls) -> tuple[str, ...]:
        return cls._numbered_setting_values("GROQ_API_KEY")

    @classmethod
    def inception_api_keys(cls) -> tuple[str, ...]:
        return cls._numbered_setting_values("INCEPTION_API_KEY")

    @classmethod
    def opentyphoon_api_keys(cls) -> tuple[str, ...]:
        return cls._numbered_setting_values("OPENTYPHOON_API_KEY")

    @classmethod
    def ollama_api_keys(cls) -> tuple[str, ...]:
        return cls._numbered_setting_values("OLLAMA_API_KEY")

    @classmethod
    def nvidia_api_keys(cls) -> tuple[str, ...]:
        return cls._numbered_setting_values("NVIDIA_API_KEY")

    @classmethod
    def cloudflare_credentials(cls) -> tuple[tuple[str, str], ...]:
        credentials: list[tuple[str, str]] = []
        for slot in range(1, cls.MAX_CREDENTIAL_SLOTS + 1):
            suffix = "" if slot == 1 else f"_{slot}"
            account_id = getattr(settings, f"CLOUDFLARE_ACCOUNT_ID{suffix}", "")
            api_token = getattr(settings, f"CLOUDFLARE_API_TOKEN{suffix}", "")
            credential = ((account_id or "").strip(), (api_token or "").strip())
            if all(credential) and credential not in credentials:
                credentials.append(credential)
        return tuple(credentials)

    @staticmethod
    def _key_failover_worthy(error: LLMProviderError) -> bool:
        return (
            error.quota_exhausted
            or error.error_code == "generation_failed"
            or error.status_code in {401, 402, 403, 429}
        )

    @staticmethod
    def pool_specs() -> tuple[LLMProviderSpec, ...]:
        return (
            LLMProviderSpec("groq", settings.GROQ_MODEL),
            LLMProviderSpec("cloudflare", settings.CLOUDFLARE_MODEL),
            LLMProviderSpec("inception", settings.INCEPTION_MODEL),
            LLMProviderSpec("opentyphoon", settings.OPENTYPHOON_MODEL),
            LLMProviderSpec("ollama_cloud", settings.OLLAMA_CLOUD_MODEL),
            LLMProviderSpec("nvidia_nemotron", settings.NVIDIA_NEMOTRON_MODEL),
        )

    @classmethod
    def specs(cls) -> tuple[LLMProviderSpec, ...]:
        return cls.pool_specs()

    @classmethod
    def all_specs(cls) -> tuple[LLMProviderSpec, ...]:
        return cls.pool_specs()

    @classmethod
    def resolve(cls, provider_or_model: str) -> LLMProviderSpec:
        normalized = (provider_or_model or "").strip().lower()
        for spec in cls.all_specs():
            if normalized in {spec.provider.lower(), spec.model.lower()}:
                return spec
        raise ValueError(f"Unknown configured LLM provider or model: {provider_or_model}")

    @classmethod
    def _configured(cls, spec: LLMProviderSpec) -> bool:
        return {
            "groq": bool(cls.groq_api_keys()),
            "cloudflare": bool(cls.cloudflare_credentials()),
            "inception": bool(cls.inception_api_keys()),
            "opentyphoon": bool(cls.opentyphoon_api_keys()),
            "ollama_cloud": bool(cls.ollama_api_keys()),
            "nvidia_nemotron": bool(cls.nvidia_api_keys()),
        }[spec.provider]

    @classmethod
    def is_configured(cls, spec: LLMProviderSpec) -> bool:
        return cls._configured(spec)

    @classmethod
    def require_configured(cls, specs: tuple[LLMProviderSpec, ...]) -> None:
        missing = [spec.provider for spec in specs if not cls._configured(spec)]
        if missing:
            raise ValueError(f"Missing LLM credentials for: {', '.join(missing)}")

    @classmethod
    def require_all_configured(cls) -> None:
        cls.require_configured(cls.specs())

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        normalized = (text or "").strip()
        if not normalized:
            return None
        try:
            loaded = json.loads(normalized)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", normalized)
        if not match:
            return None
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _error_detail(body: Any, status_code: int) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or error.get("type") or error.get("code")
                if detail:
                    return str(detail)[:300]
            if error:
                return str(error)[:300]
            errors = body.get("errors")
            if isinstance(errors, list) and errors:
                first = errors[0]
                if isinstance(first, dict):
                    return str(first.get("message") or first.get("code") or first)[:300]
            if body.get("message"):
                return str(body["message"])[:300]
        return f"HTTP {status_code}"

    @staticmethod
    def _is_quota_error(status_code: int | None, detail: str) -> bool:
        normalized = detail.lower()
        return status_code in {402, 429} or any(
            marker in normalized
            for marker in (
                "quota",
                "rate limit",
                "resource_exhausted",
                "credit",
                "billing",
                "insufficient balance",
                "balance is insufficient",
                "too many requests",
            )
        )

    @staticmethod
    def _is_daily_quota_error(detail: str) -> bool:
        normalized = detail.lower()
        return any(
            marker in normalized
            for marker in (
                "tokens per day",
                "requests per day",
                "daily limit",
                "daily quota",
                " tpd",
                " rpd",
            )
        )

    @staticmethod
    def _retry_delay_seconds(
        response: requests.Response,
        detail: str,
        attempt: int,
    ) -> float:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After") if isinstance(headers, Mapping) else None
        if retry_after:
            try:
                return min(120.0, max(float(attempt), float(retry_after) + 0.5))
            except (TypeError, ValueError):
                pass

        match = re.search(
            r"(?:try again in|retry after)\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|seconds?|m|min|minutes?)?",
            detail,
            flags=re.IGNORECASE,
        )
        if not match:
            return float(attempt)

        delay = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit == "ms":
            delay /= 1000
        elif unit.startswith("m"):
            delay *= 60
        return min(120.0, max(float(attempt), delay + 0.5))

    @classmethod
    def _post_json(
        cls,
        provider: str,
        url: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(1, cls.MAX_NETWORK_ATTEMPTS + 1):
            try:
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=cls.REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                if attempt < cls.MAX_NETWORK_ATTEMPTS:
                    time.sleep(attempt)
                    continue
                raise LLMProviderError(
                    provider,
                    f"{provider} network request failed after {attempt} attempts",
                    error_code="network_error",
                ) from exc

            try:
                body = response.json()
            except ValueError:
                body = {}

            if response.ok:
                if not isinstance(body, dict):
                    raise LLMProviderError(
                        provider,
                        f"{provider} returned a non-object response",
                        status_code=response.status_code,
                        error_code="invalid_response",
                    )
                return body

            detail = cls._error_detail(body, response.status_code)
            retryable_generation_error = any(
                marker in detail.lower()
                for marker in ("failed to generate json", "failed_generation")
            )
            if (
                (
                    response.status_code == 429
                    and not cls._is_daily_quota_error(detail)
                )
                or response.status_code >= 500
                or retryable_generation_error
            ) and attempt < cls.MAX_NETWORK_ATTEMPTS:
                time.sleep(cls._retry_delay_seconds(response, detail, attempt))
                continue
            raise LLMProviderError(
                provider,
                f"{provider} API error: {detail}",
                status_code=response.status_code,
                error_code=(
                    "quota_exhausted"
                    if cls._is_quota_error(response.status_code, detail)
                    else "generation_failed"
                    if retryable_generation_error
                    else "http_error"
                ),
                quota_exhausted=cls._is_quota_error(response.status_code, detail),
            )
        raise AssertionError("unreachable")

    @classmethod
    def _post_json_with_credential_pool(
        cls,
        provider: str,
        credentials: tuple[tuple[str, str], ...],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not credentials:
            raise LLMProviderError(
                provider,
                f"Missing credentials for {provider}",
                error_code="missing_credentials",
            )

        start = cls._key_cursors.get(provider, 0) % len(credentials)
        for offset in range(len(credentials)):
            credential_index = (start + offset) % len(credentials)
            url, api_key = credentials[credential_index]
            try:
                body = cls._post_json(provider, url, api_key, payload)
            except LLMProviderError as exc:
                has_next_credential = offset + 1 < len(credentials)
                if not has_next_credential or not cls._key_failover_worthy(exc):
                    raise
                cls._key_cursors[provider] = (credential_index + 1) % len(credentials)
                continue

            cls._key_cursors[provider] = credential_index
            return body

        raise AssertionError("unreachable")

    @staticmethod
    def _openai_usage(body: dict[str, Any]) -> TokenUsage:
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        return TokenUsage(input_tokens, output_tokens, total_tokens)

    @staticmethod
    def _ollama_usage(body: dict[str, Any]) -> TokenUsage:
        input_tokens = int(body.get("prompt_eval_count") or 0)
        output_tokens = int(body.get("eval_count") or 0)
        return TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens)

    @classmethod
    def _call_groq(
        cls, spec: LLMProviderSpec, prompt: str, max_output_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        payload = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_output_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        credentials = tuple(
            ("https://api.groq.com/openai/v1/chat/completions", key)
            for key in cls.groq_api_keys()
        )
        body = cls._post_json_with_credential_pool(
            spec.provider, credentials, payload
        )
        text = str(body.get("choices", [{}])[0].get("message", {}).get("content") or "")
        return text, cls._openai_usage(body)

    @classmethod
    def _call_cloudflare(
        cls, spec: LLMProviderSpec, prompt: str, max_output_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        credentials = tuple(
            (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account_id}/ai/run/{spec.model}",
                api_token,
            )
            for account_id, api_token in cls.cloudflare_credentials()
        )
        body = cls._post_json_with_credential_pool(
            spec.provider,
            credentials,
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        result = body.get("result") or {}
        text = str(result.get("response") or "") if isinstance(result, dict) else ""
        if not text and isinstance(result, dict):
            choices = result.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                if isinstance(message, dict):
                    text = str(
                        message.get("content")
                        or message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    )
        usage = cls._openai_usage(result if isinstance(result, dict) else {})
        return text, usage

    @classmethod
    def _call_inception(
        cls, spec: LLMProviderSpec, prompt: str, max_output_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        credentials = tuple(
            ("https://api.inceptionlabs.ai/v1/chat/completions", key)
            for key in cls.inception_api_keys()
        )
        body = cls._post_json_with_credential_pool(
            spec.provider,
            credentials,
            {
                "model": spec.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_output_tokens,
                "temperature": temperature,
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
            },
        )
        message = body.get("choices", [{}])[0].get("message", {})
        text = str(message.get("content") or "")
        return text, cls._openai_usage(body)

    @classmethod
    def _call_opentyphoon(
        cls, spec: LLMProviderSpec, prompt: str, max_output_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        credentials = tuple(
            ("https://api.opentyphoon.ai/v1/chat/completions", key)
            for key in cls.opentyphoon_api_keys()
        )
        body = cls._post_json_with_credential_pool(
            spec.provider,
            credentials,
            {
                "model": spec.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        message = body.get("choices", [{}])[0].get("message", {})
        text = str(message.get("content") or "")
        return text, cls._openai_usage(body)

    @classmethod
    def _call_ollama_cloud(
        cls, spec: LLMProviderSpec, prompt: str, max_output_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        credentials = tuple(
            ("https://ollama.com/api/chat", key)
            for key in cls.ollama_api_keys()
        )
        body = cls._post_json_with_credential_pool(
            spec.provider,
            credentials,
            {
                "model": spec.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": max_output_tokens,
                    "temperature": temperature,
                },
            },
        )
        message = body.get("message") or {}
        text = str(message.get("content") or "") if isinstance(message, dict) else ""
        return text, cls._ollama_usage(body)

    @classmethod
    def _call_nvidia_nemotron(
        cls, spec: LLMProviderSpec, prompt: str, max_output_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        credentials = tuple(
            ("https://integrate.api.nvidia.com/v1/chat/completions", key)
            for key in cls.nvidia_api_keys()
        )
        body = cls._post_json_with_credential_pool(
            spec.provider,
            credentials,
            {
                "model": spec.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
                "temperature": temperature,
                "reasoning_effort": "none",
                "stream": False,
            },
        )
        message = body.get("choices", [{}])[0].get("message", {})
        text = str(message.get("content") or "")
        return text, cls._openai_usage(body)

    @classmethod
    def generate_json(
        cls,
        provider_or_model: str,
        prompt: str,
        *,
        operation: str,
        max_output_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMJsonResult:
        spec = cls.resolve(provider_or_model)
        if not cls._configured(spec):
            raise LLMProviderError(
                spec.provider,
                f"Missing credentials for {spec.provider}",
                error_code="missing_credentials",
            )

        started = time.perf_counter()
        usage = TokenUsage()
        try:
            if spec.provider == "groq":
                text, usage = cls._call_groq(spec, prompt, max_output_tokens, temperature)
            elif spec.provider == "cloudflare":
                text, usage = cls._call_cloudflare(spec, prompt, max_output_tokens, temperature)
            elif spec.provider == "inception":
                text, usage = cls._call_inception(
                    spec, prompt, max_output_tokens, temperature
                )
            elif spec.provider == "opentyphoon":
                text, usage = cls._call_opentyphoon(
                    spec, prompt, max_output_tokens, temperature
                )
            elif spec.provider == "ollama_cloud":
                text, usage = cls._call_ollama_cloud(
                    spec, prompt, max_output_tokens, temperature
                )
            elif spec.provider == "nvidia_nemotron":
                text, usage = cls._call_nvidia_nemotron(
                    spec, prompt, max_output_tokens, temperature
                )
            else:
                raise LLMProviderError(
                    spec.provider,
                    f"Unsupported LLM provider: {spec.provider}",
                    error_code="unsupported_provider",
                )
            data = cls._extract_json(text)
            if not data:
                raise LLMProviderError(
                    spec.provider,
                    f"{spec.provider} returned invalid JSON",
                    error_code="invalid_json",
                )
        except Exception as exc:
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            provider_error = exc if isinstance(exc, LLMProviderError) else LLMProviderError(
                spec.provider,
                f"{spec.provider} provider failure",
                error_code="provider_error",
            )
            cls._emit(LLMCallRecord(
                provider=spec.provider,
                model=spec.model,
                operation=operation,
                status="failed",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=latency_ms,
                error_code=provider_error.error_code,
                quota_exhausted=provider_error.quota_exhausted,
            ))
            raise provider_error from exc

        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        cls._emit(LLMCallRecord(
            provider=spec.provider,
            model=spec.model,
            operation=operation,
            status="success",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
        ))
        return LLMJsonResult(spec.provider, spec.model, data, usage, latency_ms)

    @staticmethod
    def _emit(record: LLMCallRecord) -> None:
        sink = _usage_sink.get()
        if sink:
            sink(record)
