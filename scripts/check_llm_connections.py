from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.llm_provider_service import LLMProviderService


PROMPT = "Reply with exactly: OK"
TIMEOUT_SECONDS = 30


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("type") or error.get("code")
            if detail:
                return f"HTTP {response.status_code}: {str(detail)[:200]}"
        if error:
            return f"HTTP {response.status_code}: {str(error)[:200]}"
        if body.get("message"):
            return f"HTTP {response.status_code}: {str(body['message'])[:200]}"
    return f"HTTP {response.status_code}"


def _post_chat(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    text_reader: Callable[[dict[str, Any]], str],
) -> None:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(_error_message(response))

    text = text_reader(response.json()).strip()
    if not text:
        raise RuntimeError("API returned an empty response")


def _check_credential_pool(
    *,
    provider: str,
    credentials: tuple[tuple[str, str], ...],
    payload: dict[str, Any],
    text_reader: Callable[[dict[str, Any]], str],
) -> None:
    if not credentials:
        raise RuntimeError(f"No {provider} credential is configured")

    last_error: Exception | None = None
    for url, api_key in credentials:
        try:
            _post_chat(
                url=url,
                api_key=api_key,
                payload=payload,
                text_reader=text_reader,
            )
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"All configured {provider} credentials failed: {last_error}")


def _openai_text(body: dict[str, Any]) -> str:
    return str(body["choices"][0]["message"]["content"])


def _cloudflare_text(body: dict[str, Any]) -> str:
    result = body.get("result", body)
    if not isinstance(result, dict):
        return ""
    if result.get("response"):
        return str(result["response"])
    choices = result.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
    return ""


def _ollama_text(body: dict[str, Any]) -> str:
    message = body.get("message") or {}
    return str(message.get("content") or "") if isinstance(message, dict) else ""


def check_groq() -> None:
    credentials = tuple(
        ("https://api.groq.com/openai/v1/chat/completions", key)
        for key in LLMProviderService.groq_api_keys()
    )
    _check_credential_pool(
        provider="Groq",
        credentials=credentials,
        payload={
            "model": settings.GROQ_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_completion_tokens": 128,
            "temperature": 0,
        },
        text_reader=_openai_text,
    )


def check_cloudflare() -> None:
    credentials = tuple(
        (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/{settings.CLOUDFLARE_MODEL}",
            api_token,
        )
        for account_id, api_token in LLMProviderService.cloudflare_credentials()
    )
    _check_credential_pool(
        provider="Cloudflare",
        credentials=credentials,
        payload={
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 256,
            "temperature": 0,
        },
        text_reader=_cloudflare_text,
    )


def check_inception() -> None:
    credentials = tuple(
        ("https://api.inceptionlabs.ai/v1/chat/completions", key)
        for key in LLMProviderService.inception_api_keys()
    )
    _check_credential_pool(
        provider="Inception",
        credentials=credentials,
        payload={
            "model": settings.INCEPTION_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_completion_tokens": 512,
            "temperature": 0,
            "reasoning_effort": "low",
        },
        text_reader=_openai_text,
    )


def check_opentyphoon() -> None:
    credentials = tuple(
        ("https://api.opentyphoon.ai/v1/chat/completions", key)
        for key in LLMProviderService.opentyphoon_api_keys()
    )
    _check_credential_pool(
        provider="OpenTyphoon",
        credentials=credentials,
        payload={
            "model": settings.OPENTYPHOON_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        text_reader=_openai_text,
    )


def check_ollama_cloud() -> None:
    credentials = tuple(
        ("https://ollama.com/api/chat", key)
        for key in LLMProviderService.ollama_api_keys()
    )
    _check_credential_pool(
        provider="Ollama Cloud",
        credentials=credentials,
        payload={
            "model": settings.OLLAMA_CLOUD_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "think": False,
            "options": {"num_predict": 16, "temperature": 0},
        },
        text_reader=_ollama_text,
    )


def check_nvidia_nemotron() -> None:
    credentials = tuple(
        ("https://integrate.api.nvidia.com/v1/chat/completions", key)
        for key in LLMProviderService.nvidia_api_keys()
    )
    _check_credential_pool(
        provider="NVIDIA Nemotron",
        credentials=credentials,
        payload={
            "model": settings.NVIDIA_NEMOTRON_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 16,
            "temperature": 0,
            "reasoning_effort": "none",
            "stream": False,
        },
        text_reader=_openai_text,
    )


def main() -> int:
    providers = [
        (
            "groq",
            settings.GROQ_MODEL,
            bool(LLMProviderService.groq_api_keys()),
            check_groq,
        ),
        (
            "cloudflare",
            settings.CLOUDFLARE_MODEL,
            bool(LLMProviderService.cloudflare_credentials()),
            check_cloudflare,
        ),
        (
            "inception",
            settings.INCEPTION_MODEL,
            bool(LLMProviderService.inception_api_keys()),
            check_inception,
        ),
        (
            "opentyphoon",
            settings.OPENTYPHOON_MODEL,
            bool(LLMProviderService.opentyphoon_api_keys()),
            check_opentyphoon,
        ),
        (
            "ollama_cloud",
            settings.OLLAMA_CLOUD_MODEL,
            bool(LLMProviderService.ollama_api_keys()),
            check_ollama_cloud,
        ),
        (
            "nvidia_nemotron",
            settings.NVIDIA_NEMOTRON_MODEL,
            bool(LLMProviderService.nvidia_api_keys()),
            check_nvidia_nemotron,
        ),
    ]
    results: list[dict[str, Any]] = []

    for provider, model, is_configured, check in providers:
        started = time.perf_counter()
        result: dict[str, Any] = {"provider": provider, "model": model}
        if not is_configured:
            result.update(status="missing_key", latency_ms=0)
        else:
            try:
                check()
                result["status"] = "connected"
            except Exception as exc:
                result.update(status="failed", error=str(exc)[:240])
            result["latency_ms"] = round((time.perf_counter() - started) * 1000)
        results.append(result)

    print(json.dumps(results, ensure_ascii=True, indent=2))
    return 0 if all(item["status"] == "connected" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
