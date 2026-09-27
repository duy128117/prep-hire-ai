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
    _post_chat(
        url="https://api.groq.com/openai/v1/chat/completions",
        api_key=settings.GROQ_API_KEY,
        payload={
            "model": settings.GROQ_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_completion_tokens": 128,
            "temperature": 0,
        },
        text_reader=_openai_text,
    )


def check_cloudflare() -> None:
    _post_chat(
        url=(
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{settings.CLOUDFLARE_ACCOUNT_ID}/ai/run/{settings.CLOUDFLARE_MODEL}"
        ),
        api_key=settings.CLOUDFLARE_API_TOKEN,
        payload={
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 256,
            "temperature": 0,
        },
        text_reader=_cloudflare_text,
    )


def check_inception() -> None:
    _post_chat(
        url="https://api.inceptionlabs.ai/v1/chat/completions",
        api_key=settings.INCEPTION_API_KEY,
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
    _post_chat(
        url="https://api.opentyphoon.ai/v1/chat/completions",
        api_key=settings.OPENTYPHOON_API_KEY,
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
    _post_chat(
        url="https://ollama.com/api/chat",
        api_key=settings.OLLAMA_API_KEY,
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
    _post_chat(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        api_key=settings.NVIDIA_API_KEY,
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
        ("groq", settings.GROQ_MODEL, bool(settings.GROQ_API_KEY), check_groq),
        (
            "cloudflare",
            settings.CLOUDFLARE_MODEL,
            bool(settings.CLOUDFLARE_ACCOUNT_ID and settings.CLOUDFLARE_API_TOKEN),
            check_cloudflare,
        ),
        (
            "inception",
            settings.INCEPTION_MODEL,
            bool(settings.INCEPTION_API_KEY),
            check_inception,
        ),
        (
            "opentyphoon",
            settings.OPENTYPHOON_MODEL,
            bool(settings.OPENTYPHOON_API_KEY),
            check_opentyphoon,
        ),
        (
            "ollama_cloud",
            settings.OLLAMA_CLOUD_MODEL,
            bool(settings.OLLAMA_API_KEY),
            check_ollama_cloud,
        ),
        (
            "nvidia_nemotron",
            settings.NVIDIA_NEMOTRON_MODEL,
            bool(settings.NVIDIA_API_KEY),
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
