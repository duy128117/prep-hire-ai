from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings


MAX_CREDENTIAL_SLOTS = 13
PROMPT = "Reply briefly with OK."
NETWORK_ATTEMPTS = 3
REQUEST_TIMEOUT = (10, 30)
TextReader = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class CredentialSpec:
    provider: str
    slot: int
    env_names: tuple[str, ...]
    identity: tuple[str, ...]
    url: str
    api_key: str
    payload: dict[str, Any]
    text_reader: TextReader


def _suffix(slot: int) -> str:
    return "" if slot == 1 else f"_{slot}"


def _setting(name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


def _openai_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return ""
    return str(
        message.get("content")
        or message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )


def _cloudflare_text(body: dict[str, Any]) -> str:
    result = body.get("result", body)
    if not isinstance(result, dict):
        return ""
    if result.get("response"):
        return str(result["response"])
    return _openai_text(result)


def _ollama_text(body: dict[str, Any]) -> str:
    message = body.get("message") or {}
    return str(message.get("content") or "") if isinstance(message, dict) else ""


def _api_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if not isinstance(body, dict):
        return f"HTTP {response.status_code}"
    error = body.get("error")
    if isinstance(error, dict):
        detail = error.get("message") or error.get("type") or error.get("code")
        if detail:
            return str(detail)[:240]
    if error:
        return str(error)[:240]
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            return str(first.get("message") or first.get("code") or first)[:240]
    return str(body.get("message") or f"HTTP {response.status_code}")[:240]


def _classify_http_error(status_code: int, detail: str) -> str:
    normalized = detail.lower()
    auth_markers = (
        "invalid api key",
        "invalid token",
        "invalid access token",
        "authentication error",
        "authentication failed",
        "unauthorized",
        "incorrect api key",
        "bad credentials",
    )
    quota_markers = (
        "quota",
        "rate limit",
        "too many requests",
        "insufficient balance",
        "billing",
        "credit",
        "resource_exhausted",
        "tokens per day",
        "requests per day",
    )
    model_markers = (
        "model not found",
        "model does not exist",
        "unknown model",
        "unsupported model",
    )
    if status_code == 401 or any(marker in normalized for marker in auth_markers):
        return "invalid_auth"
    if status_code in {402, 429} or any(marker in normalized for marker in quota_markers):
        return "quota_exhausted"
    if any(marker in normalized for marker in model_markers):
        return "model_error"
    if status_code == 403:
        return "permission_denied"
    if status_code >= 500:
        return "provider_unavailable"
    return "request_failed"


def _openai_specs(
    *,
    provider: str,
    key_prefix: str,
    url: str,
    model: str,
    token_field: str,
    max_tokens: int,
    extra: dict[str, Any] | None = None,
) -> list[CredentialSpec]:
    specs: list[CredentialSpec] = []
    for slot in range(1, MAX_CREDENTIAL_SLOTS + 1):
        env_name = f"{key_prefix}{_suffix(slot)}"
        api_key = _setting(env_name)
        if not api_key:
            continue
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            token_field: max_tokens,
            "temperature": 0,
        }
        payload.update(extra or {})
        specs.append(
            CredentialSpec(
                provider=provider,
                slot=slot,
                env_names=(env_name,),
                identity=(api_key,),
                url=url,
                api_key=api_key,
                payload=payload,
                text_reader=_openai_text,
            )
        )
    return specs


def _credential_specs() -> list[CredentialSpec]:
    specs = _openai_specs(
        provider="groq",
        key_prefix="GROQ_API_KEY",
        url="https://api.groq.com/openai/v1/chat/completions",
        model=settings.GROQ_MODEL,
        token_field="max_completion_tokens",
        max_tokens=128,
    )
    specs.extend(
        _openai_specs(
            provider="inception",
            key_prefix="INCEPTION_API_KEY",
            url="https://api.inceptionlabs.ai/v1/chat/completions",
            model=settings.INCEPTION_MODEL,
            token_field="max_completion_tokens",
            max_tokens=512,
            extra={"reasoning_effort": "low"},
        )
    )
    specs.extend(
        _openai_specs(
            provider="opentyphoon",
            key_prefix="OPENTYPHOON_API_KEY",
            url="https://api.opentyphoon.ai/v1/chat/completions",
            model=settings.OPENTYPHOON_MODEL,
            token_field="max_tokens",
            max_tokens=16,
            extra={"stream": False},
        )
    )
    specs.extend(
        _openai_specs(
            provider="nvidia_nemotron",
            key_prefix="NVIDIA_API_KEY",
            url="https://integrate.api.nvidia.com/v1/chat/completions",
            model=settings.NVIDIA_NEMOTRON_MODEL,
            token_field="max_tokens",
            max_tokens=16,
            extra={"reasoning_effort": "none", "stream": False},
        )
    )

    for slot in range(1, MAX_CREDENTIAL_SLOTS + 1):
        suffix = _suffix(slot)
        account_name = f"CLOUDFLARE_ACCOUNT_ID{suffix}"
        token_name = f"CLOUDFLARE_API_TOKEN{suffix}"
        account_id = _setting(account_name)
        api_token = _setting(token_name)
        if not account_id and not api_token:
            continue
        if not account_id or not api_token:
            specs.append(
                CredentialSpec(
                    provider="cloudflare",
                    slot=slot,
                    env_names=(account_name, token_name),
                    identity=(account_id, api_token),
                    url="",
                    api_key=api_token,
                    payload={},
                    text_reader=_cloudflare_text,
                )
            )
            continue
        specs.append(
            CredentialSpec(
                provider="cloudflare",
                slot=slot,
                env_names=(account_name, token_name),
                identity=(account_id, api_token),
                url=(
                    "https://api.cloudflare.com/client/v4/accounts/"
                    f"{account_id}/ai/run/{settings.CLOUDFLARE_MODEL}"
                ),
                api_key=api_token,
                payload={
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": 32,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                text_reader=_cloudflare_text,
            )
        )

    for slot in range(1, MAX_CREDENTIAL_SLOTS + 1):
        env_name = f"OLLAMA_API_KEY{_suffix(slot)}"
        api_key = _setting(env_name)
        if not api_key:
            continue
        specs.append(
            CredentialSpec(
                provider="ollama_cloud",
                slot=slot,
                env_names=(env_name,),
                identity=(api_key,),
                url="https://ollama.com/api/chat",
                api_key=api_key,
                payload={
                    "model": settings.OLLAMA_CLOUD_MODEL,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": 16, "temperature": 0},
                },
                text_reader=_ollama_text,
            )
        )
    return specs


def _check_one(spec: CredentialSpec) -> dict[str, Any]:
    result: dict[str, Any] = {"provider": spec.provider, "slot": spec.slot}
    if not spec.url or not spec.api_key:
        result.update(status="incomplete_pair", latency_ms=0)
        return result

    started = time.perf_counter()
    for attempt in range(1, NETWORK_ATTEMPTS + 1):
        try:
            response = requests.post(
                spec.url,
                headers={
                    "Authorization": f"Bearer {spec.api_key}",
                    "Content-Type": "application/json",
                },
                json=spec.payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            if attempt < NETWORK_ATTEMPTS:
                time.sleep(attempt)
                continue
            result.update(status="network_error", attempts=attempt)
            break

        if response.ok:
            try:
                body = response.json()
            except ValueError:
                result.update(
                    status="authenticated_invalid_response",
                    http_status=response.status_code,
                )
                break
            if not isinstance(body, dict) or not spec.text_reader(body).strip():
                result.update(
                    status="authenticated_empty_response",
                    http_status=response.status_code,
                )
            else:
                result.update(status="connected", http_status=response.status_code)
            break

        detail = _api_error(response)
        status = _classify_http_error(response.status_code, detail)
        if status == "provider_unavailable" and attempt < NETWORK_ATTEMPTS:
            time.sleep(attempt)
            continue
        result.update(status=status, http_status=response.status_code, error=detail)
        break

    result["latency_ms"] = round((time.perf_counter() - started) * 1000)
    return result


def _check_provider(specs: list[CredentialSpec]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: dict[tuple[str, ...], int] = {}
    for spec in sorted(specs, key=lambda item: item.slot):
        if spec.identity in seen:
            results.append(
                {
                    "provider": spec.provider,
                    "slot": spec.slot,
                    "status": "duplicate",
                    "duplicate_of_slot": seen[spec.identity],
                    "latency_ms": 0,
                }
            )
            continue
        seen[spec.identity] = spec.slot
        results.append(_check_one(spec))
    return results


def _disable_env_names(env_names: set[str]) -> list[str]:
    if not env_names:
        return []
    env_path = PROJECT_ROOT / ".env"
    original = env_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    disabled: list[str] = []
    patterns = {
        name: re.compile(rf"^(\s*{re.escape(name)}\s*=).*(\r?\n)?$")
        for name in env_names
    }
    for index, line in enumerate(lines):
        for name, pattern in patterns.items():
            match = pattern.match(line)
            if not match:
                continue
            newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            lines[index] = f'{match.group(1)}""{newline}'
            disabled.append(name)
            break
    updated = "".join(lines)
    temp_path = env_path.with_suffix(".env.audit.tmp")
    temp_path.write_text(updated, encoding="utf-8")
    os.replace(temp_path, env_path)
    return sorted(disabled)


def _summary(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        provider = str(result["provider"])
        status = str(result["status"])
        provider_summary = summary.setdefault(provider, {"total": 0})
        provider_summary["total"] += 1
        provider_summary[status] = provider_summary.get(status, 0) + 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit every configured LLM credential slot.")
    parser.add_argument(
        "--provider",
        action="append",
        choices=(
            "groq",
            "cloudflare",
            "inception",
            "opentyphoon",
            "ollama_cloud",
            "nvidia_nemotron",
        ),
        help="Audit only this provider; repeat the option for multiple providers.",
    )
    parser.add_argument(
        "--slots",
        help="Comma-separated slot numbers to audit, for example 3,7,8.",
    )
    parser.add_argument(
        "--disable-invalid",
        action="store_true",
        help="Blank only credentials confirmed invalid by an authentication response.",
    )
    args = parser.parse_args()

    specs = _credential_specs()
    selected_providers = set(args.provider or [])
    if selected_providers:
        specs = [spec for spec in specs if spec.provider in selected_providers]
    if args.slots:
        selected_slots = {
            int(value.strip())
            for value in args.slots.split(",")
            if value.strip()
        }
        invalid_slots = [
            slot for slot in selected_slots if slot < 1 or slot > MAX_CREDENTIAL_SLOTS
        ]
        if invalid_slots:
            parser.error(f"Slots must be between 1 and {MAX_CREDENTIAL_SLOTS}")
        specs = [spec for spec in specs if spec.slot in selected_slots]
    grouped: dict[str, list[CredentialSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.provider, []).append(spec)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(grouped)))) as executor:
        futures = {
            executor.submit(_check_provider, provider_specs): provider
            for provider, provider_specs in grouped.items()
        }
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: (str(item["provider"]), int(item["slot"])))

    disabled: list[str] = []
    if args.disable_invalid:
        invalid_slots = {
            (str(result["provider"]), int(result["slot"]))
            for result in results
            if result["status"] in {"invalid_auth", "incomplete_pair"}
        }
        names = {
            env_name
            for spec in specs
            if (spec.provider, spec.slot) in invalid_slots
            for env_name in spec.env_names
        }
        disabled = _disable_env_names(names)

    report = {
        "configured_credentials": len(specs),
        "summary": _summary(results),
        "results": results,
        "disabled_env_names": disabled,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    bad_statuses = {"invalid_auth", "incomplete_pair", "network_error", "request_failed"}
    return 1 if any(result["status"] in bad_statuses for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
