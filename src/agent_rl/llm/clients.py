"""OpenAI-compatible chat model client for the whole package."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from threading import Lock
from typing import Any, Sequence

from agent_rl.llm.audit import (
    build_interaction_record,
    build_usage_record,
    new_interaction_id,
    record_interaction,
    record_token_usage,
)
from agent_rl.llm.contracts import ChatMessage


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_base: str = ""
    api_key: str = ""
    model_name: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout_s: float = 120.0
    max_attempts: int = 3
    base_backoff_s: float = 0.6

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        return cls(
            api_base=os.getenv("LLM_API_BASE", "").strip(),
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            model_name=os.getenv("LLM_MODEL", "").strip(),
            temperature=_float_env("LLM_TEMPERATURE", 0.2),
            max_tokens=_int_env("LLM_MAX_TOKENS", 4096),
            top_p=_float_env("LLM_TOP_P", 1.0),
            timeout_s=_float_env("LLM_TIMEOUT_S", 120.0),
            max_attempts=_int_env("LLM_MAX_ATTEMPTS", 3),
            base_backoff_s=_float_env("LLM_BASE_BACKOFF_S", 0.6),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_base and self.api_key and self.model_name)


@dataclass(frozen=True)
class LLMEndpoint:
    api_base: str
    api_key: str


class EndpointPool:
    """Round-robin start point for multiple endpoints."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counter = 0

    def iter_from(self, endpoints: list[LLMEndpoint]) -> list[LLMEndpoint]:
        if not endpoints:
            return []
        with self._lock:
            start = self._counter % len(endpoints)
            self._counter += 1
        return list(endpoints[start:]) + list(endpoints[:start])


class OpenAICompatibleChatClient:
    """Small OpenAI-compatible `/chat/completions` client with audit logging."""

    def __init__(self, config: OpenAICompatibleConfig | None = None, endpoint_pool: EndpointPool | None = None) -> None:
        self.config = config or OpenAICompatibleConfig.from_env()
        self.endpoint_pool = endpoint_pool or EndpointPool()

    def complete(self, messages: Sequence[ChatMessage], *, purpose: str, json_mode: bool = True) -> str:
        if not self.config.configured:
            raise RuntimeError("LLM configuration is incomplete. Set LLM_API_BASE, LLM_API_KEY, and LLM_MODEL.")

        request_messages = [dict(message) for message in messages]
        if json_mode:
            request_messages = _ensure_json_mode_prompt_contract(request_messages)

        endpoints = _resolve_endpoints(self.config)
        interaction_id = new_interaction_id()
        last_error: BaseException | None = None
        for endpoint in self.endpoint_pool.iter_from(endpoints):
            for attempt in range(1, max(self.config.max_attempts, 1) + 1):
                started_at = time.perf_counter()
                record_interaction(
                    build_interaction_record(
                        interaction_id=interaction_id,
                        event_type="llm_request_started",
                        model_name=self.config.model_name,
                        api_base=endpoint.api_base,
                        purpose=purpose,
                        messages=request_messages,
                        json_mode=json_mode,
                        timeout_s=self.config.timeout_s,
                        success=False,
                        attempt=attempt,
                        max_attempts=self.config.max_attempts,
                    )
                )
                try:
                    payload = self._post_chat_completion(endpoint, request_messages, json_mode=json_mode)
                    content = _extract_content(payload)
                    if json_mode and not content.strip():
                        raise RuntimeError("JSON mode returned empty content.")
                    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                    duration_ms = _duration_ms(started_at)
                    record_interaction(
                        build_interaction_record(
                            interaction_id=interaction_id,
                            event_type="llm_request_succeeded",
                            model_name=self.config.model_name,
                            api_base=endpoint.api_base,
                            purpose=purpose,
                            messages=request_messages,
                            json_mode=json_mode,
                            timeout_s=self.config.timeout_s,
                            success=True,
                            attempt=attempt,
                            max_attempts=self.config.max_attempts,
                            duration_ms=duration_ms,
                            response_text=content,
                        )
                    )
                    record_token_usage(
                        build_usage_record(
                            interaction_id=interaction_id,
                            model_name=self.config.model_name,
                            api_base=endpoint.api_base,
                            purpose=purpose,
                            success=True,
                            duration_ms=duration_ms,
                            attempt=attempt,
                            max_attempts=self.config.max_attempts,
                            usage=usage,
                        )
                    )
                    return content
                except Exception as exc:  # noqa: BLE001 - audit and retry boundary.
                    last_error = exc
                    duration_ms = _duration_ms(started_at)
                    retryable = _is_retryable_error(exc)
                    record_interaction(
                        build_interaction_record(
                            interaction_id=interaction_id,
                            event_type="llm_request_failed",
                            model_name=self.config.model_name,
                            api_base=endpoint.api_base,
                            purpose=purpose,
                            messages=request_messages,
                            json_mode=json_mode,
                            timeout_s=self.config.timeout_s,
                            success=False,
                            attempt=attempt,
                            max_attempts=self.config.max_attempts,
                            duration_ms=duration_ms,
                            retryable_error=retryable,
                            error=exc,
                        )
                    )
                    record_token_usage(
                        build_usage_record(
                            interaction_id=interaction_id,
                            model_name=self.config.model_name,
                            api_base=endpoint.api_base,
                            purpose=purpose,
                            success=False,
                            duration_ms=duration_ms,
                            attempt=attempt,
                            max_attempts=self.config.max_attempts,
                            error=exc,
                        )
                    )
                    if not retryable:
                        break
                    if attempt < self.config.max_attempts:
                        _sleep_backoff(attempt=attempt, base_backoff_s=self.config.base_backoff_s)
        raise RuntimeError(f"All LLM endpoints failed: {last_error}")

    def _post_chat_completion(
        self,
        endpoint: LLMEndpoint,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            _chat_completions_url(endpoint.api_base),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {endpoint.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:  # noqa: S310
            payload = response.read().decode("utf-8")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM provider returned a non-object response.")
        return parsed


def has_llm_configuration(config: OpenAICompatibleConfig | None = None) -> bool:
    return (config or OpenAICompatibleConfig.from_env()).configured


def _resolve_endpoints(config: OpenAICompatibleConfig) -> list[LLMEndpoint]:
    bases = _split_multi(os.getenv("LLM_API_BASES", "")) or ([config.api_base] if config.api_base else [])
    keys = _split_multi(os.getenv("LLM_API_KEYS", "")) or ([config.api_key] if config.api_key else [])
    if len(keys) == 1:
        return [LLMEndpoint(api_base=base, api_key=keys[0]) for base in bases]
    return [LLMEndpoint(api_base=base, api_key=key) for base in bases for key in keys]


def _split_multi(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in raw.replace(";", ",").replace("\n", ",").split(",") if part.strip()]


def _chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM provider response missing choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LLM provider response has invalid choice shape.")
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise RuntimeError("LLM provider response missing message content.")


def _ensure_json_mode_prompt_contract(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    reminder = (
        "\n\nJSON mode contract: output exactly one valid JSON object. "
        "Do not output markdown, code fences, comments, or blank content."
    )
    rows = [dict(message) for message in messages]
    for message in rows:
        if message.get("role") == "system":
            content = str(message.get("content", ""))
            if "json mode contract" not in content.lower():
                message["content"] = content + reminder
            return rows
    return [{"role": "system", "content": reminder.strip()}] + rows


def _is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    if isinstance(error, urllib.error.URLError):
        return True
    message = str(error).lower()
    return any(flag in message for flag in ("timeout", "timed out", "rate limit", "connection", "gateway"))


def _sleep_backoff(*, attempt: int, base_backoff_s: float) -> None:
    delay = base_backoff_s * (2 ** max(attempt - 1, 0))
    delay *= 1.0 + random.random() * 0.2
    time.sleep(delay)


def _duration_ms(started_at: float) -> int:
    return max(int((time.perf_counter() - started_at) * 1000), 0)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default
