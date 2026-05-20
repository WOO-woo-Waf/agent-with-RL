"""JSONL audit logging for package-wide LLM calls."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Sequence
from uuid import uuid4

from agent_rl.llm.contracts import ChatMessage


DEFAULT_AUDIT_LOG_PATH = Path("artifacts") / "llm" / "interactions.jsonl"
DEFAULT_USAGE_LOG_PATH = Path("artifacts") / "llm" / "token_usage.jsonl"
MAX_PREVIEW_CHARS = 1200


@dataclass(frozen=True)
class LLMInteractionAuditRecord:
    timestamp: str
    interaction_id: str
    event_type: str
    model_name: str
    api_base: str
    purpose: str
    success: bool
    attempt: int
    max_attempts: int
    duration_ms: int = 0
    message_count: int = 0
    request_chars: int = 0
    response_chars: int = 0
    request_preview: str = ""
    response_preview: str = ""
    system_prompt_preview: str = ""
    user_prompt_preview: str = ""
    json_mode: bool = False
    timeout_s: float = 0.0
    retryable_error: bool = False
    prompt_profile: str = ""
    prompt_profile_version: str = ""
    global_prompt_id: str = ""
    global_prompt_version: str = ""
    global_prompt_hash: str = ""
    task_prompt_id: str = ""
    task_prompt_version: str = ""
    task_prompt_hash: str = ""
    reasoning_mode: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class LLMTokenUsageAuditRecord:
    timestamp: str
    interaction_id: str
    model_name: str
    api_base: str
    purpose: str
    success: bool
    duration_ms: int
    attempt: int
    max_attempts: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    usage_raw: Any = None
    error_type: str = ""
    error_message: str = ""


class JsonlAuditWriter:
    """Thread-safe JSONL writer for stable local audit records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def new_interaction_id() -> str:
    return f"llm-{uuid4().hex}"


def audit_enabled() -> bool:
    return os.getenv("LLM_AUDIT_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}


def record_interaction(record: LLMInteractionAuditRecord) -> None:
    if not audit_enabled():
        return
    JsonlAuditWriter(_audit_path("LLM_AUDIT_LOG_PATH", DEFAULT_AUDIT_LOG_PATH)).write(asdict(record))


def record_token_usage(record: LLMTokenUsageAuditRecord) -> None:
    if not audit_enabled():
        return
    JsonlAuditWriter(_audit_path("LLM_USAGE_LOG_PATH", DEFAULT_USAGE_LOG_PATH)).write(asdict(record))


def build_interaction_record(
    *,
    interaction_id: str,
    event_type: str,
    model_name: str,
    api_base: str,
    purpose: str,
    messages: Sequence[ChatMessage],
    json_mode: bool,
    timeout_s: float,
    success: bool,
    attempt: int,
    max_attempts: int,
    duration_ms: int = 0,
    response_text: str = "",
    retryable_error: bool = False,
    error: BaseException | None = None,
) -> LLMInteractionAuditRecord:
    metadata = extract_prompt_metadata(messages)
    summary = summarize_messages(messages)
    return LLMInteractionAuditRecord(
        timestamp=datetime.now().astimezone().isoformat(),
        interaction_id=interaction_id,
        event_type=event_type,
        model_name=model_name,
        api_base=api_base,
        purpose=purpose,
        success=success,
        attempt=max(int(attempt), 1),
        max_attempts=max(int(max_attempts), 1),
        duration_ms=max(int(duration_ms), 0),
        message_count=summary["message_count"],
        request_chars=summary["request_chars"],
        response_chars=len(response_text),
        request_preview=summary["request_preview"],
        response_preview=_preview(response_text),
        system_prompt_preview=summary["system_prompt_preview"],
        user_prompt_preview=summary["user_prompt_preview"],
        json_mode=json_mode,
        timeout_s=timeout_s,
        retryable_error=retryable_error,
        prompt_profile=metadata.get("prompt_profile", ""),
        prompt_profile_version=metadata.get("prompt_profile_version", ""),
        global_prompt_id=metadata.get("global_prompt_id", ""),
        global_prompt_version=metadata.get("global_prompt_version", ""),
        global_prompt_hash=metadata.get("global_prompt_hash", ""),
        task_prompt_id=metadata.get("task_prompt_id", ""),
        task_prompt_version=metadata.get("task_prompt_version", ""),
        task_prompt_hash=metadata.get("task_prompt_hash", ""),
        reasoning_mode=metadata.get("reasoning_mode", ""),
        error_type=error.__class__.__name__ if error else "",
        error_message=_preview(str(error) if error else ""),
    )


def build_usage_record(
    *,
    interaction_id: str,
    model_name: str,
    api_base: str,
    purpose: str,
    success: bool,
    duration_ms: int,
    attempt: int,
    max_attempts: int,
    usage: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> LLMTokenUsageAuditRecord:
    usage = usage or {}
    return LLMTokenUsageAuditRecord(
        timestamp=datetime.now().astimezone().isoformat(),
        interaction_id=interaction_id,
        model_name=model_name,
        api_base=api_base,
        purpose=purpose,
        success=success,
        duration_ms=max(int(duration_ms), 0),
        attempt=max(int(attempt), 1),
        max_attempts=max(int(max_attempts), 1),
        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
        completion_tokens=_optional_int(usage.get("completion_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        usage_raw=usage or None,
        error_type=error.__class__.__name__ if error else "",
        error_message=_preview(str(error) if error else ""),
    )


def summarize_messages(messages: Sequence[ChatMessage]) -> dict[str, Any]:
    parts = [str(message.get("content", "")) for message in messages]
    system = next((str(message.get("content", "")) for message in messages if message.get("role") == "system"), "")
    user = next((str(message.get("content", "")) for message in messages if message.get("role") == "user"), "")
    joined = "\n\n".join(parts)
    return {
        "message_count": len(messages),
        "request_chars": len(joined),
        "request_preview": _preview(joined),
        "system_prompt_preview": _preview(system),
        "user_prompt_preview": _preview(user),
    }


def extract_prompt_metadata(messages: Sequence[ChatMessage]) -> dict[str, str]:
    for message in messages:
        if message.get("role") != "system":
            continue
        content = str(message.get("content", ""))
        marker = "# Prompt Metadata"
        if marker not in content:
            continue
        _, metadata_text = content.split(marker, 1)
        metadata: dict[str, str] = {}
        for raw_line in metadata_text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata
    return {}


def _audit_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    return Path(raw) if raw else default


def _preview(text: str) -> str:
    if len(text) <= MAX_PREVIEW_CHARS:
        return text
    return text[: MAX_PREVIEW_CHARS - 15] + " ...(truncated)"


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return None
