"""Package-wide LLM contracts."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence


ChatMessage = Mapping[str, str]


class ChatModelClient(Protocol):
    """Minimal chat-model boundary used by Agent policies and integrations."""

    def complete(self, messages: Sequence[ChatMessage], *, purpose: str, json_mode: bool = True) -> str:
        ...
