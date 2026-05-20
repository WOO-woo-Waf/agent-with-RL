"""Memory implementations."""

from __future__ import annotations

from typing import Any


class InMemoryStore:
    """Small default memory store for tests, examples, and local prototypes."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def append(self, key: str, value: Any) -> None:
        items = self._values.setdefault(key, [])
        if not isinstance(items, list):
            raise TypeError(f"Memory key {key!r} is not appendable")
        items.append(value)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)
