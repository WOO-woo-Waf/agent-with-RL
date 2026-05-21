"""JSON serialization helpers for narrative dataclass snapshots."""

from __future__ import annotations

import types
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from typing import Any, get_args, get_origin, get_type_hints

from agent_rl.core import concepts as core_concepts
from agent_rl.domains import narrative as narrative_domain


def to_jsonable(value: Any) -> Any:
    """Convert nested dataclasses and datetimes into JSON-compatible values."""

    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def from_jsonable(cls: type[Any], payload: Any) -> Any:
    """Rebuild a dataclass instance from JSON-compatible data."""

    if not is_dataclass(cls) or not isinstance(payload, dict):
        return payload
    hints = get_type_hints(cls, _globalns(), _localns())
    values: dict[str, Any] = {}
    for field in fields(cls):
        if field.name in payload:
            values[field.name] = _coerce(hints.get(field.name, field.type), payload[field.name])
    return cls(**values)


def _coerce(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if value is None:
        return None
    if annotation is Any:
        return value
    if annotation is datetime and isinstance(value, str):
        return datetime.fromisoformat(value)
    if origin in (list, tuple):
        item_type = args[0] if args else Any
        return [_coerce(item_type, item) for item in value]
    if origin is dict:
        return value
    if origin is types.UnionType or origin is getattr(types, "UnionType", None):
        non_none = [item for item in args if item is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value)
        return value
    if str(origin) == "typing.Union":
        non_none = [item for item in args if item is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value)
        return value
    if is_dataclass(annotation) and isinstance(value, dict):
        return from_jsonable(annotation, value)
    return value


def _globalns() -> dict[str, Any]:
    from agent_rl.narrative_writing import requests as narrative_requests

    values: dict[str, Any] = {}
    values.update(vars(narrative_domain))
    values.update(vars(core_concepts))
    values.update(vars(narrative_requests))
    return values


def _localns() -> dict[str, Any]:
    return _globalns()
