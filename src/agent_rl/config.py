"""Cross-platform environment and configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping


SENSITIVE_KEY_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def load_env_file(path: str | Path, *, override: bool = False) -> Path | None:
    """Load a simple dotenv file on Windows, Linux, or macOS."""

    env_path = Path(path).expanduser()
    if not env_path.exists():
        return None
    values = parse_env_file(env_path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = expand_env_value(value, os.environ)
    return env_path


def load_project_env(path: str | Path | None = None, *, override: bool = False, start: str | Path | None = None) -> Path | None:
    """Find and load the project `.env`, preserving existing environment by default."""

    env_path = Path(path).expanduser() if path is not None else find_env_file(start=start)
    if env_path is None:
        return None
    return load_env_file(env_path, override=override)


def find_env_file(
    *,
    start: str | Path | None = None,
    filenames: Iterable[str] = (".env",),
    stop_at: str | Path | None = None,
) -> Path | None:
    """Search upward for an env file from a starting directory."""

    current = Path(start).expanduser() if start is not None else Path.cwd()
    if current.is_file():
        current = current.parent
    current = current.resolve()
    stop = Path(stop_at).expanduser().resolve() if stop_at is not None else None
    while True:
        for filename in filenames:
            candidate = current / filename
            if candidate.exists():
                return candidate
        if stop is not None and current == stop:
            return None
        if current.parent == current:
            return None
        current = current.parent


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a dotenv file into raw string values."""

    env_path = Path(path).expanduser()
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        if not key:
            continue
        values[key] = _strip_quotes(value.strip())
    return values


def get_env(name: str, default: str = "", *, aliases: Iterable[str] = ()) -> str:
    """Read an environment variable with optional fallback aliases."""

    for key in (name, *aliases):
        value = os.getenv(key)
        if value is not None and value != "":
            return value
    return default


def get_env_int(name: str, default: int, *, aliases: Iterable[str] = ()) -> int:
    value = get_env(name, "", aliases=aliases)
    try:
        return int(value) if value != "" else default
    except ValueError:
        return default


def get_env_float(name: str, default: float, *, aliases: Iterable[str] = ()) -> float:
    value = get_env(name, "", aliases=aliases)
    try:
        return float(value) if value != "" else default
    except ValueError:
        return default


def get_env_bool(name: str, default: bool = False, *, aliases: Iterable[str] = ()) -> bool:
    value = get_env(name, "", aliases=aliases).strip().lower()
    if value == "":
        return default
    return value not in {"0", "false", "off", "no", "n"}


def env_snapshot(keys: Iterable[str], *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a redacted env snapshot safe for logs and diagnostics."""

    source = environ or os.environ
    snapshot: dict[str, str] = {}
    for key in keys:
        value = source.get(key)
        if value in (None, ""):
            snapshot[key] = "<missing>"
        elif is_sensitive_key(key):
            snapshot[key] = "<set>"
        else:
            snapshot[key] = value
    return snapshot


def is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SENSITIVE_KEY_MARKERS)


def expand_env_value(value: str, environ: Mapping[str, str] | None = None) -> str:
    """Expand `${NAME}` placeholders with values from the provided environment."""

    source = environ or os.environ
    expanded = value
    for key, current in source.items():
        expanded = expanded.replace("${" + key + "}", current)
    return expanded


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
