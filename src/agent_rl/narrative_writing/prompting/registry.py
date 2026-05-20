"""File-backed prompt registry for narrative-writing policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: str
    task: str
    output_contract: str
    content: str
    path: Path
    content_hash: str


@dataclass(frozen=True)
class PromptBinding:
    purpose: str
    task_prompt: str


@dataclass(frozen=True)
class PromptProfile:
    id: str
    version: str
    global_prompt: str
    reasoning_mode: str
    bindings: dict[str, PromptBinding]


@dataclass(frozen=True)
class ComposedPrompt:
    system_content: str
    metadata: dict[str, str]


class PromptRegistry:
    """Loads prompt templates and purpose bindings from files."""

    def __init__(self, prompt_dir: str | Path | None = None) -> None:
        self.prompt_dir = Path(prompt_dir) if prompt_dir is not None else Path(__file__).parent / "templates"

    def load_profile(self, profile_id: str = "default") -> PromptProfile:
        path = self.prompt_dir / "profiles" / f"{profile_id}.yaml"
        data = _parse_simple_yaml_text(path.read_text(encoding="utf-8"))
        bindings_raw = data.get("bindings")
        if not isinstance(bindings_raw, dict):
            raise ValueError(f"prompt profile `{profile_id}` must define bindings")
        bindings = {
            str(purpose): PromptBinding(purpose=str(purpose), task_prompt=str(task_prompt))
            for purpose, task_prompt in bindings_raw.items()
            if str(purpose).strip() and str(task_prompt).strip()
        }
        return PromptProfile(
            id=_required_str(data, "id", path),
            version=_required_str(data, "version", path),
            global_prompt=_required_str(data, "global_prompt", path),
            reasoning_mode=str(data.get("reasoning_mode") or "internal"),
            bindings=bindings,
        )

    def get_binding(self, purpose: str, profile: PromptProfile | None = None) -> PromptBinding:
        active_profile = profile or self.load_profile()
        binding = active_profile.bindings.get(purpose)
        if binding is None:
            raise KeyError(f"no prompt binding configured for purpose `{purpose}`")
        return binding

    def load_global_prompt(self, name: str) -> PromptTemplate:
        return self._load_template(self.prompt_dir / "global" / f"{name}.md", expected_task="global")

    def load_task_prompt(self, name: str, *, expected_task: str) -> PromptTemplate:
        return self._load_template(self.prompt_dir / "tasks" / f"{name}.md", expected_task=expected_task)

    def _load_template(self, path: Path, *, expected_task: str) -> PromptTemplate:
        front_matter, content = _split_front_matter(path.read_text(encoding="utf-8"), path)
        template = PromptTemplate(
            id=_required_str(front_matter, "id", path),
            version=_required_str(front_matter, "version", path),
            task=_required_str(front_matter, "task", path),
            output_contract=_required_str(front_matter, "output_contract", path),
            content=content.strip(),
            path=path,
            content_hash=_hash_text(content.strip()),
        )
        if template.task != expected_task:
            raise ValueError(f"prompt `{path}` declares task `{template.task}`, expected `{expected_task}`")
        if not template.content:
            raise ValueError(f"prompt `{path}` has empty content")
        return template


class PromptComposer:
    """Composes global and task prompts for one model purpose."""

    def __init__(self, registry: PromptRegistry | None = None) -> None:
        self.registry = registry or PromptRegistry()

    def compose_system_prompt(self, *, purpose: str, profile_id: str = "default") -> ComposedPrompt:
        profile = self.registry.load_profile(profile_id)
        binding = self.registry.get_binding(purpose, profile)
        global_prompt = self.registry.load_global_prompt(profile.global_prompt)
        task_prompt = self.registry.load_task_prompt(binding.task_prompt, expected_task=purpose)
        metadata = {
            "prompt_profile": profile.id,
            "prompt_profile_version": profile.version,
            "global_prompt_id": global_prompt.id,
            "global_prompt_version": global_prompt.version,
            "global_prompt_hash": global_prompt.content_hash,
            "task_prompt_id": task_prompt.id,
            "task_prompt_version": task_prompt.version,
            "task_prompt_hash": task_prompt.content_hash,
            "reasoning_mode": profile.reasoning_mode,
        }
        return ComposedPrompt(
            system_content="\n\n".join([global_prompt.content, task_prompt.content, _metadata_block(metadata)]),
            metadata=metadata,
        )


def compose_system_prompt(*, purpose: str, profile_id: str = "default") -> ComposedPrompt:
    return PromptComposer().compose_system_prompt(purpose=purpose, profile_id=profile_id)


def _split_front_matter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    text = raw.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError(f"prompt `{path}` must start with front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"prompt `{path}` has unterminated front matter")
    return _parse_simple_yaml_text(text[4:end]), text[end + len("\n---\n") :]


def _parse_simple_yaml_text(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_map_key: str | None = None
    for raw_line in text.replace("\r\n", "\n").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("  ") and current_map_key:
            key, value = _split_key_value(raw_line.strip())
            nested = data.setdefault(current_map_key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"`{current_map_key}` cannot mix scalar and mapping values")
            nested[key] = value
            continue
        key, value = _split_key_value(raw_line.strip())
        if value == "":
            data[key] = {}
            current_map_key = key
        else:
            data[key] = value
            current_map_key = None
    return data


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"invalid prompt metadata line: {line}")
    key, value = line.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"invalid empty prompt metadata key: {line}")
    return key, value.strip().strip('"').strip("'")


def _required_str(data: dict[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"`{key}` is required in `{path}`")
    return value


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _metadata_block(metadata: dict[str, str]) -> str:
    lines = ["# Prompt Metadata"]
    lines.extend(f"{key}: {value}" for key, value in sorted(metadata.items()))
    return "\n".join(lines)
