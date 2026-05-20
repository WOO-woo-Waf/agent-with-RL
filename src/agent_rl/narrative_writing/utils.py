"""Internal helpers for the local narrative-writing implementation."""

from __future__ import annotations

from typing import Sequence
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def tokenize(text: str) -> set[str]:
    normalized = (
        text.replace("，", " ")
        .replace("。", " ")
        .replace("；", " ")
        .replace("：", " ")
        .replace("、", " ")
        .replace("\n", " ")
    )
    terms = {part.strip().lower() for part in normalized.split() if len(part.strip()) >= 2}
    for marker in ("主角", "伏笔", "线索", "关系", "风格", "世界", "剧情", "下一章", "不要", "必须"):
        if marker in text:
            terms.add(marker)
    return terms


def split_author_items(text: str) -> list[str]:
    for separator in ("；", ";", "。", "\n"):
        text = text.replace(separator, "|")
    return [part.strip() for part in text.split("|") if part.strip()][:6]


def unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_negative_constraint(text: str) -> bool:
    return any(marker in text for marker in ("不要", "禁止", "不能", "避免"))


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.splitlines() if part.strip()]


def candidate_names(text: str) -> list[str]:
    names: list[str] = []
    for marker in ("林舟", "沈", "顾", "江", "陈", "陆", "许", "周"):
        if marker in text and marker not in names:
            names.append(marker if len(marker) > 1 else f"{marker}姓角色")
    return names
