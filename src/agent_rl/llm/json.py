"""JSON parsing helpers for model outputs."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonParseResult:
    data: Any
    repaired: bool = False
    source: str = "json"


class JsonBlobParser:
    """Extracts and parses JSON from model output with conservative repairs."""

    def parse(self, raw: str) -> JsonParseResult:
        candidate = extract_json_text(raw)
        try:
            return JsonParseResult(data=json.loads(candidate), repaired=False, source="json")
        except json.JSONDecodeError:
            repaired = _repair_common_json(candidate)
            try:
                return JsonParseResult(data=json.loads(repaired), repaired=True, source="json_repaired")
            except json.JSONDecodeError:
                try:
                    return JsonParseResult(data=ast.literal_eval(repaired), repaired=True, source="python_literal")
                except (SyntaxError, ValueError) as exc:
                    raise ValueError("model response did not contain parseable JSON") from exc


def extract_json_text(raw: str) -> str:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("{") or text.startswith("["):
        return text
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        end = _find_balanced_end(text, start, opener, closer)
        if end >= 0:
            return text[start : end + 1]
    raise ValueError("model response did not contain a JSON object or array")


def _find_balanced_end(text: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _repair_common_json(text: str) -> str:
    repaired = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired
