"""Embedding providers for local Ollama and OpenAI-compatible services."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Embeds text for RAG retrieval."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class OllamaEmbeddingProvider:
    """Embedding provider backed by Ollama `/api/embed`."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434", model: str = "qwen3-embedding:4b", timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        body = _post_json(f"{self.base_url}/api/embed", payload, timeout_s=self.timeout_s)
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Ollama embed response did not include embeddings.")
        return [_float_vector(item) for item in embeddings]

    def is_healthy(self) -> bool:
        try:
            _get_json(f"{self.base_url}/api/tags", timeout_s=min(self.timeout_s, 5.0))
            return True
        except (OSError, ValueError, urllib.error.URLError):
            return False


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider backed by an OpenAI-compatible `/v1/embeddings` API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "Qwen/Qwen3-Embedding-4B",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "local"
        self.model = model
        self.timeout_s = timeout_s
        if not self.base_url:
            raise ValueError("embedding base_url is required")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        body = _post_json(
            f"{self.base_url}/v1/embeddings",
            payload,
            timeout_s=self.timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        rows = body.get("data")
        if not isinstance(rows, list):
            raise ValueError("embedding response did not include data rows")
        ordered = sorted(rows, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        return [_float_vector(item.get("embedding", [])) for item in ordered if isinstance(item, dict)]


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float, headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _float_vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("embedding row is not a list")
    return [float(item) for item in value]
