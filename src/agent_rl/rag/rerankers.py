"""HTTP reranker adapters."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol

from agent_rl.rag.types import RerankResult


class Reranker(Protocol):
    """Ranks candidate texts against a query."""

    def rerank(self, *, query: str, documents: list[str], top_n: int = 30) -> list[RerankResult]:
        ...


class HTTPReranker:
    """Reranker backed by `/v1/rerank`, compatible with the old narrative service."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "Qwen/Qwen3-Reranker-4B",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        if not self.base_url:
            raise ValueError("rerank base_url is required")

    def rerank(self, *, query: str, documents: list[str], top_n: int = 30) -> list[RerankResult]:
        if not documents:
            return []
        payload = json.dumps(
            {"model": self.model, "query": query, "documents": documents, "top_n": top_n},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/rerank",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        rows = body.get("results", body if isinstance(body, list) else [])
        return [_row_to_result(row, documents) for row in rows if isinstance(row, dict)]


def _row_to_result(row: dict[str, Any], documents: list[str]) -> RerankResult:
    index = int(row.get("index", 0) or 0)
    text = documents[index] if 0 <= index < len(documents) else str(row.get("text") or "")
    score = float(row.get("score", row.get("relevance_score", 0.0)) or 0.0)
    return RerankResult(index=index, score=score, text=text, metadata=dict(row))
