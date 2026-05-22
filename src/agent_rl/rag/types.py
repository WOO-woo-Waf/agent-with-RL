"""Shared RAG value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RAGDocument:
    """One document or memory item that can be embedded and retrieved."""

    document_id: str
    text: str
    story_id: str = ""
    evidence_type: str = "memory"
    source: str = ""
    related_entities: list[str] = field(default_factory=list)
    related_plot_threads: list[str] = field(default_factory=list)
    chapter_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    """A ranked vector retrieval result."""

    evidence_id: str
    evidence_type: str
    source: str
    text: str
    score: float = 0.0
    related_entities: list[str] = field(default_factory=list)
    related_plot_threads: list[str] = field(default_factory=list)
    chapter_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """A reranked document row returned by a reranker service."""

    index: int
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
