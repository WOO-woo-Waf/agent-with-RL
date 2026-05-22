"""SQLite vector store for local RAG experiments."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Protocol, Sequence

from agent_rl.rag.types import RAGDocument, VectorSearchResult


class VectorStore(Protocol):
    """Stores and searches embedded RAG documents."""

    def upsert_documents(self, documents: Sequence[RAGDocument], embeddings: Sequence[Sequence[float]], *, collection_id: str = "default") -> int:
        ...

    def search(
        self,
        *,
        embedding: Sequence[float],
        story_id: str = "",
        evidence_types: Sequence[str] | None = None,
        collection_id: str = "default",
        limit: int = 20,
    ) -> list[VectorSearchResult]:
        ...


class SQLiteVectorStore:
    """Portable local vector store using SQLite plus Python cosine scoring.

    This is intentionally simple and dependency-light. It is good enough for
    local learning, tests, and small projects; FAISS/pgvector/Qdrant can replace
    this through the same `VectorStore` port later.
    """

    def __init__(self, path: str | Path = Path("artifacts") / "rag" / "vector_store.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def upsert_documents(self, documents: Sequence[RAGDocument], embeddings: Sequence[Sequence[float]], *, collection_id: str = "default") -> int:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        with self._connect() as conn:
            for document, embedding in zip(documents, embeddings):
                conn.execute(
                    """
                    INSERT INTO rag_vectors(collection_id, document_id, story_id, evidence_type, source, text,
                                            related_entities, related_plot_threads, chapter_index, embedding_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection_id, document_id) DO UPDATE SET
                        story_id=excluded.story_id,
                        evidence_type=excluded.evidence_type,
                        source=excluded.source,
                        text=excluded.text,
                        related_entities=excluded.related_entities,
                        related_plot_threads=excluded.related_plot_threads,
                        chapter_index=excluded.chapter_index,
                        embedding_json=excluded.embedding_json,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        collection_id,
                        document.document_id,
                        document.story_id,
                        document.evidence_type,
                        document.source,
                        document.text,
                        json.dumps(document.related_entities, ensure_ascii=False),
                        json.dumps(document.related_plot_threads, ensure_ascii=False),
                        document.chapter_index,
                        json.dumps([float(item) for item in embedding]),
                        json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                    ),
                )
        return len(documents)

    def search(
        self,
        *,
        embedding: Sequence[float],
        story_id: str = "",
        evidence_types: Sequence[str] | None = None,
        collection_id: str = "default",
        limit: int = 20,
    ) -> list[VectorSearchResult]:
        sql = "SELECT * FROM rag_vectors WHERE collection_id = ?"
        params: list[object] = [collection_id]
        if story_id:
            sql += " AND story_id = ?"
            params.append(story_id)
        evidence_filter = [item for item in (evidence_types or ()) if item]
        if evidence_filter:
            placeholders = ",".join("?" for _ in evidence_filter)
            sql += f" AND evidence_type IN ({placeholders})"
            params.extend(evidence_filter)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        scored: list[VectorSearchResult] = []
        query_vector = [float(item) for item in embedding]
        for row in rows:
            vector = [float(item) for item in json.loads(row["embedding_json"])]
            score = _cosine(query_vector, vector)
            scored.append(
                VectorSearchResult(
                    evidence_id=str(row["document_id"]),
                    evidence_type=str(row["evidence_type"]),
                    source=str(row["source"] or "sqlite_vector"),
                    text=str(row["text"]),
                    score=score,
                    related_entities=list(json.loads(row["related_entities"] or "[]")),
                    related_plot_threads=list(json.loads(row["related_plot_threads"] or "[]")),
                    chapter_index=row["chapter_index"],
                    metadata=dict(json.loads(row["metadata_json"] or "{}")),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def count(self, *, collection_id: str = "default", story_id: str = "") -> int:
        sql = "SELECT COUNT(*) AS count FROM rag_vectors WHERE collection_id = ?"
        params: list[object] = [collection_id]
        if story_id:
            sql += " AND story_id = ?"
            params.append(story_id)
        with self._connect() as conn:
            return int(conn.execute(sql, params).fetchone()["count"])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_vectors(
                    collection_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    related_entities TEXT NOT NULL,
                    related_plot_threads TEXT NOT NULL,
                    chapter_index INTEGER,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(collection_id, document_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_vectors_story ON rag_vectors(collection_id, story_id)")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))
