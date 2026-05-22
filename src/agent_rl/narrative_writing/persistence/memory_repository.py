"""SQLite-backed narrative memory repository with optional FTS search."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

from agent_rl.domains.narrative import CompressedMemoryBlock, MemoryAtom, NarrativeEvidence, NarrativeTaskState
from agent_rl.narrative_writing.serialization import from_jsonable, to_jsonable
from agent_rl.narrative_writing.utils import new_id


class SQLiteNarrativeMemoryRepository:
    """Stores memory atoms and compressed blocks in SQLite.

    The repository uses FTS5 when available and falls back to LIKE search on
    SQLite builds that do not ship the extension.
    """

    def __init__(self, path: str | Path = Path("artifacts") / "narrative-memory" / "memory.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def upsert_state_memory(self, state: NarrativeTaskState) -> None:
        with self._connect() as conn:
            for atom in state.memory_atoms:
                conn.execute(
                    """
                    INSERT INTO memory_atoms(story_id, memory_id, memory_type, text, canonical, status,
                                             importance, freshness, related_entities, source_span_ids,
                                             state_version_no, invalidation_reason, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(story_id, memory_id) DO UPDATE SET
                        memory_type=excluded.memory_type,
                        text=excluded.text,
                        canonical=excluded.canonical,
                        status=excluded.status,
                        importance=excluded.importance,
                        freshness=excluded.freshness,
                        related_entities=excluded.related_entities,
                        source_span_ids=excluded.source_span_ids,
                        state_version_no=excluded.state_version_no,
                        invalidation_reason=excluded.invalidation_reason,
                        payload=excluded.payload
                    """,
                    (
                        state.story_id,
                        atom.memory_id,
                        atom.memory_type,
                        atom.text,
                        1 if atom.canonical else 0,
                        atom.status,
                        atom.importance,
                        atom.freshness,
                        json.dumps(atom.related_entities, ensure_ascii=False),
                        json.dumps(atom.source_span_ids, ensure_ascii=False),
                        atom.state_version_no,
                        atom.invalidation_reason,
                        json.dumps(to_jsonable(atom), ensure_ascii=False, sort_keys=True),
                    ),
                )
                if atom.canonical and atom.status != "deprecated":
                    self._upsert_fts(conn, state.story_id, atom.memory_id, atom.memory_type, atom.text)
                else:
                    conn.execute("DELETE FROM memory_fts WHERE story_id = ? AND item_id = ?", (state.story_id, atom.memory_id))
            for block in state.compressed_memory:
                conn.execute(
                    """
                    INSERT INTO compressed_memory(story_id, block_id, block_type, scope, summary,
                                                  key_points, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(story_id, block_id) DO UPDATE SET
                        block_type=excluded.block_type,
                        scope=excluded.scope,
                        summary=excluded.summary,
                        key_points=excluded.key_points,
                        payload=excluded.payload
                    """,
                    (
                        state.story_id,
                        block.block_id,
                        block.block_type,
                        block.scope,
                        block.summary,
                        json.dumps(block.key_points, ensure_ascii=False),
                        json.dumps(to_jsonable(block), ensure_ascii=False, sort_keys=True),
                    ),
                )
                text = block.summary + "\n" + "\n".join(block.key_points)
                self._upsert_fts(conn, state.story_id, block.block_id, f"compressed:{block.block_type}", text)

    def search(self, story_id: str, query: str, *, limit: int = 12) -> list[NarrativeEvidence]:
        text = query.strip()
        if not text:
            return []
        with self._connect() as conn:
            rows = self._search_fts(conn, story_id, text, limit=limit)
            if not rows:
                rows = self._search_like(conn, story_id, text, limit=limit)
        evidence: list[NarrativeEvidence] = []
        for row in rows:
            evidence.append(
                NarrativeEvidence(
                    evidence_id=new_id("ev-memory"),
                    evidence_type=str(row["source_type"]),
                    source="sqlite_memory",
                    text=str(row["text"]),
                    usage_hint="retrieved_memory",
                    final_score=float(row["score"]),
                )
            )
        return evidence

    def load_memory_atoms(self, story_id: str, *, include_deprecated: bool = False) -> list[MemoryAtom]:
        sql = "SELECT payload FROM memory_atoms WHERE story_id = ?"
        params: list[object] = [story_id]
        if not include_deprecated:
            sql += " AND status != 'deprecated' AND canonical = 1"
        sql += " ORDER BY state_version_no DESC, importance DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [from_jsonable(MemoryAtom, json.loads(row["payload"])) for row in rows]

    def load_compressed_memory(self, story_id: str) -> list[CompressedMemoryBlock]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM compressed_memory WHERE story_id = ? ORDER BY block_id",
                (story_id,),
            ).fetchall()
        return [from_jsonable(CompressedMemoryBlock, json.loads(row["payload"])) for row in rows]

    def invalidate_memory_atoms(self, story_id: str, memory_ids: Sequence[str], *, reason: str = "") -> int:
        ids = [item for item in memory_ids if item]
        if not ids:
            return 0
        with self._connect() as conn:
            count = 0
            for memory_id in ids:
                row = conn.execute(
                    "SELECT payload FROM memory_atoms WHERE story_id = ? AND memory_id = ?",
                    (story_id, memory_id),
                ).fetchone()
                if row is None:
                    continue
                atom = from_jsonable(MemoryAtom, json.loads(row["payload"]))
                atom.canonical = False
                atom.status = "deprecated"
                atom.invalidation_reason = reason
                conn.execute(
                    """
                    UPDATE memory_atoms
                    SET canonical = 0, status = 'deprecated', invalidation_reason = ?, payload = ?
                    WHERE story_id = ? AND memory_id = ?
                    """,
                    (reason, json.dumps(to_jsonable(atom), ensure_ascii=False, sort_keys=True), story_id, memory_id),
                )
                conn.execute("DELETE FROM memory_fts WHERE story_id = ? AND item_id = ?", (story_id, memory_id))
                count += 1
        return count

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_atoms(
                    story_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    canonical INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    importance REAL NOT NULL,
                    freshness REAL NOT NULL,
                    related_entities TEXT NOT NULL,
                    source_span_ids TEXT NOT NULL,
                    state_version_no INTEGER,
                    invalidation_reason TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(story_id, memory_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compressed_memory(
                    story_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    block_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_points TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(story_id, block_id)
                )
                """
            )
            self._ensure_fts(conn)

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(story_id UNINDEXED, item_id UNINDEXED, source_type UNINDEXED, text)
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_atoms_story ON memory_atoms(story_id)")
        except sqlite3.OperationalError:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_fts(
                    story_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY(story_id, item_id)
                )
                """
            )

    def _upsert_fts(self, conn: sqlite3.Connection, story_id: str, item_id: str, source_type: str, text: str) -> None:
        conn.execute("DELETE FROM memory_fts WHERE story_id = ? AND item_id = ?", (story_id, item_id))
        conn.execute(
            "INSERT INTO memory_fts(story_id, item_id, source_type, text) VALUES (?, ?, ?, ?)",
            (story_id, item_id, source_type, text),
        )

    def _search_fts(self, conn: sqlite3.Connection, story_id: str, query: str, *, limit: int) -> list[sqlite3.Row]:
        try:
            return conn.execute(
                """
                SELECT story_id, item_id, source_type, text, 1.0 / (1.0 + bm25(memory_fts)) AS score
                FROM memory_fts
                WHERE story_id = ? AND memory_fts MATCH ?
                ORDER BY bm25(memory_fts)
                LIMIT ?
                """,
                (story_id, _fts_query(query), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    def _search_like(self, conn: sqlite3.Connection, story_id: str, query: str, *, limit: int) -> list[sqlite3.Row]:
        terms = [term for term in query.split() if term]
        if not terms:
            terms = [query]
        sql = "SELECT story_id, item_id, source_type, text, 0.25 AS score FROM memory_fts WHERE story_id = ?"
        params: list[object] = [story_id]
        for term in terms[:4]:
            sql += " AND text LIKE ?"
            params.append(f"%{term}%")
        sql += " LIMIT ?"
        params.append(limit)
        return conn.execute(sql, params).fetchall()


def _fts_query(query: str) -> str:
    terms = [term.replace('"', "") for term in query.split() if term.strip()]
    return " OR ".join(f'"{term}"' for term in terms[:12]) or query.replace('"', "")


__all__ = ["SQLiteNarrativeMemoryRepository"]
