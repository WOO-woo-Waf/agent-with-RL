"""File-backed repository for narrative source-analysis assets."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

from agent_rl.domains.narrative import (
    ChapterAnalysisResult,
    ChunkAnalysisResult,
    GlobalStoryAnalysisResult,
    NarrativeSourceAnalysis,
    SourceChunk,
    SourceDocument,
)


T = TypeVar("T")


class FileNarrativeAnalysisRepository:
    """Stores analysis assets in a portable JSON/JSONL layout.

    This is intentionally a repository adapter, not domain logic. The layout is
    easy to inspect during learning and can later be replaced by SQLite,
    PostgreSQL, pgvector, or a managed artifact store without changing the
    narrative agent policies.
    """

    def __init__(self, root: str | Path = Path("artifacts") / "narrative") -> None:
        self.root = Path(root)

    def save_source_analysis(self, analysis: NarrativeSourceAnalysis) -> None:
        story_dir = self._story_dir(analysis.story_id, analysis.task_id)
        story_dir.mkdir(parents=True, exist_ok=True)
        self.save_source_documents(
            story_id=analysis.story_id,
            task_id=analysis.task_id,
            documents=analysis.source_documents,
        )
        self.save_source_chunks(
            story_id=analysis.story_id,
            task_id=analysis.task_id,
            chunks=analysis.source_chunks,
        )
        self.save_chunk_analyses(
            story_id=analysis.story_id,
            task_id=analysis.task_id,
            analyses=analysis.chunk_analyses,
        )
        self.save_chapter_analyses(
            story_id=analysis.story_id,
            task_id=analysis.task_id,
            analyses=analysis.chapter_analyses,
        )
        if analysis.global_analysis is not None:
            self.save_global_analysis(
                story_id=analysis.story_id,
                task_id=analysis.task_id,
                analysis=analysis.global_analysis,
            )
        self._write_json(story_dir / "source_analysis.json", _to_jsonable(analysis))
        self._write_json(
            story_dir / "manifest.json",
            {
                "story_id": analysis.story_id,
                "task_id": analysis.task_id,
                "analysis_id": analysis.analysis_id,
                "saved_at": _utc_now(),
                "layout": "agent_rl.narrative_analysis.v1",
                "files": [
                    "manifest.json",
                    "source_analysis.json",
                    "source_documents.json",
                    "source_chunks.jsonl",
                    "chunk_analysis.jsonl",
                    "chapter_analysis.jsonl",
                    "global_analysis.json",
                ],
                "coverage": dict(analysis.coverage),
                "trace": list(analysis.trace),
            },
        )

    def save_source_documents(self, *, story_id: str, task_id: str, documents: Sequence[SourceDocument]) -> None:
        self._write_json(self._story_dir(story_id, task_id) / "source_documents.json", [_to_jsonable(item) for item in documents])

    def save_source_chunks(self, *, story_id: str, task_id: str, chunks: Sequence[SourceChunk]) -> None:
        self._write_jsonl(self._story_dir(story_id, task_id) / "source_chunks.jsonl", chunks)

    def save_chunk_analyses(
        self,
        *,
        story_id: str,
        task_id: str,
        analyses: Sequence[ChunkAnalysisResult],
    ) -> None:
        self._write_jsonl(self._story_dir(story_id, task_id) / "chunk_analysis.jsonl", analyses)

    def save_chapter_analyses(
        self,
        *,
        story_id: str,
        task_id: str,
        analyses: Sequence[ChapterAnalysisResult],
    ) -> None:
        self._write_jsonl(self._story_dir(story_id, task_id) / "chapter_analysis.jsonl", analyses)

    def save_global_analysis(
        self,
        *,
        story_id: str,
        task_id: str,
        analysis: GlobalStoryAnalysisResult,
    ) -> None:
        self._write_json(self._story_dir(story_id, task_id) / "global_analysis.json", _to_jsonable(analysis))

    def load_chunk_analyses(self, *, story_id: str, task_id: str) -> list[ChunkAnalysisResult]:
        return _load_typed_jsonl(self._story_dir(story_id, task_id) / "chunk_analysis.jsonl", ChunkAnalysisResult)

    def load_chapter_analyses(self, *, story_id: str, task_id: str) -> list[ChapterAnalysisResult]:
        return _load_typed_jsonl(self._story_dir(story_id, task_id) / "chapter_analysis.jsonl", ChapterAnalysisResult)

    def load_global_analysis(self, *, story_id: str, task_id: str) -> GlobalStoryAnalysisResult | None:
        path = self._story_dir(story_id, task_id) / "global_analysis.json"
        if not path.exists():
            return None
        return GlobalStoryAnalysisResult(**json.loads(path.read_text(encoding="utf-8")))

    def _story_dir(self, story_id: str, task_id: str) -> Path:
        safe_story = _safe_path_part(story_id or "story")
        safe_task = _safe_path_part(task_id or "task")
        return self.root / safe_story / safe_task

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: Iterable[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True) for row in rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _load_typed_jsonl(path: Path, cls: type[T]) -> list[T]:
    if not path.exists():
        return []
    rows: list[T] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(cls(**json.loads(line)))
    return rows


def _safe_path_part(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return clean[:120] or "item"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
