"""File-backed runtime state repository for long-form narrative sessions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_rl.core.concepts import Observation, Trajectory
from agent_rl.domains.narrative import ChapterBlueprint, DraftBranch, DraftCandidate, NarrativeSourceAnalysis, NarrativeTaskState
from agent_rl.narrative_writing.serialization import from_jsonable, to_jsonable


class FileNarrativeStateRepository:
    """Stores state snapshots, blueprints, drafts, trajectories, and run metadata."""

    def __init__(self, root: str | Path = Path("artifacts") / "narrative-state") -> None:
        self.root = Path(root)

    def save_state_snapshot(self, state: NarrativeTaskState, *, run_id: str = "") -> Path:
        story_dir = self._story_dir(state.story_id)
        path = story_dir / "state_snapshots" / f"state-v{state.state_version_no:04d}{_suffix(run_id)}.json"
        self._write_json(path, {"saved_at": _utc_now(), "state": to_jsonable(state)})
        return path

    def load_state_snapshot(self, story_id: str, *, path: str | Path | None = None) -> NarrativeTaskState:
        snapshot_path = Path(path) if path is not None else self._latest(self._story_dir(story_id) / "state_snapshots")
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return from_jsonable(NarrativeTaskState, payload.get("state", payload))

    def save_workflow_snapshot(self, story_id: str, workflow: Any, *, run_id: str = "") -> Path:
        path = self._story_dir(story_id) / "workflow_snapshots" / f"workflow{_suffix(run_id)}.json"
        self._write_json(path, {"saved_at": _utc_now(), "workflow": to_jsonable(workflow)})
        return path

    def save_trajectory(self, story_id: str, trajectory: Trajectory, *, run_id: str = "") -> Path:
        path = self._story_dir(story_id) / "trajectories" / f"trajectory{_suffix(run_id)}.json"
        self._write_json(path, {"saved_at": _utc_now(), "trajectory": to_jsonable(trajectory)})
        return path

    def save_blueprint(self, story_id: str, blueprint: ChapterBlueprint, *, chapter_index: int | None = None) -> Path:
        index = chapter_index if chapter_index is not None else blueprint.chapter_index
        path = self._story_dir(story_id) / "blueprints" / f"chapter-{index:04d}.json"
        self._write_json(path, {"saved_at": _utc_now(), "blueprint": to_jsonable(blueprint)})
        return path

    def save_draft(self, story_id: str, draft: DraftCandidate, *, chapter_index: int | None = None) -> Path:
        chapter_label = f"chapter-{chapter_index:04d}" if chapter_index is not None else draft.draft_id
        path = self._story_dir(story_id) / "drafts" / f"{chapter_label}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(draft.content, encoding="utf-8")
        return path

    def save_branches(self, story_id: str, branches: list[DraftBranch], *, run_id: str = "") -> list[Path]:
        paths: list[Path] = []
        branch_dir = self._story_dir(story_id) / "branches"
        for branch in branches:
            branch_path = branch_dir / f"{_safe_path_part(branch.branch_id)}{_suffix(run_id)}.json"
            self._write_json(branch_path, {"saved_at": _utc_now(), "branch": to_jsonable(branch)})
            paths.append(branch_path)
        return paths

    def save_run_result(self, story_id: str, payload: dict[str, Any], *, run_id: str = "") -> Path:
        path = self._story_dir(story_id) / "run_results" / f"run-result{_suffix(run_id)}.json"
        self._write_json(path, {"saved_at": _utc_now(), **to_jsonable(payload)})
        return path

    def load_source_analysis(self, path: str | Path) -> NarrativeSourceAnalysis:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return from_jsonable(NarrativeSourceAnalysis, payload)

    def save_session_snapshot(
        self,
        *,
        session_id: str,
        state: NarrativeTaskState,
        workflow: Any,
        trajectory: Trajectory,
        request: Any,
        observation: Observation | None = None,
        memory_snapshot: dict[str, Any] | None = None,
    ) -> Path:
        path = self._story_dir(state.story_id) / "sessions" / f"{_safe_path_part(session_id)}.json"
        self._write_json(
            path,
            {
                "schema_version": 1,
                "saved_at": _utc_now(),
                "session_id": session_id,
                "story_id": state.story_id,
                "task_id": state.task_id,
                "request": to_jsonable(request),
                "state": to_jsonable(state),
                "workflow": to_jsonable(workflow),
                "trajectory": to_jsonable(trajectory),
                "observation": to_jsonable(observation) if observation is not None else None,
                "memory_snapshot": to_jsonable(memory_snapshot or {}),
            },
        )
        return path

    def load_session_snapshot(self, session_id: str, *, story_id: str = "") -> dict[str, Any]:
        if story_id:
            path = self._story_dir(story_id) / "sessions" / f"{_safe_path_part(session_id)}.json"
        else:
            candidates = sorted(self.root.glob(f"*/sessions/{_safe_path_part(session_id)}.json"))
            if not candidates:
                raise FileNotFoundError(f"No session snapshot found for {session_id}")
            path = candidates[-1]
        return json.loads(path.read_text(encoding="utf-8"))

    def _story_dir(self, story_id: str) -> Path:
        return self.root / _safe_path_part(story_id or "story")

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(path)

    def _latest(self, directory: Path) -> Path:
        candidates = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No snapshots found under {directory}")
        return candidates[0]


def _safe_path_part(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return clean[:120] or "item"


def _suffix(run_id: str) -> str:
    return f"-{_safe_path_part(run_id)}" if run_id else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
