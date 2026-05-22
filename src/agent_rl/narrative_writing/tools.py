"""Narrative tool adapters used by ReAct environments and scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from agent_rl.domains.narrative import NarrativeSourceAnalysis, NarrativeTaskState
from agent_rl.narrative_writing.bootstrap import build_author_constraints
from agent_rl.narrative_writing.longform_context import DraftCompressionTool, LongformContextSelector
from agent_rl.narrative_writing.persistence import FileNarrativeStateRepository
from agent_rl.narrative_writing.ports import NarrativeStateRepository
from agent_rl.narrative_writing.requests import AuthorRequest


@dataclass(frozen=True)
class NarrativeToolResult:
    """Compact tool result that can be surfaced as the next observation."""

    tool_name: str
    success: bool
    summary: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


class LoadAnalysisTool:
    """Loads source analysis artifacts into a runnable NarrativeTaskState."""

    def __init__(self, repository: NarrativeStateRepository | None = None) -> None:
        self.repository = repository or FileNarrativeStateRepository()

    def invoke(self, *, source_analysis_path: str | Path, request: AuthorRequest) -> tuple[NarrativeTaskState, NarrativeToolResult]:
        analysis = self.repository.load_source_analysis(source_analysis_path)
        state = build_state_from_analysis(analysis, request)
        return state, NarrativeToolResult(
            tool_name="load_analysis",
            success=True,
            summary="analysis loaded into narrative state",
            metrics={
                "characters_count": float(len(state.characters)),
                "events_count": float(len(state.events)),
                "plot_threads_count": float(len(state.plot_threads)),
                "source_chunks_count": float(len(state.source_chunks)),
            },
            artifacts=[str(source_analysis_path)],
        )


class ScanWorkspaceTool:
    """Collects safe workspace facts without reading source novel contents."""

    def invoke(
        self,
        *,
        cwd: str | Path = ".",
        analysis_path: str = "",
        state_snapshot_path: str = "",
        artifact_root: str = "",
    ) -> NarrativeToolResult:
        root = Path(cwd)
        artifacts = []
        for raw in (analysis_path, state_snapshot_path, artifact_root):
            if raw:
                artifacts.append(str(Path(raw)))
        payload = {
            "cwd": str(root.resolve()),
            "has_env": (root / ".env").exists(),
            "llm_configured": bool(os.getenv("LLM_API_BASE") and os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL")),
            "analysis_path_exists": bool(analysis_path and Path(analysis_path).exists()),
            "state_snapshot_exists": bool(state_snapshot_path and Path(state_snapshot_path).exists()),
            "artifact_root_exists": bool(artifact_root and Path(artifact_root).exists()),
        }
        return NarrativeToolResult(
            tool_name="scan_workspace",
            success=True,
            summary="workspace scanned",
            artifacts=artifacts,
            payload=payload,
        )


class SaveNarrativeArtifactsTool:
    """Persists state, workflow, trajectory, blueprint, draft, and run metadata."""

    def __init__(self, repository: NarrativeStateRepository | None = None) -> None:
        self.repository = repository or FileNarrativeStateRepository()

    def invoke(self, *, state: NarrativeTaskState, workflow: Any = None, trajectory: Any = None, run_id: str = "") -> NarrativeToolResult:
        artifacts: list[str] = []
        artifacts.append(str(self.repository.save_state_snapshot(state, run_id=run_id)))
        if workflow is not None:
            artifacts.append(str(self.repository.save_workflow_snapshot(state.story_id, workflow, run_id=run_id)))
            if getattr(workflow, "proposed_blueprint", None) is not None:
                artifacts.append(str(self.repository.save_blueprint(state.story_id, workflow.proposed_blueprint)))
            if getattr(workflow, "draft", None) is not None:
                chapter_index = getattr(getattr(workflow, "proposed_blueprint", None), "chapter_index", None)
                artifacts.append(str(self.repository.save_draft(state.story_id, workflow.draft, chapter_index=chapter_index)))
            branches = list(getattr(workflow, "branches", []) or [])
            if branches and hasattr(self.repository, "save_branches"):
                artifacts.extend(str(path) for path in self.repository.save_branches(state.story_id, branches, run_id=run_id))
        if trajectory is not None:
            artifacts.append(str(self.repository.save_trajectory(state.story_id, trajectory, run_id=run_id)))
        return NarrativeToolResult(
            tool_name="save_artifacts",
            success=True,
            summary="narrative runtime artifacts saved",
            metrics={"artifact_count": float(len(artifacts))},
            artifacts=artifacts,
        )


def build_state_from_analysis(analysis: NarrativeSourceAnalysis, request: AuthorRequest) -> NarrativeTaskState:
    """Build canonical task state from reusable deep-analysis artifacts."""

    state = NarrativeTaskState(task_id=request.task_id, story_id=request.story_id, goal=request.request)
    state.source_analyses.append(analysis)
    state.source_documents.extend(analysis.source_documents)
    state.source_chunks.extend(analysis.source_chunks)
    state.author_constraints.extend(build_author_constraints(request.constraints))
    state.characters.extend(analysis.characters)
    state.events.extend(analysis.events)
    state.plot_threads.extend(analysis.plot_threads)
    state.world_rules.extend(analysis.world_rules)
    state.style_profile = analysis.style_profile
    state.style_snippets.extend(analysis.style_snippets)
    state.memory_atoms.extend(analysis.memory_atoms)
    state.metadata["source_analysis_id"] = analysis.analysis_id
    state.metadata["source_analysis_coverage"] = dict(analysis.coverage)
    state.metadata["writing_direction"] = request.writing_direction
    if analysis.global_analysis is not None:
        state.metadata["global_analysis_id"] = analysis.global_analysis.analysis_id
        state.metadata["global_chapter_count"] = analysis.global_analysis.chapter_count
    return state


__all__ = [
    "DraftCompressionTool",
    "LoadAnalysisTool",
    "LongformContextSelector",
    "NarrativeToolResult",
    "SaveNarrativeArtifactsTool",
    "ScanWorkspaceTool",
    "build_state_from_analysis",
]
