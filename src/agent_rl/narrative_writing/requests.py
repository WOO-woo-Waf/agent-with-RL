"""Request/response DTOs for author-facing narrative Agent interactions."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_rl.core import Trajectory
from agent_rl.domains.narrative import ChapterBlueprint, DraftBranch, DraftCandidate, NarrativeTaskState


@dataclass(frozen=True)
class ReferenceMaterial:
    """User-provided material for canon, style, or cross-reference evidence."""

    title: str
    text: str
    source_type: str = "target_continuation"
    author: str = ""


@dataclass(frozen=True)
class AuthorRequest:
    """One author operation request.

    Missing fields are allowed so the Agent can ask the author for reference
    materials, writing direction, constraints, or confirmation.
    """

    request: str
    session_id: str = ""
    story_id: str = "story-default"
    task_id: str = "task-default"
    references: tuple[ReferenceMaterial, ...] = ()
    writing_direction: str = ""
    blueprint_feedback: str = ""
    constraints: tuple[str, ...] = ()
    target_chapter_index: int = 1
    confirm_plan: bool = False
    branch_count: int = 1
    selected_branch_id: str = ""
    auto_repair: bool = True
    max_repair_attempts: int = 1
    target_word_count: int = 1200
    analysis_path: str = ""
    state_snapshot_path: str = ""
    persist_artifacts: bool = False
    artifact_root: str = ""


@dataclass
class AuthorQuestion:
    """Question returned when required author context is missing."""

    question_id: str
    prompt: str
    reason: str
    required: bool = True


@dataclass
class NarrativeRunResult:
    """Result of one narrative Agent interaction or execution."""

    state: NarrativeTaskState
    trajectory: Trajectory
    questions: list[AuthorQuestion] = field(default_factory=list)
    assistant_message: str = ""
    requires_confirmation: bool = False
    proposed_blueprint: ChapterBlueprint | None = None
    branches: list[DraftBranch] = field(default_factory=list)
    draft: DraftCandidate | None = None
    committed: bool = False
