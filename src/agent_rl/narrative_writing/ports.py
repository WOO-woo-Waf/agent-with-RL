"""Ports for replacing narrative Agent policies and infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from agent_rl.core.concepts import Observation, Trajectory
from agent_rl.domains.narrative import (
    AuthorConversation,
    DraftBranch,
    ChapterBlueprint,
    ChapterAnalysisResult,
    ChapterPlan,
    ChunkAnalysisResult,
    DraftCandidate,
    DraftRepairPlan,
    DraftRevisionCandidate,
    EvaluationReport,
    EvidencePack,
    GlobalStoryAnalysisResult,
    NarrativeSourceAnalysis,
    NarrativeQuery,
    NarrativeEvidence,
    NarrativeTaskState,
    MemoryAtom,
    CompressedMemoryBlock,
    SourceChunk,
    SourceDocument,
    StateChangeProposal,
    WorkingMemoryContext,
)
from agent_rl.narrative_writing.requests import AuthorQuestion, AuthorRequest
from agent_rl.narrative_writing.requests import ReferenceMaterial


class AuthorInteractionPolicy(Protocol):
    """Decides whether the Agent has enough author context to proceed."""

    def missing_questions(self, request: AuthorRequest, state: NarrativeTaskState | None) -> list[AuthorQuestion]:
        ...


class NarrativeAnalysisPolicy(Protocol):
    """Analyzes source/reference material into reusable narrative assets."""

    def analyze(
        self,
        references: Sequence[ReferenceMaterial],
        *,
        task_id: str,
        story_id: str,
        goal: str,
        writing_direction: str = "",
    ) -> NarrativeSourceAnalysis:
        ...


class NarrativeRetrievalPolicy(Protocol):
    """Builds task-aware evidence for generation and validation."""

    def retrieve(self, state: NarrativeTaskState, query: NarrativeQuery) -> EvidencePack:
        ...


class NarrativeRetrievalEvaluationPolicy(Protocol):
    """Evaluates retrieved evidence coverage before generation."""

    def evaluate(self, evidence_pack: EvidencePack, query: NarrativeQuery) -> EvaluationReport:
        ...


class NarrativePlanningPolicy(Protocol):
    """Turns author intent and retrieved evidence into a runnable plan."""

    def propose_blueprint(self, state: NarrativeTaskState, request: AuthorRequest) -> ChapterBlueprint:
        ...

    def build_chapter_plan(
        self,
        state: NarrativeTaskState,
        blueprint: ChapterBlueprint,
        evidence_pack: EvidencePack,
        request: AuthorRequest,
    ) -> ChapterPlan:
        ...


class NarrativeContextPolicy(Protocol):
    """Builds a budgeted context manifest for generation and audit."""

    def build(
        self,
        state: NarrativeTaskState,
        evidence_pack: EvidencePack,
        plan: ChapterPlan,
        request: AuthorRequest,
    ) -> WorkingMemoryContext:
        ...


class NarrativeWriterPolicy(Protocol):
    """Generates a draft from plan, state, and evidence."""

    def generate(
        self,
        state: NarrativeTaskState,
        plan: ChapterPlan,
        evidence_pack: EvidencePack,
        working_context: WorkingMemoryContext | None = None,
    ) -> DraftCandidate:
        ...


class NarrativeExtractorPolicy(Protocol):
    """Extracts candidate state updates from draft text."""

    def extract(self, state: NarrativeTaskState, draft: DraftCandidate) -> list[StateChangeProposal]:
        ...


class NarrativeEvaluatorPolicy(Protocol):
    """Evaluates whether a draft and its proposed state changes can be committed."""

    def evaluate(
        self,
        state: NarrativeTaskState,
        draft: DraftCandidate,
        changes: Sequence[StateChangeProposal],
    ) -> list[EvaluationReport]:
        ...


class NarrativeRepairPolicy(Protocol):
    """Repairs blocked drafts before the agent gives up and rolls back."""

    def repair(
        self,
        state: NarrativeTaskState,
        draft: DraftCandidate,
        reports: Sequence[EvaluationReport],
        *,
        attempt_no: int,
    ) -> DraftRevisionCandidate:
        ...


class NarrativeBranchSelectionPolicy(Protocol):
    """Selects a draft branch when the author has not explicitly chosen one."""

    def select_branch(self, branches: Sequence[DraftBranch]) -> DraftBranch | None:
        ...


class NarrativeMemoryPolicy(Protocol):
    """Writes accepted changes into long-term and compressed memory."""

    def apply(self, state: NarrativeTaskState, changes: Sequence[StateChangeProposal]) -> NarrativeTaskState:
        ...


class NarrativeAnalysisRepository(Protocol):
    """Persists reusable source-analysis assets behind a replaceable boundary."""

    def save_source_analysis(self, analysis: NarrativeSourceAnalysis) -> None:
        ...

    def save_source_documents(self, *, story_id: str, task_id: str, documents: Sequence[SourceDocument]) -> None:
        ...

    def save_source_chunks(self, *, story_id: str, task_id: str, chunks: Sequence[SourceChunk]) -> None:
        ...

    def save_chunk_analyses(
        self,
        *,
        story_id: str,
        task_id: str,
        analyses: Sequence[ChunkAnalysisResult],
    ) -> None:
        ...

    def save_chapter_analyses(
        self,
        *,
        story_id: str,
        task_id: str,
        analyses: Sequence[ChapterAnalysisResult],
    ) -> None:
        ...

    def save_global_analysis(
        self,
        *,
        story_id: str,
        task_id: str,
        analysis: GlobalStoryAnalysisResult,
    ) -> None:
        ...

    def load_chunk_analyses(self, *, story_id: str, task_id: str) -> list[ChunkAnalysisResult]:
        ...

    def load_chapter_analyses(self, *, story_id: str, task_id: str) -> list[ChapterAnalysisResult]:
        ...

    def load_global_analysis(self, *, story_id: str, task_id: str) -> GlobalStoryAnalysisResult | None:
        ...

    def load_source_analysis(self, *, story_id: str, task_id: str) -> NarrativeSourceAnalysis | None:
        ...


class NarrativeStateRepository(Protocol):
    """Persists long-running narrative runtime state and artifacts."""

    def save_state_snapshot(self, state: NarrativeTaskState, *, run_id: str = "") -> Path:
        ...

    def load_state_snapshot(self, story_id: str, *, path: str | Path | None = None) -> NarrativeTaskState:
        ...

    def save_workflow_snapshot(self, story_id: str, workflow: object, *, run_id: str = "") -> Path:
        ...

    def save_trajectory(self, story_id: str, trajectory: Trajectory, *, run_id: str = "") -> Path:
        ...

    def save_blueprint(self, story_id: str, blueprint: ChapterBlueprint, *, chapter_index: int | None = None) -> Path:
        ...

    def save_draft(self, story_id: str, draft: DraftCandidate, *, chapter_index: int | None = None) -> Path:
        ...

    def save_branches(self, story_id: str, branches: Sequence[DraftBranch], *, run_id: str = "") -> list[Path]:
        ...

    def save_run_result(self, story_id: str, payload: dict[str, object], *, run_id: str = "") -> Path:
        ...

    def load_source_analysis(self, path: str | Path) -> NarrativeSourceAnalysis:
        ...

    def save_session_snapshot(
        self,
        *,
        session_id: str,
        state: NarrativeTaskState,
        workflow: object,
        trajectory: Trajectory,
        request: object,
        observation: Observation | None = None,
        memory_snapshot: dict[str, object] | None = None,
    ) -> Path:
        ...

    def load_session_snapshot(self, session_id: str, *, story_id: str = "") -> dict[str, object]:
        ...


class NarrativeConversationRepository(Protocol):
    """Persists author conversation and preference state."""

    def save_conversation(self, conversation: AuthorConversation) -> Path:
        ...

    def load_conversation(self, session_id: str, *, story_id: str = "") -> AuthorConversation | None:
        ...


class NarrativeBranchRepository(Protocol):
    """Persists candidate branches before author selection."""

    def save_branches(self, story_id: str, branches: Sequence[DraftBranch], *, run_id: str = "") -> list[Path]:
        ...


class NarrativeMemoryRepository(Protocol):
    """Persists and queries long-term narrative memory indexes."""

    def upsert_state_memory(self, state: NarrativeTaskState) -> None:
        ...

    def search(self, story_id: str, query: str, *, limit: int = 12) -> list[NarrativeEvidence]:
        ...

    def load_memory_atoms(self, story_id: str, *, include_deprecated: bool = False) -> list[MemoryAtom]:
        ...

    def load_compressed_memory(self, story_id: str) -> list[CompressedMemoryBlock]:
        ...

    def invalidate_memory_atoms(self, story_id: str, memory_ids: Sequence[str], *, reason: str = "") -> int:
        ...


class NarrativeEvaluationRepository(Protocol):
    """Persists evaluation reports for later policy comparison and audit."""

    def save_reports(self, story_id: str, reports: Sequence[EvaluationReport], *, run_id: str = "") -> list[Path]:
        ...


class NarrativeJobRepository(Protocol):
    """Persists and retrieves background narrative jobs."""

    def enqueue(self, job: Any) -> Path:
        ...

    def save(self, job: Any) -> Path:
        ...

    def load(self, job_id: str) -> Any:
        ...

    def next_queued(self) -> Any | None:
        ...
