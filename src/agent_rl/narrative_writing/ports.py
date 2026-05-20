"""Ports for replacing narrative Agent policies and infrastructure."""

from __future__ import annotations

from typing import Protocol, Sequence

from agent_rl.domains.narrative import (
    ChapterBlueprint,
    ChapterPlan,
    DraftCandidate,
    EvaluationReport,
    EvidencePack,
    NarrativeQuery,
    NarrativeTaskState,
    StateChangeProposal,
)
from agent_rl.narrative_writing.requests import AuthorQuestion, AuthorRequest


class AuthorInteractionPolicy(Protocol):
    """Decides whether the Agent has enough author context to proceed."""

    def missing_questions(self, request: AuthorRequest, state: NarrativeTaskState | None) -> list[AuthorQuestion]:
        ...


class NarrativeRetrievalPolicy(Protocol):
    """Builds task-aware evidence for generation and validation."""

    def retrieve(self, state: NarrativeTaskState, query: NarrativeQuery) -> EvidencePack:
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


class NarrativeWriterPolicy(Protocol):
    """Generates a draft from plan, state, and evidence."""

    def generate(
        self,
        state: NarrativeTaskState,
        plan: ChapterPlan,
        evidence_pack: EvidencePack,
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


class NarrativeMemoryPolicy(Protocol):
    """Writes accepted changes into long-term and compressed memory."""

    def apply(self, state: NarrativeTaskState, changes: Sequence[StateChangeProposal]) -> NarrativeTaskState:
        ...
