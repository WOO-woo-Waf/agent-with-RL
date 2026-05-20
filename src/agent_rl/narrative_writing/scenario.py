"""Scenario adapter for the narrative-writing domain."""

from __future__ import annotations

from typing import Sequence

from agent_rl.core import Action, Observation
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
from agent_rl.narrative_writing.bootstrap import build_initial_state
from agent_rl.narrative_writing.policies import (
    CompositeNarrativeEvaluatorPolicy,
    KeywordNarrativeRetrievalPolicy,
    RuleBasedExtractorPolicy,
    RuleBasedPlanningPolicy,
    SimpleNarrativeMemoryPolicy,
    TemplateNarrativeWriterPolicy,
)
from agent_rl.narrative_writing.ports import (
    NarrativeEvaluatorPolicy,
    NarrativeExtractorPolicy,
    NarrativeMemoryPolicy,
    NarrativePlanningPolicy,
    NarrativeRetrievalPolicy,
    NarrativeWriterPolicy,
)
from agent_rl.narrative_writing.requests import AuthorRequest
from agent_rl.narrative_writing.utils import new_id


class NarrativeScenarioAdapter:
    """Exposes novel-writing capability to a generic Agent runtime."""

    scenario_type = "narrative_writing"

    def __init__(
        self,
        retrieval_policy: NarrativeRetrievalPolicy | None = None,
        planning_policy: NarrativePlanningPolicy | None = None,
        writer_policy: NarrativeWriterPolicy | None = None,
        extractor_policy: NarrativeExtractorPolicy | None = None,
        evaluator_policy: NarrativeEvaluatorPolicy | None = None,
        memory_policy: NarrativeMemoryPolicy | None = None,
    ) -> None:
        self.retrieval_policy = retrieval_policy or KeywordNarrativeRetrievalPolicy()
        self.planning_policy = planning_policy or RuleBasedPlanningPolicy()
        self.writer_policy = writer_policy or TemplateNarrativeWriterPolicy()
        self.extractor_policy = extractor_policy or RuleBasedExtractorPolicy()
        self.evaluator_policy = evaluator_policy or CompositeNarrativeEvaluatorPolicy()
        self.memory_policy = memory_policy or SimpleNarrativeMemoryPolicy()

    def build_initial_state(self, request: AuthorRequest) -> NarrativeTaskState:
        return build_initial_state(request)

    def build_observation(self, state: NarrativeTaskState) -> Observation:
        return Observation(
            payload={
                "task_id": state.task_id,
                "story_id": state.story_id,
                "goal": state.goal,
                "state_version_no": state.state_version_no,
                "characters": [character.name for character in state.characters],
                "plot_threads": [thread.name for thread in state.plot_threads],
                "author_constraints": [constraint.text for constraint in state.author_constraints],
                "memory_blocks": len(state.compressed_memory),
                "reports": [report.report_type for report in state.reports],
            },
            source=self.scenario_type,
        )

    def list_actions(self, state: NarrativeTaskState) -> list[Action]:
        return [
            Action("retrieve_context", kind="tool"),
            Action("propose_plan", kind="tool"),
            Action("generate_draft", kind="tool"),
            Action("extract_changes", kind="tool"),
            Action("evaluate", kind="tool"),
            Action("commit_or_rollback", kind="tool"),
            Action("ask_author", kind="control"),
        ]

    def build_query(self, state: NarrativeTaskState, request: AuthorRequest) -> NarrativeQuery:
        query_parts = [
            request.request,
            request.writing_direction,
            " ".join(constraint.text for constraint in state.author_constraints),
            " ".join(thread.name for thread in state.plot_threads),
            " ".join(character.name for character in state.characters),
        ]
        return NarrativeQuery(
            query_id=new_id("query"),
            query_text=" ".join(part for part in query_parts if part),
            query_type="chapter_continuation",
            target_chapter_index=request.target_chapter_index,
            involved_character_ids=[character.character_id for character in state.characters[:3]],
            plot_thread_ids=[thread.thread_id for thread in state.plot_threads[:3]],
            required_evidence_types=[
                "author_constraint",
                "compressed_memory",
                "character_profile",
                "plot_thread",
                "world_rule",
                "style_snippet",
            ],
        )

    def retrieve_context(self, state: NarrativeTaskState, query: NarrativeQuery) -> EvidencePack:
        pack = self.retrieval_policy.retrieve(state, query)
        state.evidence_pack = pack
        return pack

    def propose_plan(self, state: NarrativeTaskState, request: AuthorRequest) -> ChapterBlueprint:
        blueprint = self.planning_policy.propose_blueprint(state, request)
        state.chapter_blueprints.append(blueprint)
        state.metadata["pending_blueprint_id"] = blueprint.blueprint_id
        return blueprint

    def build_chapter_plan(
        self,
        state: NarrativeTaskState,
        blueprint: ChapterBlueprint,
        evidence_pack: EvidencePack,
        request: AuthorRequest,
    ) -> ChapterPlan:
        plan = self.planning_policy.build_chapter_plan(state, blueprint, evidence_pack, request)
        state.chapter_plan = plan
        return plan

    def generate_draft(self, state: NarrativeTaskState, plan: ChapterPlan, evidence_pack: EvidencePack) -> DraftCandidate:
        draft = self.writer_policy.generate(state, plan, evidence_pack)
        state.draft = draft
        return draft

    def extract_changes(self, state: NarrativeTaskState, draft: DraftCandidate) -> list[StateChangeProposal]:
        changes = self.extractor_policy.extract(state, draft)
        state.pending_changes = list(changes)
        return changes

    def evaluate(
        self,
        state: NarrativeTaskState,
        draft: DraftCandidate,
        changes: Sequence[StateChangeProposal],
    ) -> list[EvaluationReport]:
        reports = self.evaluator_policy.evaluate(state, draft, changes)
        state.reports = list(reports)
        return reports

    def commit_or_rollback(
        self,
        state: NarrativeTaskState,
        changes: Sequence[StateChangeProposal],
        reports: Sequence[EvaluationReport],
    ) -> bool:
        blocking = [report for report in reports if report.blocks_commit]
        if blocking:
            state.metadata["commit_status"] = "rolled_back"
            state.metadata["rollback_reason"] = [issue.summary for report in blocking for issue in report.issues]
            return False
        self.memory_policy.apply(state, changes)
        state.metadata["commit_status"] = "committed"
        state.pending_changes = []
        return True
