"""Stateful narrative-writing session for long-running author interaction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from agent_rl.core.concepts import AgentState, Decision, Goal, Observation, Trajectory, TrajectoryStep, utc_now
from agent_rl.core.memory import InMemoryStore
from agent_rl.domains.narrative import (
    AuthorConversation,
    AuthorDecision,
    AuthorMessage,
    AuthorPreferenceProfile,
    NarrativeTaskState,
)
from agent_rl.narrative_writing.agent import default_max_steps
from agent_rl.narrative_writing.longform_context import MemoryGovernancePolicy
from agent_rl.narrative_writing.persistence import FileNarrativeConversationRepository, FileNarrativeStateRepository
from agent_rl.narrative_writing.policies import BasicAuthorInteractionPolicy
from agent_rl.narrative_writing.ports import (
    AuthorInteractionPolicy,
    NarrativeConversationRepository,
    NarrativeEvaluationRepository,
    NarrativeMemoryRepository,
    NarrativeStateRepository,
)
from agent_rl.narrative_writing.react import (
    NarrativeAuthorLedPolicy,
    NarrativeReActEnvironment,
    narrative_goal_from_request,
)
from agent_rl.narrative_writing.requests import AuthorRequest, NarrativeRunResult
from agent_rl.narrative_writing.rag_index import NarrativeRAGIndexingService
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter
from agent_rl.narrative_writing.serialization import from_jsonable
from agent_rl.rag import RAGModelService


class NarrativeWritingSession:
    """Keeps a narrative ReAct environment alive across author turns.

    This is the package-level starting point for a Codex-like loop: the caller
    can step, pause for author confirmation, merge new author input, and keep
    running with the same task state and trajectory.
    """

    def __init__(
        self,
        request: AuthorRequest,
        *,
        scenario: NarrativeScenarioAdapter | None = None,
        state: NarrativeTaskState | None = None,
        interaction_policy: AuthorInteractionPolicy | None = None,
        state_repository: NarrativeStateRepository | None = None,
        conversation_repository: NarrativeConversationRepository | None = None,
        memory_repository: NarrativeMemoryRepository | None = None,
        evaluation_repository: NarrativeEvaluationRepository | None = None,
        rag_service: RAGModelService | None = None,
        rag_collection_id: str = "narrative",
        auto_rag_index: bool = False,
        rag_index_batch_size: int | None = None,
        conversation: AuthorConversation | None = None,
        goal: Goal | None = None,
        auto_checkpoint: bool = True,
    ) -> None:
        self.request = request
        self.session_id = request.session_id or f"session-{request.story_id}-{request.task_id}"
        self.goal = goal or narrative_goal_from_request(request)
        self.memory = InMemoryStore()
        self.policy = NarrativeAuthorLedPolicy()
        self.conversation_repository = conversation_repository or FileNarrativeConversationRepository()
        self.memory_repository = memory_repository
        self.evaluation_repository = evaluation_repository
        self.rag_service = rag_service
        self.rag_collection_id = rag_collection_id
        self.auto_rag_index = auto_rag_index
        self.rag_index_batch_size = rag_index_batch_size
        self.conversation = conversation or _new_conversation(self.session_id, request)
        if scenario is None and (memory_repository is not None or evaluation_repository is not None):
            scenario = NarrativeScenarioAdapter(
                memory_repository=memory_repository,
                evaluation_repository=evaluation_repository,
            )
        self.env = NarrativeReActEnvironment(
            request,
            scenario=scenario,
            task_state=state,
            interaction_policy=interaction_policy or BasicAuthorInteractionPolicy(),
            state_repository=state_repository,
        )
        self.trajectory = Trajectory(goal=self.goal)
        self.observation: Observation | None = None
        self.started = False
        self.closed = False
        self.auto_checkpoint = auto_checkpoint
        if not self.conversation.messages:
            self._record_author_message(request.request, metadata={"event": "session_started"})

    @property
    def state(self) -> NarrativeTaskState:
        return self.env.task_state

    @property
    def workflow_phase(self) -> str:
        return self.env.workflow.phase

    @property
    def outcome(self) -> str:
        return self.env.workflow.outcome

    def start(self, *, seed: int | None = None) -> Observation:
        if self.started:
            return self._require_observation()
        self.observation, reset_info = self.env.reset(seed=seed)
        self.trajectory.metadata["reset_info"] = dict(reset_info)
        self.started = True
        return self.observation

    def step(self) -> Decision:
        if self.closed:
            raise RuntimeError("session is closed")
        observation = self.start()
        actions = self.env.available_actions()
        state = AgentState(
            goal=self.goal,
            observation=observation,
            memory=self.memory,
            step_index=len(self.trajectory.steps),
        )
        decision = self.policy.select_action(state, actions)
        transition = self.env.step(decision.action)
        step = TrajectoryStep(
            index=len(self.trajectory.steps),
            observation=observation,
            action=decision.action,
            agent_id=decision.action.agent_id,
            rationale=decision.rationale,
            reward=transition.reward,
            next_observation=transition.next_observation,
            ended_at=utc_now(),
            metadata=dict(transition.info),
        )
        self.trajectory.append(step)
        self.memory.append("trajectory", step)
        if decision.action.name == "save_artifacts":
            path = self.env.state_repository.save_trajectory(self.state.story_id, self.trajectory, run_id=self.request.task_id)
            self.env.workflow.artifacts.append(str(path))
        if decision.action.name == "commit_state" and self.env.workflow.committed:
            self._auto_index_rag()
        self.observation = transition.next_observation
        if self.env.workflow.last_message:
            self._record_assistant_message(
                self.env.workflow.last_message,
                metadata={"phase": self.env.workflow.phase, "action": decision.action.name},
            )
        if transition.terminated:
            self.trajectory.outcome = str(transition.info.get("outcome") or "terminated")
        elif transition.truncated:
            self.trajectory.outcome = str(transition.info.get("outcome") or "truncated")
        else:
            self.trajectory.outcome = "running"
        self._checkpoint()
        return decision

    def run_until_pause(self, *, max_steps: int | None = None) -> NarrativeRunResult:
        limit = max_steps or default_max_steps(self.request)
        for _ in range(limit):
            if self.trajectory.outcome != "running" and self.trajectory.steps:
                break
            self.step()
            if self.trajectory.outcome != "running":
                break
        else:
            self.trajectory.outcome = "max_steps"
            self._checkpoint()
        return self.result()

    def apply_author_input(self, **changes: Any) -> AuthorRequest:
        """Merge author input into the live request and resume from a pause."""

        self.request = replace(self.request, **changes)
        self._record_author_change(changes)
        self.env.request = self.request
        self.env.workflow.request = self.request
        self.goal = narrative_goal_from_request(self.request)
        self.trajectory.goal = self.goal
        if self.env.workflow.phase == "blueprint_proposed" and _is_blueprint_revision(changes):
            self._prepare_blueprint_revision(changes)
        elif self.env.workflow.phase == "branch_selection_pending" and self.request.selected_branch_id:
            self.env.workflow.outcome = "running"
            self.trajectory.outcome = "running"
        elif self.env.workflow.phase in {"needs_author_input", "blueprint_proposed"}:
            self.env.workflow.outcome = "running"
            self.trajectory.outcome = "running"
            if self.env.workflow.phase == "needs_author_input":
                self.env.workflow.questions = self.env.interaction_policy.missing_questions(
                    self.request,
                    self.env.task_state if self.env._has_domain_state() else None,
                )
                if not self.env.workflow.questions:
                    self.env.workflow.phase = "initialized"
        self.observation = self.env._observation()
        self._checkpoint()
        return self.request

    def save(self) -> str:
        self.conversation_repository.save_conversation(self.conversation)
        path = self.env.state_repository.save_session_snapshot(
            session_id=self.session_id,
            state=self.env.task_state,
            workflow=self.env.workflow,
            trajectory=self.trajectory,
            request=self.request,
            observation=self.observation,
            memory_snapshot=self.memory.snapshot(),
        )
        return str(path)

    @classmethod
    def resume(
        cls,
        session_id: str,
        *,
        story_id: str = "",
        scenario: NarrativeScenarioAdapter | None = None,
        interaction_policy: AuthorInteractionPolicy | None = None,
        state_repository: NarrativeStateRepository | None = None,
        conversation_repository: NarrativeConversationRepository | None = None,
        memory_repository: NarrativeMemoryRepository | None = None,
        evaluation_repository: NarrativeEvaluationRepository | None = None,
        rag_service: RAGModelService | None = None,
        rag_collection_id: str = "narrative",
        auto_rag_index: bool = False,
        rag_index_batch_size: int | None = None,
    ) -> "NarrativeWritingSession":
        repository = state_repository or FileNarrativeStateRepository()
        conv_repository = conversation_repository or FileNarrativeConversationRepository()
        payload = repository.load_session_snapshot(session_id, story_id=story_id)
        request = from_jsonable(AuthorRequest, payload["request"])
        state = from_jsonable(NarrativeTaskState, payload["state"])
        conversation = conv_repository.load_conversation(session_id, story_id=story_id or request.story_id)
        session = cls(
            request,
            scenario=scenario,
            state=state,
            interaction_policy=interaction_policy,
            state_repository=repository,
            conversation_repository=conv_repository,
            memory_repository=memory_repository,
            evaluation_repository=evaluation_repository,
            rag_service=rag_service,
            rag_collection_id=rag_collection_id,
            auto_rag_index=auto_rag_index,
            rag_index_batch_size=rag_index_batch_size,
            conversation=conversation,
            auto_checkpoint=True,
        )
        session.session_id = str(payload.get("session_id") or session_id)
        session.env.workflow = from_jsonable(type(session.env.workflow), payload["workflow"])
        session.trajectory = from_jsonable(Trajectory, payload["trajectory"])
        raw_observation = payload.get("observation")
        session.observation = from_jsonable(Observation, raw_observation) if raw_observation else session.env._observation()
        _restore_memory(session.memory, payload.get("memory_snapshot", {}))
        session.started = True
        session.closed = False
        return session

    def result(self) -> NarrativeRunResult:
        return NarrativeRunResult(
            state=self.env.task_state,
            trajectory=self.trajectory,
            questions=self.env.workflow.questions,
            assistant_message=self.env.workflow.last_message,
            requires_confirmation=self.trajectory.outcome == "needs_confirmation",
            proposed_blueprint=self.env.workflow.proposed_blueprint,
            branches=list(self.env.workflow.branches),
            draft=self.env.workflow.draft,
            committed=self.env.workflow.committed,
        )

    def rollback(self, *, reason: str = "") -> NarrativeRunResult:
        self.env.workflow.committed = False
        self.env.workflow.phase = "rolled_back"
        self.env.workflow.outcome = "rolled_back"
        self.env.workflow.last_message = reason or "Session rolled back by author/operator."
        self.trajectory.outcome = "rolled_back"
        self._record_author_message(
            reason or "rollback",
            metadata={"event": "session_rollback"},
        )
        self._checkpoint()
        return self.result()

    def export_chapter(self, path: str) -> str:
        if self.env.workflow.draft is None:
            raise RuntimeError("no draft is available to export")
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.env.workflow.draft.content, encoding="utf-8")
        self.env.workflow.artifacts.append(str(target))
        self._checkpoint()
        return str(target)

    def invalidate_memory(
        self,
        *,
        text: str = "",
        memory_ids: Sequence[str] = (),
        reason: str = "",
    ) -> list[str]:
        invalidated = list(memory_ids)
        requested_ids = {memory_id for memory_id in memory_ids if memory_id}
        for atom in self.state.memory_atoms:
            if atom.memory_id in requested_ids and atom.status != "deprecated":
                atom.canonical = False
                atom.status = "deprecated"
                atom.invalidation_reason = reason or "invalidated by author/operator"
        if text:
            invalidated.extend(MemoryGovernancePolicy().invalidate_by_text(self.state, text, reason=reason))
        unique_ids = sorted({memory_id for memory_id in invalidated if memory_id})
        if self.memory_repository is not None:
            self.memory_repository.invalidate_memory_atoms(self.state.story_id, unique_ids, reason=reason)
            self.memory_repository.upsert_state_memory(self.state)
        if unique_ids:
            self._record_author_message(
                reason or f"invalidate memory: {', '.join(unique_ids)}",
                metadata={"event": "memory_invalidated", "memory_ids": unique_ids, "text": text},
            )
        self._checkpoint()
        return unique_ids

    def index_rag(self, rag_service: RAGModelService, *, collection_id: str = "narrative", batch_size: int | None = None) -> int:
        count = NarrativeRAGIndexingService(rag_service, collection_id=collection_id).index_state(self.state, batch_size=batch_size)
        self.state.metadata["rag_index"] = {
            "collection_id": collection_id,
            "indexed_count": count,
            "service": rag_service.__class__.__name__,
            "mode": "manual",
        }
        self._checkpoint()
        return count

    def _auto_index_rag(self) -> None:
        if not self.auto_rag_index or self.rag_service is None:
            return
        try:
            count = NarrativeRAGIndexingService(self.rag_service, collection_id=self.rag_collection_id).index_state(
                self.state,
                batch_size=self.rag_index_batch_size,
            )
        except Exception as exc:  # noqa: BLE001 - commit must not be rolled back by optional indexing.
            self.state.metadata["rag_index"] = {
                "collection_id": self.rag_collection_id,
                "status": "failed",
                "error": str(exc),
                "mode": "auto_on_commit",
            }
            return
        self.state.metadata["rag_index"] = {
            "collection_id": self.rag_collection_id,
            "indexed_count": count,
            "service": self.rag_service.__class__.__name__,
            "mode": "auto_on_commit",
            "status": "succeeded",
        }

    def close(self) -> None:
        if not self.closed:
            self.env.close()
            self.closed = True

    def _require_observation(self) -> Observation:
        if self.observation is None:
            raise RuntimeError("session has not been started")
        return self.observation

    def _record_author_message(self, content: str, *, metadata: dict[str, Any] | None = None) -> None:
        text = str(content or "").strip()
        if not text:
            return
        message = AuthorMessage(
            message_id=f"msg-{len(self.conversation.messages) + 1:04d}",
            role="author",
            content=text,
            created_at=utc_now().isoformat(),
            metadata=dict(metadata or {}),
        )
        self.conversation.messages.append(message)
        _update_preferences(self.conversation, message)

    def _record_assistant_message(self, content: str, *, metadata: dict[str, Any] | None = None) -> None:
        text = str(content or "").strip()
        if not text:
            return
        if self.conversation.messages and self.conversation.messages[-1].role == "assistant" and self.conversation.messages[-1].content == text:
            return
        self.conversation.messages.append(
            AuthorMessage(
                message_id=f"msg-{len(self.conversation.messages) + 1:04d}",
                role="assistant",
                content=text,
                created_at=utc_now().isoformat(),
                metadata=dict(metadata or {}),
            )
        )

    def _record_author_change(self, changes: dict[str, Any]) -> None:
        meaningful = {key: value for key, value in changes.items() if value not in (None, "", (), [])}
        if not meaningful:
            return
        summary_parts = [f"{key}={value}" for key, value in sorted(meaningful.items())]
        self._record_author_message("; ".join(summary_parts), metadata={"event": "author_input_applied", "changes": meaningful})
        if meaningful.get("confirm_plan") is True:
            self.conversation.decisions.append(
                AuthorDecision(
                    decision_id=f"decision-{len(self.conversation.decisions) + 1:04d}",
                    decision_type="confirm",
                    summary="Author confirmed current blueprint.",
                    target_type="blueprint",
                    target_id=self.env.workflow.proposed_blueprint.blueprint_id if self.env.workflow.proposed_blueprint else "",
                    confirmed=True,
                    created_at=utc_now().isoformat(),
                    metadata={"changes": meaningful},
                )
            )
        elif "writing_direction" in meaningful or "constraints" in meaningful:
            self.conversation.decisions.append(
                AuthorDecision(
                    decision_id=f"decision-{len(self.conversation.decisions) + 1:04d}",
                    decision_type="revise",
                    summary="Author revised planning input.",
                    target_type="request",
                    created_at=utc_now().isoformat(),
                    metadata={"changes": meaningful},
                )
            )
        if meaningful.get("selected_branch_id"):
            self.conversation.decisions.append(
                AuthorDecision(
                    decision_id=f"decision-{len(self.conversation.decisions) + 1:04d}",
                    decision_type="select_branch",
                    summary="Author selected a draft branch.",
                    target_type="draft_branch",
                    target_id=str(meaningful["selected_branch_id"]),
                    confirmed=True,
                    created_at=utc_now().isoformat(),
                    metadata={"changes": meaningful},
                )
            )

    def _prepare_blueprint_revision(self, changes: dict[str, Any]) -> None:
        old = self.env.workflow.proposed_blueprint
        history = self.env.task_state.metadata.setdefault("blueprint_revision_history", [])
        if old is not None:
            history.append(
                {
                    "blueprint_id": old.blueprint_id,
                    "revision_no": old.revision_no,
                    "feedback": str(changes.get("blueprint_feedback") or ""),
                    "created_at": utc_now().isoformat(),
                }
            )
        self.env.workflow.proposed_blueprint = None
        self.env.workflow.evidence_pack = None
        self.env.workflow.working_context = None
        self.env.workflow.draft = None
        self.env.workflow.draft_segments = []
        self.env.workflow.branches = []
        self.env.workflow.pending_changes = []
        self.env.workflow.reports = []
        self.env.workflow.current_segment_index = 0
        self.env.workflow.phase = "analysis_ready"
        self.env.workflow.outcome = "running"
        self.env.workflow.last_message = "Blueprint revision requested; generating a new blueprint version next."
        self.trajectory.outcome = "running"

    def _checkpoint(self) -> None:
        if self.auto_checkpoint:
            self.save()


def _new_conversation(session_id: str, request: AuthorRequest) -> AuthorConversation:
    return AuthorConversation(
        conversation_id=f"conversation-{session_id}",
        session_id=session_id,
        story_id=request.story_id,
        task_id=request.task_id,
        preference_profile=AuthorPreferenceProfile(
            profile_id=f"author-preferences-{request.story_id}",
            story_id=request.story_id,
        ),
    )


def _is_blueprint_revision(changes: dict[str, Any]) -> bool:
    if changes.get("confirm_plan") is True:
        return False
    return any(key in changes for key in ("blueprint_feedback", "writing_direction", "constraints"))


def _restore_memory(memory: InMemoryStore, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if key == "trajectory" and isinstance(value, list):
            memory.set(key, [from_jsonable(TrajectoryStep, item) for item in value])
        else:
            memory.set(str(key), value)


def _update_preferences(conversation: AuthorConversation, message: AuthorMessage) -> None:
    profile = conversation.preference_profile
    if profile is None:
        profile = AuthorPreferenceProfile(profile_id=f"author-preferences-{conversation.story_id}", story_id=conversation.story_id)
        conversation.preference_profile = profile
    text = message.content
    lowered = text.lower()
    target: list[str] | None = None
    if any(marker in text for marker in ("不要", "不能", "避免", "禁")) or "do not" in lowered:
        target = profile.taboo_patterns
    elif any(marker in text for marker in ("对话", "说话", "dialogue")):
        target = profile.dialogue_preferences
    elif any(marker in text for marker in ("节奏", "慢", "快", "pacing")):
        target = profile.pacing_preferences
    elif any(marker in text for marker in ("风格", "文风", "style")):
        target = profile.style_preferences
    elif message.metadata.get("event") == "author_input_applied":
        target = profile.revision_preferences
    if target is not None and text not in target:
        target.append(text)
        profile.evidence_message_ids.append(message.message_id)
