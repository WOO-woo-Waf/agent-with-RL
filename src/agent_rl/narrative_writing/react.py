"""ReAct-style environment for author-led narrative writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from agent_rl.core.concepts import Action, AgentState, Decision, Goal, Observation, Reward, Transition
from agent_rl.domains.narrative import (
    ChapterBlueprint,
    ChapterPlan,
    DraftCandidate,
    EvaluationReport,
    EvidencePack,
    NarrativeTaskState,
    StateChangeProposal,
    WorkingMemoryContext,
)
from agent_rl.narrative_writing.policies import BasicAuthorInteractionPolicy
from agent_rl.narrative_writing.ports import AuthorInteractionPolicy
from agent_rl.narrative_writing.requests import AuthorQuestion, AuthorRequest
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter
from agent_rl.narrative_writing.persistence import FileNarrativeStateRepository
from agent_rl.narrative_writing.tools import (
    DraftCompressionTool,
    LoadAnalysisTool,
    SaveNarrativeArtifactsTool,
    ScanWorkspaceTool,
)

NarrativeWorkflowPhase = Literal[
    "initialized",
    "workspace_observed",
    "needs_author_input",
    "analysis_ready",
    "blueprint_proposed",
    "blueprint_confirmed",
    "context_ready",
    "draft_ready",
    "evaluated",
    "compressed",
    "committed",
    "artifacts_saved",
    "rolled_back",
    "completed",
]


@dataclass
class NarrativeWorkflowState:
    """Runtime workflow state around the canonical narrative task state."""

    request: AuthorRequest
    phase: NarrativeWorkflowPhase = "initialized"
    questions: list[AuthorQuestion] = field(default_factory=list)
    proposed_blueprint: ChapterBlueprint | None = None
    evidence_pack: EvidencePack | None = None
    working_context: WorkingMemoryContext | None = None
    draft: DraftCandidate | None = None
    draft_segments: list[DraftCandidate] = field(default_factory=list)
    pending_changes: list[StateChangeProposal] = field(default_factory=list)
    reports: list[EvaluationReport] = field(default_factory=list)
    committed: bool = False
    outcome: str = "running"
    last_message: str = ""
    last_tool_result: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    current_segment_index: int = 0
    action_history: list[str] = field(default_factory=list)


class NarrativeReActEnvironment:
    """Environment wrapper that exposes narrative writing as observe-decide-act steps."""

    scenario_type = "narrative_react_writing"

    def __init__(
        self,
        request: AuthorRequest,
        *,
        scenario: NarrativeScenarioAdapter | None = None,
        task_state: NarrativeTaskState | None = None,
        interaction_policy: AuthorInteractionPolicy | None = None,
        state_repository: FileNarrativeStateRepository | None = None,
    ) -> None:
        self.request = request
        self.scenario = scenario or NarrativeScenarioAdapter()
        self.state_repository = state_repository or FileNarrativeStateRepository(request.artifact_root or "artifacts/narrative-state")
        self.task_state = task_state or NarrativeTaskState(
            task_id=request.task_id,
            story_id=request.story_id,
            goal=request.request,
        )
        self.workflow = NarrativeWorkflowState(request=request)
        self.interaction_policy = interaction_policy or BasicAuthorInteractionPolicy()
        self.load_analysis_tool = LoadAnalysisTool(self.state_repository)
        self.save_artifacts_tool = SaveNarrativeArtifactsTool(self.state_repository)
        self.scan_workspace_tool = ScanWorkspaceTool()
        self.compression_tool = DraftCompressionTool()
        self._closed = False

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        self.workflow.phase = "initialized"
        self.workflow.outcome = "running"
        self.workflow.questions = self.interaction_policy.missing_questions(
            self.request,
            self.task_state if self._has_domain_state() else None,
        )
        if self.workflow.questions:
            self.workflow.phase = "needs_author_input"
        return self._observation(), {"scenario_type": self.scenario_type, "seed": seed}

    def available_actions(self) -> Sequence[Action]:
        phase = self.workflow.phase
        if phase == "needs_author_input":
            return (Action("ask_author", kind="control"),)
        if phase == "initialized":
            return (Action("scan_workspace", kind="tool"),)
        if phase == "workspace_observed":
            if self.request.state_snapshot_path and not self._has_domain_state():
                return (Action("load_state_snapshot", kind="tool"),)
            if self.request.analysis_path and not self._has_domain_state():
                return (Action("load_analysis", kind="tool"),)
            return (Action("analyze_source", kind="tool"),)
        if phase == "analysis_ready":
            return (Action("propose_blueprint", kind="planning"),)
        if phase == "blueprint_proposed":
            if self.request.confirm_plan:
                return (Action("confirm_blueprint", kind="planning"),)
            return (Action("wait_for_confirmation", kind="control"),)
        if phase == "blueprint_confirmed":
            return (Action("retrieve_context", kind="tool"),)
        if phase == "context_ready":
            if self.task_state.working_context is None:
                return (Action("build_working_context", kind="tool"),)
            if self._should_write_segments():
                if self.workflow.current_segment_index < len(self._require_blueprint().segments):
                    return (Action("generate_segment", kind="writing"),)
                if self.workflow.draft_segments and self.workflow.draft is None:
                    return (Action("merge_draft_segments", kind="writing"),)
            return (Action("generate_draft", kind="writing"),)
        if phase == "draft_ready":
            return (Action("evaluate_draft", kind="writing"),)
        if phase == "evaluated":
            if any(report.blocks_commit for report in self.workflow.reports):
                return (Action("rollback", kind="writing"),)
            return (Action("compress_new_draft", kind="tool"),)
        if phase == "compressed":
            return (Action("commit_state", kind="writing"),)
        if phase == "committed" and self.request.persist_artifacts:
            return (Action("save_artifacts", kind="tool"),)
        return (Action("stop", kind="control"),)

    def step(self, action: Action) -> Transition:
        previous = self._observation()
        self.workflow.action_history.append(action.name)
        info: dict[str, Any] = {}
        reward = Reward(0.0)
        terminated = False

        if action.name == "ask_author":
            self.workflow.outcome = "needs_author_input"
            self.workflow.last_message = "我需要先补齐参考材料和写作方向，才能安全规划和续写。"
            reward = Reward(-0.1, {"needs_author_input": 1.0})
            terminated = True
        elif action.name == "scan_workspace":
            result = self.scan_workspace_tool.invoke(
                analysis_path=self.request.analysis_path,
                state_snapshot_path=self.request.state_snapshot_path,
                artifact_root=self.request.artifact_root,
            )
            self.workflow.phase = "workspace_observed"
            self.workflow.last_message = "Workspace observed."
            self.workflow.last_tool_result = {"tool_name": result.tool_name, **dict(result.payload)}
            reward = Reward(0.05, {"workspace_observed": 1.0})
        elif action.name == "load_state_snapshot":
            self.task_state = self.state_repository.load_state_snapshot(
                self.request.story_id,
                path=self.request.state_snapshot_path,
            )
            self.workflow.phase = "analysis_ready"
            self.workflow.last_message = "Loaded narrative state snapshot."
            self.workflow.last_tool_result = {
                "tool_name": "load_state_snapshot",
                "state_version_no": self.task_state.state_version_no,
            }
            reward = Reward(0.1, {"state_snapshot_loaded": 1.0})
        elif action.name == "load_analysis":
            self.task_state, result = self.load_analysis_tool.invoke(
                source_analysis_path=self.request.analysis_path,
                request=self.request,
            )
            self.workflow.phase = "analysis_ready"
            self.workflow.last_message = "Loaded analysis artifacts into narrative state."
            self.workflow.last_tool_result = result.payload | {
                "tool_name": result.tool_name,
                "metrics": dict(result.metrics),
                "artifacts": list(result.artifacts),
            }
            reward = Reward(0.1, {"analysis_loaded": 1.0, **result.metrics})
        elif action.name == "analyze_source":
            if not self._has_domain_state():
                self.task_state = self.scenario.build_initial_state(self.request)
            self.workflow.phase = "analysis_ready"
            self.workflow.last_message = "已建立小说任务状态，可以生成章节蓝图。"
            self.workflow.last_tool_result = {
                "tool_name": "analyze_source",
                "source_chunks_count": len(self.task_state.source_chunks),
                "characters_count": len(self.task_state.characters),
                "plot_threads_count": len(self.task_state.plot_threads),
            }
            reward = Reward(0.1, {"analysis_ready": 1.0})
        elif action.name == "propose_blueprint":
            blueprint = self.scenario.propose_plan(self.task_state, self.request)
            self.workflow.proposed_blueprint = blueprint
            self.workflow.phase = "blueprint_proposed"
            self.workflow.last_message = _blueprint_confirmation_message(blueprint)
            self.workflow.last_tool_result = {
                "tool_name": "propose_blueprint",
                "blueprint_id": blueprint.blueprint_id,
                "segment_count": len(blueprint.segments),
                "target_total_chars": blueprint.target_total_chars,
            }
            reward = Reward(0.1, {"plan_created": 1.0})
        elif action.name == "wait_for_confirmation":
            self.workflow.outcome = "needs_confirmation"
            self.workflow.last_message = _blueprint_confirmation_message(self._require_blueprint())
            reward = Reward(0.0, {"needs_confirmation": 1.0})
            terminated = True
        elif action.name == "confirm_blueprint":
            blueprint = self._require_blueprint()
            blueprint.confirmed = True
            self.workflow.phase = "blueprint_confirmed"
            self.workflow.last_message = "章节蓝图已确认，准备检索上下文。"
            self.workflow.last_tool_result = {"tool_name": "confirm_blueprint", "blueprint_id": blueprint.blueprint_id}
            reward = Reward(0.1, {"blueprint_confirmed": 1.0})
        elif action.name == "retrieve_context":
            query = self.scenario.build_query(self.task_state, self.request)
            pack = self.scenario.retrieve_context(self.task_state, query)
            self.workflow.evidence_pack = pack
            self.workflow.phase = "context_ready"
            self.workflow.last_message = "已检索人物、剧情、世界和风格证据。"
            self.workflow.last_tool_result = {
                "tool_name": "retrieve_context",
                "evidence_count": len(pack.all_evidence()),
                "trace": list(pack.retrieval_trace[-2:]),
            }
            reward = Reward(0.1, {"evidence_count": float(len(pack.all_evidence()))})
        elif action.name == "build_working_context":
            pack = self._require_evidence_pack()
            blueprint = self._require_blueprint()
            plan = self.scenario.build_chapter_plan(self.task_state, blueprint, pack, self.request)
            context = self.scenario.build_working_context(self.task_state, plan, pack, self.request)
            self.workflow.working_context = context
            self.workflow.phase = "context_ready"
            self.workflow.last_message = "已装配写作上下文。"
            self.workflow.last_tool_result = {
                "tool_name": "build_working_context",
                "section_count": len(context.sections),
                "estimated_tokens": context.estimated_tokens,
                "longform_layers": dict(context.metadata.get("longform_layers", {})),
            }
            reward = Reward(0.1, {"context_sections": float(len(context.sections))})
        elif action.name == "generate_draft":
            pack = self._require_evidence_pack()
            blueprint = self._require_blueprint()
            plan = self.task_state.chapter_plan or self.scenario.build_chapter_plan(
                self.task_state,
                blueprint,
                pack,
                self.request,
            )
            context = self.task_state.working_context or self.scenario.build_working_context(
                self.task_state,
                plan,
                pack,
                self.request,
            )
            draft = self.scenario.generate_draft(self.task_state, plan, pack, context)
            self.workflow.draft = draft
            self.workflow.working_context = context
            self.workflow.phase = "draft_ready"
            self.workflow.last_message = "已生成草稿，准备评估。"
            self.workflow.last_tool_result = {
                "tool_name": "generate_draft",
                "draft_id": draft.draft_id,
                "draft_chars": len(draft.content),
                "writer_policy": draft.metadata.get("writer_policy", ""),
            }
            reward = Reward(0.2, {"draft_created": 1.0, "draft_chars": float(len(draft.content))})
        elif action.name == "generate_segment":
            pack = self._require_evidence_pack()
            blueprint = self._require_blueprint()
            context = self.task_state.working_context
            if context is None:
                plan = self.task_state.chapter_plan or self.scenario.build_chapter_plan(self.task_state, blueprint, pack, self.request)
                context = self.scenario.build_working_context(self.task_state, plan, pack, self.request)
            segment = blueprint.segments[self.workflow.current_segment_index]
            segment_plan = ChapterPlan(
                plan_id=f"{self.task_state.chapter_plan.plan_id if self.task_state.chapter_plan else blueprint.blueprint_id}:{segment.segment_id}",
                chapter_index=blueprint.chapter_index,
                objective=segment.goal,
                source_blueprint_id=blueprint.blueprint_id,
                target_word_count=segment.target_chars,
                required_beats=list(segment.required_beats or [segment.goal]),
                scene_plan_ids=[segment.segment_id],
                continuity_must_keep=[
                    segment.entry_state,
                    segment.exit_state,
                    *list(self.task_state.chapter_plan.continuity_must_keep if self.task_state.chapter_plan else []),
                ],
                completion_criteria={
                    "segment_id": segment.segment_id,
                    "target_chars": segment.target_chars,
                    "forbidden_beats": list(segment.forbidden_beats),
                },
            )
            draft = self.scenario.generate_draft(self.task_state, segment_plan, pack, context)
            draft.metadata["segment_id"] = segment.segment_id
            draft.metadata["segment_index"] = self.workflow.current_segment_index
            self.workflow.draft_segments.append(draft)
            self.workflow.current_segment_index += 1
            self.workflow.working_context = context
            self.workflow.phase = "context_ready"
            self.workflow.last_message = "Generated draft segment."
            self.workflow.last_tool_result = {
                "tool_name": "generate_segment",
                "segment_id": segment.segment_id,
                "segment_index": self.workflow.current_segment_index - 1,
                "draft_chars": len(draft.content),
                "remaining_segments": max(0, len(blueprint.segments) - self.workflow.current_segment_index),
            }
            reward = Reward(0.2, {"segment_created": 1.0, "draft_chars": float(len(draft.content))})
        elif action.name == "merge_draft_segments":
            blueprint = self._require_blueprint()
            content = "\n\n".join(segment.content for segment in self.workflow.draft_segments if segment.content.strip())
            planned_beats = [beat for segment in self.workflow.draft_segments for beat in segment.planned_beat_ids]
            style_targets = [item for segment in self.workflow.draft_segments for item in segment.style_targets]
            continuity_notes = [item for segment in self.workflow.draft_segments for item in segment.continuity_notes]
            draft = DraftCandidate(
                draft_id=f"draft-{blueprint.blueprint_id}",
                content=content,
                planned_beat_ids=planned_beats,
                style_targets=style_targets,
                continuity_notes=continuity_notes,
                metadata={
                    "writer_policy": "SegmentedNarrativeWriter",
                    "segment_count": len(self.workflow.draft_segments),
                    "source_blueprint_id": blueprint.blueprint_id,
                },
            )
            self.task_state.draft = draft
            self.workflow.draft = draft
            self.workflow.phase = "draft_ready"
            self.workflow.last_message = "Merged draft segments."
            self.workflow.last_tool_result = {
                "tool_name": "merge_draft_segments",
                "draft_id": draft.draft_id,
                "draft_chars": len(draft.content),
                "segment_count": len(self.workflow.draft_segments),
            }
            reward = Reward(0.15, {"draft_created": 1.0, "segment_count": float(len(self.workflow.draft_segments))})
        elif action.name == "evaluate_draft":
            draft = self._require_draft()
            changes = self.scenario.extract_changes(self.task_state, draft)
            reports = self.scenario.evaluate(self.task_state, draft, changes)
            self.workflow.pending_changes = list(changes)
            self.workflow.reports = list(reports)
            self.workflow.phase = "evaluated"
            self.workflow.last_message = "已完成草稿抽取和评估。"
            self.workflow.last_tool_result = {
                "tool_name": "evaluate_draft",
                "change_count": len(changes),
                "blocking_reports": sum(1 for report in reports if report.blocks_commit),
                "average_report_score": _average_report_score(reports),
            }
            reward = _reward_from_reports(reports, committed=False)
        elif action.name == "compress_new_draft":
            result = self.compression_tool.compress(self.task_state, self.workflow.pending_changes)
            self.workflow.phase = "compressed"
            self.workflow.last_message = "Draft changes compressed into pending memory signals."
            self.workflow.last_tool_result = {"tool_name": "compress_new_draft", **result}
            reward = Reward(0.1, {"memory_atoms_pending": float(result["memory_atoms_pending"])})
        elif action.name == "commit_state":
            committed = self.scenario.commit_or_rollback(
                self.task_state,
                self.workflow.pending_changes,
                self.workflow.reports,
            )
            self.workflow.committed = committed
            self.workflow.phase = "committed" if committed else "rolled_back"
            self.workflow.outcome = "committed" if committed else "rolled_back"
            self.workflow.last_message = _final_message(committed, self.workflow.reports)
            self.workflow.last_tool_result = {
                "tool_name": "commit_state",
                "committed": committed,
                "state_version_no": self.task_state.state_version_no,
                "memory_atoms_count": len(self.task_state.memory_atoms),
                "compressed_memory_count": len(self.task_state.compressed_memory),
            }
            reward = _reward_from_reports(self.workflow.reports, committed=committed)
            terminated = not self.request.persist_artifacts
        elif action.name == "save_artifacts":
            result = self.save_artifacts_tool.invoke(
                state=self.task_state,
                workflow=self.workflow,
                run_id=self.request.task_id,
            )
            self.workflow.phase = "artifacts_saved"
            self.workflow.outcome = "committed" if self.workflow.committed else self.workflow.outcome
            self.workflow.artifacts.extend(result.artifacts)
            self.workflow.last_message = "Narrative artifacts saved."
            self.workflow.last_tool_result = {
                "tool_name": result.tool_name,
                "artifact_count": len(result.artifacts),
                "artifacts": list(result.artifacts),
            }
            reward = Reward(0.1, {"artifact_count": float(len(result.artifacts))})
            terminated = True
        elif action.name == "rollback":
            self.workflow.committed = False
            self.workflow.phase = "rolled_back"
            self.workflow.outcome = "rolled_back"
            self.workflow.last_message = _final_message(False, self.workflow.reports)
            reward = _reward_from_reports(self.workflow.reports, committed=False)
            terminated = True
        elif action.name == "stop":
            self.workflow.outcome = self.workflow.outcome if self.workflow.outcome != "running" else "stopped"
            terminated = True
        else:
            self.workflow.outcome = "invalid_action"
            self.workflow.last_message = f"Unsupported narrative action: {action.name}"
            reward = Reward(-1.0, {"invalid_action": 1.0})
            terminated = True

        next_observation = self._observation()
        info["phase"] = self.workflow.phase
        info["outcome"] = self.workflow.outcome
        info["message"] = self.workflow.last_message
        return Transition(
            observation=previous,
            action=action,
            next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            info=info,
        )

    def close(self) -> None:
        self._closed = True

    def _observation(self) -> Observation:
        available_action_names = [action.name for action in self.available_actions()] if not self._closed else []
        draft_chars = len(self.task_state.draft.content) if self.task_state.draft is not None else 0
        payload = {
            "phase": self.workflow.phase,
            "story_id": self.task_state.story_id,
            "task_id": self.task_state.task_id,
            "goal": self.task_state.goal,
            "has_analysis": bool(self.task_state.source_analyses),
            "has_blueprint": self.workflow.proposed_blueprint is not None,
            "blueprint_confirmed": bool(self.workflow.proposed_blueprint and self.workflow.proposed_blueprint.confirmed),
            "has_evidence": self.task_state.evidence_pack is not None,
            "has_working_context": self.task_state.working_context is not None,
            "has_draft": self.task_state.draft is not None,
            "draft_char_count": draft_chars,
            "draft_segment_count": len(self.workflow.draft_segments),
            "current_segment_index": self.workflow.current_segment_index,
            "questions": [question.question_id for question in self.workflow.questions],
            "available_action_names": available_action_names,
            "action_history": list(self.workflow.action_history),
            "last_tool_result": dict(self.workflow.last_tool_result),
            "artifacts": list(self.workflow.artifacts),
            "memory_atoms_count": len(self.task_state.memory_atoms),
            "compressed_memory_count": len(self.task_state.compressed_memory),
        }
        return Observation(payload=payload, source=self.scenario_type)

    def _has_domain_state(self) -> bool:
        return bool(
            self.task_state.source_analyses
            or self.task_state.source_documents
            or self.task_state.source_chunks
            or self.task_state.characters
            or self.task_state.plot_threads
            or self.task_state.memory_atoms
        )

    def _require_blueprint(self) -> ChapterBlueprint:
        if self.workflow.proposed_blueprint is None:
            raise RuntimeError("blueprint is not available")
        return self.workflow.proposed_blueprint

    def _require_evidence_pack(self) -> EvidencePack:
        if self.workflow.evidence_pack is None:
            raise RuntimeError("evidence pack is not available")
        return self.workflow.evidence_pack

    def _require_draft(self) -> DraftCandidate:
        if self.workflow.draft is None:
            raise RuntimeError("draft is not available")
        return self.workflow.draft

    def _should_write_segments(self) -> bool:
        blueprint = self.workflow.proposed_blueprint
        return bool(
            blueprint
            and blueprint.segments
            and (blueprint.target_total_chars >= 12000 or self.request.target_word_count >= 12000)
        )


class NarrativeAuthorLedPolicy:
    """Deterministic ReAct policy that respects author confirmation gates."""

    _priority = (
        "ask_author",
        "scan_workspace",
        "load_state_snapshot",
        "load_analysis",
        "analyze_source",
        "propose_blueprint",
        "wait_for_confirmation",
        "confirm_blueprint",
        "retrieve_context",
        "build_working_context",
        "generate_segment",
        "merge_draft_segments",
        "generate_draft",
        "evaluate_draft",
        "compress_new_draft",
        "commit_state",
        "save_artifacts",
        "rollback",
        "stop",
    )

    def select_action(self, state: AgentState, actions: Sequence[Action]) -> Decision:
        by_name = {action.name: action for action in actions}
        for name in self._priority:
            if name in by_name:
                return Decision(action=by_name[name], rationale=f"author-led narrative workflow phase selects {name}")
        return Decision(action=Action("stop", kind="control"), rationale="no narrative action available")


def narrative_goal_from_request(request: AuthorRequest) -> Goal:
    criteria = ["produce narrative draft", "update canonical state if valid"]
    if request.target_word_count:
        criteria.append(f"target at least {request.target_word_count} characters when requested")
    criteria.extend(request.constraints)
    return Goal(request.request, tuple(criteria), metadata={"story_id": request.story_id, "task_id": request.task_id})


def _reward_from_reports(reports: Sequence[EvaluationReport], committed: bool) -> Reward:
    if not reports:
        return Reward(-1.0, {"no_reports": 1.0})
    score = sum(report.overall_score for report in reports) / len(reports)
    return Reward(
        value=score if committed else score - (0.2 if any(report.blocks_commit for report in reports) else 0.0),
        dimensions={
            "committed": 1.0 if committed else 0.0,
            "average_report_score": score,
            "blocking_reports": float(sum(1 for report in reports if report.blocks_commit)),
        },
    )


def _average_report_score(reports: Sequence[EvaluationReport]) -> float:
    if not reports:
        return 0.0
    return sum(report.overall_score for report in reports) / len(reports)


def _blueprint_confirmation_message(blueprint: ChapterBlueprint) -> str:
    required = "；".join(blueprint.required_beats) or "未拆出必写节拍"
    forbidden = "；".join(blueprint.forbidden_beats) or "无"
    segment_lines = []
    for index, segment in enumerate(blueprint.segments, start=1):
        target = f"，约 {segment.target_chars} 字" if segment.target_chars else ""
        segment_lines.append(f"{index}. {segment.goal}{target}")
    segments = "\n".join(segment_lines) if segment_lines else "未拆分段落"
    return (
        "已生成章节蓝图，需要作者确认后再写入执行。\n"
        f"章节目标：{blueprint.chapter_goal}\n"
        f"总字数目标：{blueprint.target_total_chars or '未指定'}\n"
        f"必写节拍：{required}\n"
        f"禁止发展：{forbidden}\n"
        f"节奏目标：{blueprint.pacing_target}\n"
        f"段落规划：\n{segments}\n"
        "确认后请以 confirm_plan=True 重新运行。"
    )


def _final_message(committed: bool, reports: Sequence[EvaluationReport]) -> str:
    if committed:
        return "草稿已通过核心校验并提交为 canonical state，本轮记忆已压缩。"
    issues = [issue.summary for report in reports for issue in report.issues if issue.severity == "blocker"]
    return "草稿未提交，需要修复：" + "；".join(issues)
