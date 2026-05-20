"""Application service that runs the narrative-writing Agent use case."""

from __future__ import annotations

from typing import Sequence

from agent_rl.core.concepts import Action, Goal, Observation, Reward, Trajectory, TrajectoryStep, utc_now
from agent_rl.domains.narrative import EvaluationReport, NarrativeTaskState
from agent_rl.narrative_writing.policies import BasicAuthorInteractionPolicy
from agent_rl.narrative_writing.ports import AuthorInteractionPolicy
from agent_rl.narrative_writing.requests import AuthorRequest, NarrativeRunResult
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter


class NarrativeWritingAgent:
    """Usable narrative-writing Agent with author interaction gates."""

    def __init__(
        self,
        scenario: NarrativeScenarioAdapter | None = None,
        interaction_policy: AuthorInteractionPolicy | None = None,
    ) -> None:
        self.scenario = scenario or NarrativeScenarioAdapter()
        self.interaction_policy = interaction_policy or BasicAuthorInteractionPolicy()

    def run(self, request: AuthorRequest, state: NarrativeTaskState | None = None) -> NarrativeRunResult:
        working_state = state or self.scenario.build_initial_state(request)
        goal = Goal(request.request, ("produce narrative draft", "update canonical state if valid"))
        trajectory = Trajectory(goal=goal, metadata={"scenario_type": self.scenario.scenario_type})

        questions = self.interaction_policy.missing_questions(request, state)
        if questions:
            trajectory.outcome = "needs_author_input"
            return NarrativeRunResult(
                state=working_state,
                trajectory=trajectory,
                questions=questions,
                assistant_message="我需要先补齐参考材料和写作方向，才能安全规划和续写。",
            )

        observation = self.scenario.build_observation(working_state)
        blueprint = self.scenario.propose_plan(working_state, request)
        _append_step(
            trajectory,
            0,
            observation,
            Action("propose_plan", payload={"blueprint_id": blueprint.blueprint_id}, kind="tool"),
            Reward(0.1, {"plan_created": 1.0}),
            self.scenario.build_observation(working_state),
            "turn author direction into chapter blueprint",
        )

        if not request.confirm_plan:
            trajectory.outcome = "needs_confirmation"
            return NarrativeRunResult(
                state=working_state,
                trajectory=trajectory,
                assistant_message=_blueprint_confirmation_message(blueprint),
                requires_confirmation=True,
                proposed_blueprint=blueprint,
            )

        query = self.scenario.build_query(working_state, request)
        pack = self.scenario.retrieve_context(working_state, query)
        _append_step(
            trajectory,
            1,
            observation,
            Action("retrieve_context", payload={"query_id": query.query_id}, kind="tool"),
            Reward(0.1, {"evidence_count": float(len(pack.all_evidence()))}),
            self.scenario.build_observation(working_state),
            "retrieve author, character, plot, world, style evidence",
        )

        plan = self.scenario.build_chapter_plan(working_state, blueprint, pack, request)
        working_context = self.scenario.build_working_context(working_state, plan, pack, request)
        _append_step(
            trajectory,
            2,
            observation,
            Action("build_working_context", payload={"context_id": working_context.context_id}, kind="tool"),
            Reward(0.1, {"context_sections": float(len(working_context.sections))}),
            self.scenario.build_observation(working_state),
            "assemble budgeted working context from plan and evidence",
        )

        draft = self.scenario.generate_draft(working_state, plan, pack, working_context)
        _append_step(
            trajectory,
            3,
            observation,
            Action("generate_draft", payload={"draft_id": draft.draft_id}, kind="tool"),
            Reward(0.2, {"draft_created": 1.0}),
            self.scenario.build_observation(working_state),
            "generate draft from chapter plan and evidence pack",
        )

        changes = self.scenario.extract_changes(working_state, draft)
        reports = self.scenario.evaluate(working_state, draft, changes)
        committed = self.scenario.commit_or_rollback(working_state, changes, reports)
        _append_step(
            trajectory,
            4,
            observation,
            Action("commit_or_rollback", payload={"committed": committed}, kind="tool"),
            _reward_from_reports(reports, committed),
            self.scenario.build_observation(working_state),
            "gate canonical state update through evaluators",
        )
        trajectory.outcome = "committed" if committed else "rolled_back"
        return NarrativeRunResult(
            state=working_state,
            trajectory=trajectory,
            assistant_message=_final_message(committed, reports),
            draft=draft,
            committed=committed,
        )


def _append_step(
    trajectory: Trajectory,
    index: int,
    observation: Observation,
    action: Action,
    reward: Reward,
    next_observation: Observation,
    rationale: str,
) -> None:
    trajectory.append(
        TrajectoryStep(
            index=index,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            rationale=rationale,
            ended_at=utc_now(),
        )
    )


def _reward_from_reports(reports: Sequence[EvaluationReport], committed: bool) -> Reward:
    if not reports:
        return Reward(-1.0, {"no_reports": 1.0})
    score = sum(report.overall_score for report in reports) / len(reports)
    return Reward(
        value=score if committed else score - 1.0,
        dimensions={
            "committed": 1.0 if committed else 0.0,
            "average_report_score": score,
            "blocking_reports": float(sum(1 for report in reports if report.blocks_commit)),
        },
    )


def _blueprint_confirmation_message(blueprint) -> str:
    required = "；".join(blueprint.required_beats) or "未拆出必写节拍"
    forbidden = "；".join(blueprint.forbidden_beats) or "无"
    return (
        "已生成章节蓝图，需要作者确认后再写入执行。\n"
        f"章节目标：{blueprint.chapter_goal}\n"
        f"必写节拍：{required}\n"
        f"禁止发展：{forbidden}\n"
        f"节奏目标：{blueprint.pacing_target}\n"
        "确认后请以 confirm_plan=True 重新运行。"
    )


def _final_message(committed: bool, reports: Sequence[EvaluationReport]) -> str:
    if committed:
        return "草稿已通过核心校验并提交为 canonical state，本轮记忆已压缩。"
    issues = [issue.summary for report in reports for issue in report.issues if issue.severity == "blocker"]
    return "草稿未提交，需要修复：" + "；".join(issues)
