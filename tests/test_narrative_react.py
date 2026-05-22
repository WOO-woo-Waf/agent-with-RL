from agent_rl.core import AgentRuntime, Goal
from agent_rl.core.memory import InMemoryStore
from agent_rl.domains.narrative import DraftCandidate, EvaluationIssue, EvaluationReport, NarrativeTaskState, StateChangeProposal
from agent_rl.narrative_writing import (
    AuthorRequest,
    NarrativeAuthorLedPolicy,
    NarrativeReActEnvironment,
    NarrativeWritingAgent,
    NarrativeWritingSession,
    ReferenceMaterial,
    NarrativeScenarioAdapter,
)
from agent_rl.narrative_writing.persistence import FileNarrativeConversationRepository, FileNarrativeStateRepository


def _reference() -> ReferenceMaterial:
    return ReferenceMaterial(
        title="参考小说",
        text="林舟站在旧仓库门口。雨声很低，他握着密信，没有立刻说出真相。",
    )


def test_narrative_react_environment_stops_for_blueprint_confirmation() -> None:
    request = AuthorRequest(
        request="帮我续写",
        references=(_reference(),),
        writing_direction="下一章找到密信，但不要立刻和解",
        confirm_plan=False,
    )
    env = NarrativeReActEnvironment(request)
    trajectory = AgentRuntime(NarrativeAuthorLedPolicy(), memory=InMemoryStore()).run(
        Goal("续写下一章"),
        env,
        max_steps=8,
    )

    assert trajectory.outcome == "needs_confirmation"
    assert [step.action.name for step in trajectory.steps] == [
        "scan_workspace",
        "analyze_source",
        "propose_blueprint",
        "wait_for_confirmation",
    ]
    assert env.workflow.proposed_blueprint is not None


def test_narrative_react_environment_commits_after_author_confirmation() -> None:
    result = NarrativeWritingAgent().run(
        AuthorRequest(
            request="帮我续写",
            references=(_reference(),),
            writing_direction="下一章找到密信，保持关系紧张",
            constraints=("不要让主角立刻原谅对方",),
            confirm_plan=True,
        )
    )

    assert result.trajectory.outcome == "committed"
    assert [step.action.name for step in result.trajectory.steps] == [
        "scan_workspace",
        "analyze_source",
        "propose_blueprint",
        "confirm_blueprint",
        "retrieve_context",
        "build_working_context",
        "generate_draft",
        "evaluate_draft",
        "compress_new_draft",
        "commit_state",
    ]
    assert result.committed is True


def test_chapter_blueprint_segments_receive_target_char_budget() -> None:
    result = NarrativeWritingAgent().run(
        AuthorRequest(
            request="帮我规划",
            references=(_reference(),),
            writing_direction="下一章找到密信，保持关系紧张",
            target_word_count=30000,
            confirm_plan=False,
        )
    )

    blueprint = result.proposed_blueprint
    assert blueprint is not None
    assert blueprint.target_total_chars == 30000
    assert len(blueprint.segments) >= 4
    assert sum(segment.target_chars for segment in blueprint.segments) == 30000


def test_narrative_session_can_pause_for_confirmation_and_resume() -> None:
    session = NarrativeWritingSession(
        AuthorRequest(
            request="甯垜缁啓",
            references=(_reference(),),
            writing_direction="涓嬩竴绔犳壘鍒板瘑淇★紝淇濇寔鍏崇郴绱у紶",
            confirm_plan=False,
        )
    )

    first = session.run_until_pause()
    assert first.requires_confirmation is True
    assert first.proposed_blueprint is not None

    session.apply_author_input(confirm_plan=True)
    second = session.run_until_pause()

    assert second.trajectory.outcome == "committed"
    assert second.committed is True
    action_names = [step.action.name for step in second.trajectory.steps]
    assert "wait_for_confirmation" in action_names
    assert "confirm_blueprint" in action_names


def test_narrative_session_can_persist_and_resume_after_confirmation_pause(tmp_path) -> None:
    repository = FileNarrativeStateRepository(tmp_path)
    conversation_repository = FileNarrativeConversationRepository(tmp_path / "conversation")
    session = NarrativeWritingSession(
        AuthorRequest(
            request="甯垜缁啓",
            session_id="session-unit",
            story_id="story-unit",
            task_id="task-unit",
            references=(_reference(),),
            writing_direction="涓嬩竴绔犳壘鍒板瘑淇★紝淇濇寔鍏崇郴绱у紶",
            confirm_plan=False,
        ),
        state_repository=repository,
        conversation_repository=conversation_repository,
    )

    first = session.run_until_pause()
    assert first.requires_confirmation is True
    saved_path = session.save()

    resumed = NarrativeWritingSession.resume(
        "session-unit",
        story_id="story-unit",
        state_repository=repository,
        conversation_repository=conversation_repository,
    )
    assert resumed.workflow_phase == "blueprint_proposed"
    assert resumed.result().requires_confirmation is True

    resumed.apply_author_input(confirm_plan=True)
    final = resumed.run_until_pause()

    assert saved_path.endswith("session-unit.json")
    assert final.trajectory.outcome == "committed"
    assert final.committed is True
    conversation = conversation_repository.load_conversation("session-unit", story_id="story-unit")
    assert conversation is not None
    assert conversation.decisions[-1].decision_type == "confirm"
    assert conversation.preference_profile is not None


def test_narrative_session_auto_checkpoints_and_restores_memory(tmp_path) -> None:
    repository = FileNarrativeStateRepository(tmp_path)
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            session_id="session-auto",
            story_id="story-auto",
            task_id="task-auto",
            references=(_reference(),),
            writing_direction="keep tension",
            confirm_plan=False,
        ),
        state_repository=repository,
    )

    session.run_until_pause()
    resumed = NarrativeWritingSession.resume("session-auto", story_id="story-auto", state_repository=repository)

    assert resumed.workflow_phase == "blueprint_proposed"
    assert len(resumed.memory.get("trajectory", [])) == len(resumed.trajectory.steps)
    assert resumed.result().requires_confirmation is True


def test_blueprint_revision_creates_new_confirmable_version() -> None:
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            references=(_reference(),),
            writing_direction="first plan",
            confirm_plan=False,
        )
    )

    first = session.run_until_pause()
    old_blueprint = first.proposed_blueprint
    assert old_blueprint is not None

    session.apply_author_input(writing_direction="revised plan", blueprint_feedback="make the conflict stronger")
    revised = session.run_until_pause()

    assert revised.requires_confirmation is True
    assert revised.proposed_blueprint is not None
    assert revised.proposed_blueprint.blueprint_id != old_blueprint.blueprint_id
    assert revised.proposed_blueprint.parent_blueprint_id == old_blueprint.blueprint_id
    assert revised.proposed_blueprint.revision_no == old_blueprint.revision_no + 1
    assert session.state.metadata["blueprint_revision_history"]


class _OneShotBlockingEvaluator:
    def evaluate(
        self,
        state: NarrativeTaskState,
        draft: DraftCandidate,
        changes: list[StateChangeProposal],
    ) -> list[EvaluationReport]:
        if draft.metadata.get("repaired"):
            return [EvaluationReport(report_id="report-pass", report_type="repair_gate", status="passed", overall_score=1.0)]
        return [
            EvaluationReport(
                report_id="report-block",
                report_type="repair_gate",
                status="failed",
                overall_score=0.0,
                issues=[
                    EvaluationIssue(
                        issue_id="issue-block",
                        issue_type="repairable",
                        severity="blocker",
                        summary="needs repair",
                        suggested_repair="add an explicit repair note",
                    )
                ],
            )
        ]


def test_repair_loop_fixes_blocked_draft_before_commit() -> None:
    scenario = NarrativeScenarioAdapter(evaluator_policy=_OneShotBlockingEvaluator())
    result = NarrativeWritingAgent(scenario=scenario).run(
        AuthorRequest(
            request="continue",
            references=(_reference(),),
            writing_direction="repairable direction",
            confirm_plan=True,
            max_repair_attempts=1,
        )
    )

    action_names = [step.action.name for step in result.trajectory.steps]
    assert "repair_draft" in action_names
    assert result.trajectory.outcome == "committed"
    assert result.committed is True


def test_branch_generation_pauses_for_author_selection_then_commits() -> None:
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            session_id="session-branch",
            story_id="story-branch",
            task_id="task-branch",
            references=(_reference(),),
            writing_direction="offer two paths",
            confirm_plan=True,
            branch_count=2,
        )
    )

    first = session.run_until_pause()
    assert first.trajectory.outcome == "needs_branch_selection"
    assert len(first.branches) == 2

    session.apply_author_input(selected_branch_id=first.branches[1].branch_id)
    final = session.run_until_pause()

    assert final.trajectory.outcome == "committed"
    assert final.committed is True
    assert final.draft is not None
    assert final.draft.draft_id == first.branches[1].draft.draft_id
