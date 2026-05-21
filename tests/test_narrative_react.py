from agent_rl.core import AgentRuntime, Goal
from agent_rl.core.memory import InMemoryStore
from agent_rl.narrative_writing import (
    AuthorRequest,
    NarrativeAuthorLedPolicy,
    NarrativeReActEnvironment,
    NarrativeWritingAgent,
    ReferenceMaterial,
)


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
