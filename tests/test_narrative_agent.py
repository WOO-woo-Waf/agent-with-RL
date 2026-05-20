from agent_rl.narrative_writing import AuthorRequest, NarrativeWritingAgent, ReferenceMaterial


def _reference() -> ReferenceMaterial:
    return ReferenceMaterial(
        title="参考小说",
        text="林舟站在旧仓库门口。雨声很低，他握着密信，没有立刻说出真相。",
    )


def test_narrative_agent_asks_for_missing_author_context() -> None:
    result = NarrativeWritingAgent().run(AuthorRequest(request="帮我续写"))

    assert result.trajectory.outcome == "needs_author_input"
    assert {question.question_id for question in result.questions} == {
        "reference_material",
        "writing_direction",
    }


def test_narrative_agent_requires_plan_confirmation_before_generation() -> None:
    result = NarrativeWritingAgent().run(
        AuthorRequest(
            request="帮我续写",
            references=(_reference(),),
            writing_direction="下一章找到密信，但不要立刻和解",
            confirm_plan=False,
        )
    )

    assert result.trajectory.outcome == "needs_confirmation"
    assert result.requires_confirmation is True
    assert result.proposed_blueprint is not None


def test_narrative_agent_runs_and_commits_after_confirmation() -> None:
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
    assert result.committed is True
    assert result.draft is not None
    assert result.state.state_version_no == 1
    assert result.state.memory_atoms
    assert result.state.compressed_memory
