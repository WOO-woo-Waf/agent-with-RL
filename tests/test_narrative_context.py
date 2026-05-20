from agent_rl.narrative_writing import AuthorRequest, ReferenceMaterial
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter


def test_narrative_context_is_built_between_retrieval_and_writing() -> None:
    request = AuthorRequest(
        request="续写下一章",
        references=(ReferenceMaterial(title="参考", text="林舟握着密信，没有立刻解释。"),),
        writing_direction="下一章继续推进密信线索",
        constraints=("不要让主角立刻原谅对方",),
        confirm_plan=True,
    )
    scenario = NarrativeScenarioAdapter()
    state = scenario.build_initial_state(request)
    blueprint = scenario.propose_plan(state, request)
    query = scenario.build_query(state, request)
    evidence_pack = scenario.retrieve_context(state, query)
    plan = scenario.build_chapter_plan(state, blueprint, evidence_pack, request)

    context = scenario.build_working_context(state, plan, evidence_pack, request)

    assert context.sections
    assert context.render_for_model()
    assert context.estimated_tokens > 0
    assert state.working_context is context
    assert "retrieval_evaluation_report" in state.metadata


def test_agent_result_carries_context_metadata_in_draft() -> None:
    from agent_rl.narrative_writing import NarrativeWritingAgent

    result = NarrativeWritingAgent().run(
        AuthorRequest(
            request="续写下一章",
            references=(ReferenceMaterial(title="参考", text="林舟握着密信，没有立刻解释。"),),
            writing_direction="下一章继续推进密信线索",
            confirm_plan=True,
        )
    )

    assert result.draft is not None
    assert result.state.working_context is not None
    assert result.draft.metadata["working_context_id"] == result.state.working_context.context_id
