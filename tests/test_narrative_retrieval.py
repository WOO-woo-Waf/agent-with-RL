from agent_rl.narrative_writing import AuthorRequest, ReferenceMaterial
from agent_rl.narrative_writing.policies.retrieval import CompositeNarrativeRetrievalPolicy, RetrievalQuota
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter


def test_composite_retrieval_selects_source_chunks_and_records_trace() -> None:
    request = AuthorRequest(
        request="续写下一章",
        references=(
            ReferenceMaterial(
                title="主线",
                text="林舟握着密信，没有立刻解释。\n仓库外的雨声压低了所有人的呼吸。",
                source_type="target_continuation",
            ),
            ReferenceMaterial(
                title="风格参考",
                text="街灯在潮湿的夜色里慢慢亮起。",
                source_type="reference_style",
            ),
        ),
        writing_direction="下一章继续推进密信线索",
        constraints=("不要让主角立刻原谅对方",),
    )
    scenario = NarrativeScenarioAdapter(
        retrieval_policy=CompositeNarrativeRetrievalPolicy(RetrievalQuota(source=4, plot=3, style=3))
    )
    state = scenario.build_initial_state(request)
    query = scenario.build_query(state, request)

    pack = scenario.retrieve_context(state, query)

    assert any(evidence.evidence_type == "source_chunk" for evidence in pack.plot_evidence)
    assert pack.author_plan_evidence
    assert pack.style_evidence
    assert pack.retrieval_trace[0]["policy"] == "CompositeNarrativeRetrievalPolicy"
    assert state.metadata["retrieval_evaluation_report"]["metrics"]["coverage"] > 0


def test_composite_retrieval_keeps_quota_limits() -> None:
    reference = ReferenceMaterial(
        title="参考",
        text="\n".join(f"林舟在第 {index} 段继续追查密信。" for index in range(12)),
    )
    request = AuthorRequest(
        request="续写下一章",
        references=(reference,),
        writing_direction="继续追查密信",
    )
    scenario = NarrativeScenarioAdapter(
        retrieval_policy=CompositeNarrativeRetrievalPolicy(RetrievalQuota(plot=1, source=2, style=1))
    )
    state = scenario.build_initial_state(request)
    query = scenario.build_query(state, request)

    pack = scenario.retrieve_context(state, query)

    assert len(pack.plot_evidence) <= 3
    assert len(pack.style_evidence) <= 1
