from agent_rl.narrative_writing import AuthorRequest, ReferenceMaterial
from agent_rl.narrative_writing.bootstrap import build_initial_state
from agent_rl.narrative_writing.policies.analysis import RuleBasedSourceAnalysisPolicy
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter


def test_rule_based_source_analysis_creates_chunks_and_assets() -> None:
    reference = ReferenceMaterial(
        title="参考小说",
        text="第一章\n林舟握着密信。\n雨声压低了仓库里的呼吸。\n第二章\n沈姓角色没有立刻解释。",
    )

    analysis = RuleBasedSourceAnalysisPolicy(max_chunk_chars=18).analyze(
        (reference,),
        task_id="task-analysis",
        story_id="story-analysis",
        goal="续写下一章",
        writing_direction="继续围绕密信推进",
    )

    assert analysis.source_documents
    assert len(analysis.source_chunks) >= 2
    assert analysis.characters
    assert analysis.events
    assert analysis.style_profile is not None
    assert analysis.coverage["source_chunk_count"] == float(len(analysis.source_chunks))
    assert analysis.trace[0]["policy"] == "RuleBasedSourceAnalysisPolicy"


def test_bootstrap_stores_source_analysis_on_task_state() -> None:
    state = build_initial_state(
        AuthorRequest(
            request="续写下一章",
            references=(ReferenceMaterial(title="参考", text="林舟握着密信。\n沈姓角色没有立刻解释。"),),
            writing_direction="继续围绕密信推进",
        )
    )

    assert state.source_analyses
    assert state.source_chunks
    assert state.events
    assert state.metadata["source_analysis_coverage"]["event_count"] == float(len(state.events))


def test_scenario_uses_injected_analysis_policy() -> None:
    class EmptyAnalysisPolicy:
        def analyze(self, references, *, task_id, story_id, goal, writing_direction=""):
            from agent_rl.domains.narrative import NarrativeSourceAnalysis

            return NarrativeSourceAnalysis(
                analysis_id="analysis-empty",
                task_id=task_id,
                story_id=story_id,
                coverage={"reference_count": float(len(references))},
            )

    scenario = NarrativeScenarioAdapter(analysis_policy=EmptyAnalysisPolicy())
    state = scenario.build_initial_state(
        AuthorRequest(
            request="续写下一章",
            references=(ReferenceMaterial(title="参考", text="林舟握着密信。"),),
            writing_direction="继续围绕密信推进",
        )
    )

    assert state.source_analyses[0].analysis_id == "analysis-empty"
    assert state.source_chunks == []
