from pathlib import Path

from agent_rl.core import AgentRuntime, Goal
from agent_rl.core.memory import InMemoryStore
from agent_rl.narrative_writing import (
    AuthorRequest,
    FileNarrativeAnalysisRepository,
    FileNarrativeStateRepository,
    LoadAnalysisTool,
    NarrativeAuthorLedPolicy,
    NarrativeReActEnvironment,
    NarrativeWritingAgent,
    ReferenceMaterial,
)
from agent_rl.narrative_writing.policies import RuleBasedSourceAnalysisPolicy


def _reference() -> ReferenceMaterial:
    return ReferenceMaterial(
        title="参考小说",
        text=(
            "林舟站在旧仓库门口。雨声很低，他握着密信，没有立刻说出真相。\n\n"
            "对方沉默地站在灯下，旧账和新的线索同时压了下来。"
        ),
    )


def test_file_state_repository_round_trips_narrative_state(tmp_path: Path) -> None:
    result = NarrativeWritingAgent().run(
        AuthorRequest(
            request="帮我续写",
            references=(_reference(),),
            writing_direction="下一章找到密信，保持关系紧张",
            confirm_plan=True,
        )
    )
    repository = FileNarrativeStateRepository(tmp_path)

    snapshot_path = repository.save_state_snapshot(result.state, run_id="unit")
    loaded = repository.load_state_snapshot(result.state.story_id, path=snapshot_path)

    assert loaded.story_id == result.state.story_id
    assert loaded.state_version_no == result.state.state_version_no
    assert len(loaded.memory_atoms) == len(result.state.memory_atoms)
    assert loaded.compressed_memory


def test_load_analysis_tool_builds_state_from_source_analysis(tmp_path: Path) -> None:
    request = AuthorRequest(
        request="规划下一章",
        references=(_reference(),),
        writing_direction="下一章继续推进密信线索",
    )
    analysis = RuleBasedSourceAnalysisPolicy().analyze(
        request.references,
        task_id=request.task_id,
        story_id=request.story_id,
        goal=request.request,
        writing_direction=request.writing_direction,
    )
    analysis_repo = FileNarrativeAnalysisRepository(tmp_path / "analysis")
    analysis_repo.save_source_analysis(analysis)
    source_analysis_path = tmp_path / "analysis" / request.story_id / request.task_id / "source_analysis.json"

    state, tool_result = LoadAnalysisTool(FileNarrativeStateRepository(tmp_path / "state")).invoke(
        source_analysis_path=source_analysis_path,
        request=request,
    )

    assert tool_result.success is True
    assert state.source_analyses
    assert state.source_chunks
    assert state.plot_threads


def test_longform_layers_are_attached_to_working_context() -> None:
    result = NarrativeWritingAgent().run(
        AuthorRequest(
            request="帮我续写",
            references=(_reference(),),
            writing_direction="下一章找到密信，保持关系紧张",
            confirm_plan=True,
        )
    )

    assert result.state.working_context is not None
    source_types = {section.source_type for section in result.state.working_context.sections}
    assert {"longform_near", "longform_mid", "longform_global"} & source_types
    assert result.state.working_context.metadata["longform_layers"]["near"] > 0


def test_react_environment_can_persist_artifacts(tmp_path: Path) -> None:
    request = AuthorRequest(
        request="帮我续写",
        references=(_reference(),),
        writing_direction="下一章找到密信，保持关系紧张",
        confirm_plan=True,
        persist_artifacts=True,
        artifact_root=str(tmp_path),
    )
    env = NarrativeReActEnvironment(request)
    trajectory = AgentRuntime(NarrativeAuthorLedPolicy(), memory=InMemoryStore()).run(
        Goal("续写并保存"),
        env,
        max_steps=12,
    )

    assert trajectory.outcome == "committed"
    assert env.workflow.artifacts
    assert any("state_snapshots" in item for item in env.workflow.artifacts)
    assert any(Path(item).exists() for item in env.workflow.artifacts)


def test_longform_target_uses_segment_generation() -> None:
    result = NarrativeWritingAgent().run(
        AuthorRequest(
            request="帮我续写长章节",
            references=(_reference(),),
            writing_direction="下一章找到密信，保持关系紧张",
            target_word_count=12000,
            confirm_plan=True,
        )
    )

    action_names = [step.action.name for step in result.trajectory.steps]
    assert "generate_segment" in action_names
    assert "merge_draft_segments" in action_names
    assert result.draft is not None
    assert result.draft.metadata["writer_policy"] == "SegmentedNarrativeWriter"
