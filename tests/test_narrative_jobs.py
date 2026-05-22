from agent_rl.narrative_writing import AuthorRequest, NarrativeWritingSession, ReferenceMaterial
from agent_rl.narrative_writing.jobs import FileNarrativeJobRepository, NarrativeJob, NarrativeJobRunner
from agent_rl.narrative_writing.persistence import (
    FileNarrativeConversationRepository,
    FileNarrativeStateRepository,
    SQLiteNarrativeMemoryRepository,
)
from agent_rl.narrative_writing.run_graph import NarrativeTaskNode, ParallelToolExecutor


def test_narrative_job_runner_resumes_and_confirms_blueprint(tmp_path) -> None:
    state_repository = FileNarrativeStateRepository(tmp_path / "state")
    conversation_repository = FileNarrativeConversationRepository(tmp_path / "conversation")
    job_repository = FileNarrativeJobRepository(tmp_path / "jobs")
    memory_repository = SQLiteNarrativeMemoryRepository(tmp_path / "memory.sqlite3")
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            session_id="job-session",
            story_id="job-story",
            task_id="job-task",
            references=(ReferenceMaterial(title="ref", text="rain, letter, old warehouse"),),
            writing_direction="continue the warehouse clue",
            confirm_plan=False,
        ),
        state_repository=state_repository,
        conversation_repository=conversation_repository,
        memory_repository=memory_repository,
    )
    paused = session.run_until_pause()
    assert paused.requires_confirmation is True

    job_repository.enqueue(
        NarrativeJob(
            job_id="job-confirm",
            job_type="confirm_blueprint",
            session_id="job-session",
            story_id="job-story",
        )
    )
    finished = NarrativeJobRunner(
        job_repository=job_repository,
        state_repository=state_repository,
        conversation_repository=conversation_repository,
        memory_repository=memory_repository,
    ).run_next()

    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.result_summary["outcome"] == "committed"


def test_narrative_job_runner_can_propose_blueprint_and_compress_memory(tmp_path) -> None:
    state_repository = FileNarrativeStateRepository(tmp_path / "state")
    conversation_repository = FileNarrativeConversationRepository(tmp_path / "conversation")
    job_repository = FileNarrativeJobRepository(tmp_path / "jobs")
    memory_repository = SQLiteNarrativeMemoryRepository(tmp_path / "memory.sqlite3")
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            session_id="job-session-2",
            story_id="job-story-2",
            task_id="job-task-2",
            references=(ReferenceMaterial(title="ref", text="rain, letter, old warehouse"),),
            writing_direction="continue the warehouse clue",
            confirm_plan=False,
        ),
        state_repository=state_repository,
        conversation_repository=conversation_repository,
        memory_repository=memory_repository,
    )
    session.save()

    job_repository.enqueue(
        NarrativeJob(
            job_id="job-blueprint",
            job_type="blueprint_proposal",
            session_id="job-session-2",
            story_id="job-story-2",
        )
    )
    runner = NarrativeJobRunner(
        job_repository=job_repository,
        state_repository=state_repository,
        conversation_repository=conversation_repository,
        memory_repository=memory_repository,
    )
    blueprint_job = runner.run_next()

    assert blueprint_job is not None
    assert blueprint_job.status == "succeeded"
    assert blueprint_job.result_summary["requires_confirmation"] is True

    resumed = NarrativeWritingSession.resume(
        "job-session-2",
        story_id="job-story-2",
        state_repository=state_repository,
        conversation_repository=conversation_repository,
        memory_repository=memory_repository,
    )
    resumed.apply_author_input(confirm_plan=True)
    resumed.run_until_pause()
    assert memory_repository.load_memory_atoms("job-story-2")

    job_repository.enqueue(
        NarrativeJob(
            job_id="job-memory-compress",
            job_type="memory_compression",
            session_id="job-session-2",
            story_id="job-story-2",
            payload={"decay_amount": 0.1},
        )
    )
    memory_job = runner.run_next()

    assert memory_job is not None
    assert memory_job.status == "succeeded"
    assert memory_job.result_summary["outcome"] == "committed"


def test_parallel_tool_executor_collects_candidate_results() -> None:
    nodes = [
        NarrativeTaskNode(node_id="b", task_type="branch", payload={"value": 2}),
        NarrativeTaskNode(node_id="a", task_type="branch", payload={"value": 1}),
    ]

    results = ParallelToolExecutor(max_workers=2).run(nodes, lambda node: {"value": node.payload["value"] * 2})

    assert [result.node_id for result in results] == ["a", "b"]
    assert [result.payload["value"] for result in results] == [2, 4]
