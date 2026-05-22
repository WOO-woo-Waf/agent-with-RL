from agent_rl.domains.narrative import CompressedMemoryBlock, MemoryAtom, NarrativeQuery, NarrativeTaskState
from agent_rl.narrative_writing import AuthorRequest, NarrativeScenarioAdapter, NarrativeWritingSession, ReferenceMaterial
from agent_rl.narrative_writing.persistence import SQLiteNarrativeMemoryRepository
from agent_rl.narrative_writing.policies import SQLiteFTSNarrativeRetrievalPolicy
from agent_rl.narrative_writing.utils import new_id


def test_sqlite_memory_repository_indexes_searches_and_invalidates(tmp_path) -> None:
    repository = SQLiteNarrativeMemoryRepository(tmp_path / "memory.sqlite3")
    state = NarrativeTaskState(
        task_id="memory-task",
        story_id="memory-story",
        goal="continue",
        memory_atoms=[
            MemoryAtom(
                memory_id="memory-1",
                memory_type="plot_progress",
                text="The sealed warehouse letter points to the river gate.",
                canonical=True,
                importance=0.9,
                freshness=1.0,
            )
        ],
        compressed_memory=[
            CompressedMemoryBlock(
                block_id="compressed-1",
                block_type="chapter_delta",
                scope="state_version:1",
                summary="Warehouse clue moved toward the river gate.",
                key_points=["sealed letter", "river gate"],
            )
        ],
    )

    repository.upsert_state_memory(state)

    matches = repository.search("memory-story", "river gate", limit=4)
    assert [item.source for item in matches] == ["sqlite_memory", "sqlite_memory"]
    assert repository.load_memory_atoms("memory-story")[0].memory_id == "memory-1"
    assert repository.load_compressed_memory("memory-story")[0].block_id == "compressed-1"

    assert repository.invalidate_memory_atoms("memory-story", ["memory-1"], reason="author changed clue") == 1
    assert repository.load_memory_atoms("memory-story") == []
    deprecated = repository.load_memory_atoms("memory-story", include_deprecated=True)
    assert deprecated[0].status == "deprecated"
    assert deprecated[0].invalidation_reason == "author changed clue"


def test_session_commit_upserts_memory_repository(tmp_path) -> None:
    repository = SQLiteNarrativeMemoryRepository(tmp_path / "memory.sqlite3")
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            session_id="memory-session",
            story_id="memory-story",
            task_id="memory-task",
            references=(ReferenceMaterial(title="ref", text="Rain, warehouse, sealed letter, river gate."),),
            writing_direction="continue toward the river gate",
            confirm_plan=True,
        ),
        memory_repository=repository,
    )

    result = session.run_until_pause()

    assert result.committed is True
    assert repository.load_memory_atoms("memory-story")


def test_sqlite_retrieval_policy_adds_persisted_memory_evidence(tmp_path) -> None:
    repository = SQLiteNarrativeMemoryRepository(tmp_path / "memory.sqlite3")
    state = NarrativeTaskState(
        task_id="retrieval-task",
        story_id="retrieval-story",
        goal="continue",
        memory_atoms=[
            MemoryAtom(
                memory_id="memory-1",
                memory_type="plot_progress",
                text="The heroine hides the black key under the bridge.",
                canonical=True,
                importance=0.8,
                freshness=1.0,
            )
        ],
    )
    repository.upsert_state_memory(state)

    pack = SQLiteFTSNarrativeRetrievalPolicy(repository).retrieve(
        state,
        NarrativeQuery(
            query_id=new_id("query"),
            query_text="black key bridge",
            query_type="chapter_continuation",
        ),
    )

    assert any(item.source == "sqlite_memory" and "black key" in item.text for item in pack.plot_evidence)
    assert pack.retrieval_trace[-1]["persisted_memory_count"] >= 1


def test_scenario_with_memory_repository_uses_sqlite_retrieval_policy(tmp_path) -> None:
    repository = SQLiteNarrativeMemoryRepository(tmp_path / "memory.sqlite3")
    scenario = NarrativeScenarioAdapter(memory_repository=repository)

    assert isinstance(scenario.retrieval_policy, SQLiteFTSNarrativeRetrievalPolicy)


def test_session_can_invalidate_memory_in_state_and_repository(tmp_path) -> None:
    repository = SQLiteNarrativeMemoryRepository(tmp_path / "memory.sqlite3")
    session = NarrativeWritingSession(
        AuthorRequest(
            request="continue",
            session_id="invalidate-session",
            story_id="invalidate-story",
            task_id="invalidate-task",
            references=(ReferenceMaterial(title="ref", text="The warehouse clue was false."),),
            writing_direction="continue",
            confirm_plan=True,
        ),
        memory_repository=repository,
    )
    session.run_until_pause()
    atom = repository.load_memory_atoms("invalidate-story")[0]

    invalidated = session.invalidate_memory(memory_ids=(atom.memory_id,), reason="author rejected this clue")

    assert invalidated == [atom.memory_id]
    active_ids = {item.memory_id for item in repository.load_memory_atoms("invalidate-story")}
    deprecated = {item.memory_id: item for item in repository.load_memory_atoms("invalidate-story", include_deprecated=True)}
    assert atom.memory_id not in active_ids
    assert deprecated[atom.memory_id].status == "deprecated"
