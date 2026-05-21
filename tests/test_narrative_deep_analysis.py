import json

from agent_rl.narrative_writing.persistence import FileNarrativeAnalysisRepository
from agent_rl.narrative_writing.policies import LLMDeepNarrativeAnalysisPolicy
from agent_rl.narrative_writing.requests import ReferenceMaterial


class FakeChatClient:
    def __init__(self) -> None:
        self.purposes = []

    def complete(self, messages, *, purpose: str, json_mode: bool = True) -> str:
        self.purposes.append(purpose)
        assert json_mode is True
        if purpose == "novel_chunk_analysis":
            return json.dumps(
                {
                    "summary": "Lin receives a sealed letter and chooses not to reveal it.",
                    "characters": [{"name": "Lin", "goal": "protect the letter"}],
                    "events": [{"summary": "Lin hides the letter", "participants": ["Lin"]}],
                    "world_facts": ["letters can carry political risk"],
                    "plot_threads": ["sealed letter mystery"],
                    "open_questions": ["who sent the letter"],
                    "style": {"pov": "close third", "dialogue_style": "restrained"},
                    "evidence": {
                        "source_quotes": ["Lin received the sealed letter."],
                        "style_snippets": ["rain pressed the warehouse quiet"],
                        "retrieval_keywords": ["letter", "warehouse"],
                    },
                    "state_completeness": {"confidence": 0.8},
                }
            )
        if purpose == "novel_chapter_analysis":
            return json.dumps(
                {
                    "chapter_index": 1,
                    "chapter_summary": "The sealed letter becomes the chapter hook.",
                    "chapter_synopsis": "Lin hides a risky sealed letter.",
                    "chapter_events": ["Lin hides the letter"],
                    "characters_involved": ["Lin"],
                    "plot_progress": ["sealed letter mystery opens"],
                    "world_rules_confirmed": ["letters can carry political risk"],
                    "open_questions": ["who sent the letter"],
                    "retrieval_keywords": ["letter"],
                    "state_completeness": {"confidence": 0.85},
                }
            )
        if purpose == "novel_global_analysis":
            return json.dumps(
                {
                    "story_id": "story-deep",
                    "title": "Deep Story",
                    "story_synopsis": "A sealed-letter mystery begins.",
                    "character_cards": [
                        {
                            "character_id": "char-lin",
                            "name": "Lin",
                            "current_goals": ["protect the letter"],
                            "knowledge_boundary": ["knows the letter exists"],
                        }
                    ],
                    "plot_threads": [
                        {
                            "thread_id": "plot-letter",
                            "name": "sealed letter mystery",
                            "stage": "opened",
                            "open_questions": ["who sent the letter"],
                        }
                    ],
                    "world_rules": [{"rule_id": "rule-letter", "rule_text": "Letters can carry political risk."}],
                    "style_bible": {"narrative_pov": "close third", "dialogue_ratio": 0.2},
                    "continuation_constraints": ["do not reveal the sender yet"],
                    "state_completeness": {"overall_score": 0.8},
                }
            )
        raise AssertionError(f"unexpected purpose {purpose}")


def test_llm_deep_analysis_builds_three_level_assets_and_persists(tmp_path) -> None:
    repository = FileNarrativeAnalysisRepository(tmp_path)
    policy = LLMDeepNarrativeAnalysisPolicy(
        FakeChatClient(),
        repository=repository,
        max_chunk_chars=1000,
    )

    analysis = policy.analyze(
        (
            ReferenceMaterial(
                title="Reference",
                text="Chapter 1\nLin received the sealed letter.\nRain pressed the warehouse quiet.",
            ),
        ),
        task_id="task-deep",
        story_id="story-deep",
        goal="continue the next chapter",
        writing_direction="keep the letter mystery moving",
    )

    assert analysis.chunk_analyses
    assert analysis.chapter_analyses
    assert analysis.global_analysis is not None
    assert analysis.characters[0].name == "Lin"
    assert analysis.plot_threads[0].name == "sealed letter mystery"
    assert analysis.coverage["llm_global_analysis_count"] == 1.0

    story_dir = tmp_path / "story-deep" / "task-deep"
    assert (story_dir / "manifest.json").exists()
    assert (story_dir / "chunk_analysis.jsonl").exists()
    loaded_global = repository.load_global_analysis(story_id="story-deep", task_id="task-deep")
    assert loaded_global is not None
    assert loaded_global.story_synopsis == "A sealed-letter mystery begins."
