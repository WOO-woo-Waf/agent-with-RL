from agent_rl.narrative_writing.prompting import PromptComposer, PromptRegistry, compose_system_prompt


def test_default_prompt_registry_loads_profile_and_task_prompt() -> None:
    registry = PromptRegistry()
    profile = registry.load_profile()
    binding = registry.get_binding("draft_generation", profile)
    template = registry.load_task_prompt(binding.task_prompt, expected_task="draft_generation")

    assert profile.id == "default"
    assert template.id == "draft_generation"
    assert template.content_hash


def test_prompt_composer_includes_metadata() -> None:
    prompt = compose_system_prompt(purpose="draft_generation")

    assert "Prompt Metadata" in prompt.system_content
    assert prompt.metadata["task_prompt_id"] == "draft_generation"
    assert prompt.metadata["global_prompt_id"] == "narrative_global_default"


def test_prompt_composer_can_use_explicit_registry() -> None:
    composer = PromptComposer(PromptRegistry())
    prompt = composer.compose_system_prompt(purpose="state_extraction")

    assert prompt.metadata["task_prompt_id"] == "state_extraction"
