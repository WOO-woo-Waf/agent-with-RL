from agent_rl.narrative_writing import build_narrative_writing_agent
from agent_rl.narrative_writing.policies import LLMNarrativeExtractorPolicy, LLMNarrativeWriterPolicy


def test_narrative_agent_factory_uses_local_agent_without_llm() -> None:
    agent = build_narrative_writing_agent(use_llm=False)

    assert not isinstance(agent.scenario.writer_policy, LLMNarrativeWriterPolicy)


def test_narrative_agent_factory_requires_llm_when_strict(tmp_path, monkeypatch) -> None:
    for key in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)

    try:
        build_narrative_writing_agent(
            use_llm=True,
            fallback_to_local=False,
            env_path=tmp_path / "missing.env",
        )
    except RuntimeError as exc:
        assert "LLM is requested" in str(exc)
    else:
        raise AssertionError("expected strict LLM configuration failure")


def test_narrative_agent_factory_wires_llm_policies(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")

    agent = build_narrative_writing_agent(use_llm=True)

    assert isinstance(agent.scenario.writer_policy, LLMNarrativeWriterPolicy)
    assert isinstance(agent.scenario.extractor_policy, LLMNarrativeExtractorPolicy)
