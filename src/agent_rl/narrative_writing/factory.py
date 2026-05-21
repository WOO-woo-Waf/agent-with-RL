"""Factories for assembling narrative-writing Agents."""

from __future__ import annotations

from pathlib import Path

from agent_rl.config import load_project_env
from agent_rl.llm import OpenAICompatibleChatClient, OpenAICompatibleConfig, has_llm_configuration
from agent_rl.narrative_writing.agent import NarrativeWritingAgent
from agent_rl.narrative_writing.persistence import FileNarrativeAnalysisRepository
from agent_rl.narrative_writing.policies import (
    LLMDeepNarrativeAnalysisPolicy,
    LLMNarrativeExtractorPolicy,
    LLMNarrativeWriterPolicy,
)
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter


def build_narrative_writing_agent(
    *,
    use_llm: bool | None = None,
    use_llm_analysis: bool | None = None,
    env_path: str | Path | None = None,
    fallback_to_local: bool = True,
    persist_analysis: bool = True,
    analysis_repository_root: str | Path | None = None,
) -> NarrativeWritingAgent:
    """Build a narrative agent with optional LLM analysis/writer/extractor policies."""

    load_project_env(env_path, start=Path.cwd())
    config = OpenAICompatibleConfig.from_env()
    llm_configured = has_llm_configuration(config)
    enable_llm = llm_configured if use_llm is None else use_llm
    if enable_llm and not llm_configured:
        if not fallback_to_local:
            raise RuntimeError("LLM is requested but LLM_API_BASE, LLM_API_KEY, or LLM_MODEL is missing.")
        enable_llm = False
    if not enable_llm:
        return NarrativeWritingAgent()

    client = OpenAICompatibleChatClient(config)
    enable_llm_analysis = enable_llm if use_llm_analysis is None else use_llm_analysis
    repository = (
        FileNarrativeAnalysisRepository(analysis_repository_root or Path("artifacts") / "narrative")
        if persist_analysis and enable_llm_analysis
        else None
    )
    scenario = NarrativeScenarioAdapter(
        analysis_policy=LLMDeepNarrativeAnalysisPolicy(client, repository=repository) if enable_llm_analysis else None,
        writer_policy=LLMNarrativeWriterPolicy(client),
        extractor_policy=LLMNarrativeExtractorPolicy(client),
    )
    return NarrativeWritingAgent(scenario=scenario)
