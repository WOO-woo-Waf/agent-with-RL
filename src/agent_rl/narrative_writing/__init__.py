"""OOAD implementation of the narrative-writing Agent scenario."""

from agent_rl.narrative_writing.agent import NarrativeWritingAgent
from agent_rl.narrative_writing.ingestion import (
    build_author_request_from_files,
    load_reference_directory,
    load_reference_file,
    read_text_file,
)
from agent_rl.narrative_writing.prompting import PromptComposer, PromptRegistry, compose_system_prompt
from agent_rl.narrative_writing.policies import LLMNarrativeExtractorPolicy, LLMNarrativeWriterPolicy, RuleBasedSourceAnalysisPolicy
from agent_rl.llm import ChatModelClient, JsonBlobParser
from agent_rl.llm import (
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
    has_llm_configuration,
)
from agent_rl.narrative_writing.requests import (
    AuthorQuestion,
    AuthorRequest,
    NarrativeRunResult,
    ReferenceMaterial,
)
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter

__all__ = [
    "AuthorQuestion",
    "AuthorRequest",
    "ChatModelClient",
    "JsonBlobParser",
    "LLMNarrativeExtractorPolicy",
    "LLMNarrativeWriterPolicy",
    "NarrativeRunResult",
    "NarrativeScenarioAdapter",
    "NarrativeWritingAgent",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleConfig",
    "PromptComposer",
    "PromptRegistry",
    "ReferenceMaterial",
    "RuleBasedSourceAnalysisPolicy",
    "build_author_request_from_files",
    "compose_system_prompt",
    "has_llm_configuration",
    "load_reference_directory",
    "load_reference_file",
    "read_text_file",
]
