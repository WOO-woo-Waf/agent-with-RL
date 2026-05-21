"""OOAD implementation of the narrative-writing Agent scenario."""

from agent_rl.narrative_writing.agent import NarrativeWritingAgent
from agent_rl.narrative_writing.factory import build_narrative_writing_agent
from agent_rl.narrative_writing.ingestion import (
    build_author_request_from_files,
    load_reference_directory,
    load_reference_file,
    read_text_file,
)
from agent_rl.narrative_writing.persistence import FileNarrativeAnalysisRepository
from agent_rl.narrative_writing.persistence import FileNarrativeStateRepository
from agent_rl.narrative_writing.prompting import PromptComposer, PromptRegistry, compose_system_prompt
from agent_rl.narrative_writing.policies import (
    LLMDeepNarrativeAnalysisPolicy,
    LLMNarrativeExtractorPolicy,
    LLMNarrativeWriterPolicy,
    RuleBasedSourceAnalysisPolicy,
)
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
from agent_rl.narrative_writing.react import NarrativeAuthorLedPolicy, NarrativeReActEnvironment, NarrativeWorkflowState
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter
from agent_rl.narrative_writing.tools import (
    LoadAnalysisTool,
    NarrativeToolResult,
    SaveNarrativeArtifactsTool,
    ScanWorkspaceTool,
    build_state_from_analysis,
)

__all__ = [
    "AuthorQuestion",
    "AuthorRequest",
    "ChatModelClient",
    "FileNarrativeAnalysisRepository",
    "FileNarrativeStateRepository",
    "JsonBlobParser",
    "LLMDeepNarrativeAnalysisPolicy",
    "LLMNarrativeExtractorPolicy",
    "LLMNarrativeWriterPolicy",
    "NarrativeRunResult",
    "NarrativeScenarioAdapter",
    "NarrativeAuthorLedPolicy",
    "NarrativeReActEnvironment",
    "NarrativeToolResult",
    "NarrativeWritingAgent",
    "NarrativeWorkflowState",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleConfig",
    "PromptComposer",
    "PromptRegistry",
    "ReferenceMaterial",
    "RuleBasedSourceAnalysisPolicy",
    "LoadAnalysisTool",
    "SaveNarrativeArtifactsTool",
    "ScanWorkspaceTool",
    "build_author_request_from_files",
    "build_narrative_writing_agent",
    "build_state_from_analysis",
    "compose_system_prompt",
    "has_llm_configuration",
    "load_reference_directory",
    "load_reference_file",
    "read_text_file",
]
