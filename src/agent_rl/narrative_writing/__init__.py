"""OOAD implementation of the narrative-writing Agent scenario."""

from agent_rl.narrative_writing.agent import NarrativeWritingAgent
from agent_rl.narrative_writing.agent import default_max_steps
from agent_rl.narrative_writing.factory import build_narrative_scenario, build_narrative_writing_agent
from agent_rl.narrative_writing.ingestion import (
    build_author_request_from_files,
    load_reference_directory,
    load_reference_file,
    read_text_file,
)
from agent_rl.narrative_writing.jobs import FileNarrativeJobRepository, NarrativeJob, NarrativeJobRunner
from agent_rl.narrative_writing.persistence import FileNarrativeAnalysisRepository
from agent_rl.narrative_writing.persistence import FileNarrativeConversationRepository
from agent_rl.narrative_writing.persistence import FileNarrativeEvaluationRepository
from agent_rl.narrative_writing.persistence import FileNarrativeStateRepository
from agent_rl.narrative_writing.persistence import SQLiteNarrativeMemoryRepository
from agent_rl.narrative_writing.longform_context import MemoryGovernancePolicy
from agent_rl.narrative_writing.prompting import PromptComposer, PromptRegistry, compose_system_prompt
from agent_rl.narrative_writing.policies import (
    LLMDeepNarrativeAnalysisPolicy,
    LLMNarrativeExtractorPolicy,
    LLMNarrativeWriterPolicy,
    RAGVectorNarrativeRetrievalPolicy,
    RuleBasedNarrativeRepairPolicy,
    RuleBasedSourceAnalysisPolicy,
    ScoreBasedBranchSelectionPolicy,
    SQLiteFTSNarrativeRetrievalPolicy,
)
from agent_rl.narrative_writing.rag_index import NarrativeRAGIndexingService, narrative_state_documents
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
from agent_rl.narrative_writing.run_graph import NarrativeRunGraph, NarrativeTaskNode, NarrativeTaskResult, ParallelToolExecutor
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter
from agent_rl.narrative_writing.session import NarrativeWritingSession
from agent_rl.narrative_writing.tools import (
    LoadAnalysisTool,
    NarrativeToolResult,
    SaveNarrativeArtifactsTool,
    ScanWorkspaceTool,
    build_state_from_analysis,
)
from agent_rl.narrative_writing.workbench import (
    FileNarrativeOperatorSessionRepository,
    NarrativeInteractiveWorkbench,
    NarrativeOperatorSessionState,
    NarrativeWorkbenchConfig,
    OperatorMessage,
    OperatorToolCall,
    WorkbenchDecision,
    WorkbenchOperatorPolicy,
)

__all__ = [
    "AuthorQuestion",
    "AuthorRequest",
    "ChatModelClient",
    "FileNarrativeAnalysisRepository",
    "FileNarrativeConversationRepository",
    "FileNarrativeEvaluationRepository",
    "FileNarrativeJobRepository",
    "FileNarrativeOperatorSessionRepository",
    "FileNarrativeStateRepository",
    "SQLiteNarrativeMemoryRepository",
    "JsonBlobParser",
    "LLMDeepNarrativeAnalysisPolicy",
    "LLMNarrativeExtractorPolicy",
    "LLMNarrativeWriterPolicy",
    "NarrativeRunResult",
    "NarrativeInteractiveWorkbench",
    "NarrativeOperatorSessionState",
    "NarrativeJob",
    "NarrativeJobRunner",
    "NarrativeRunGraph",
    "NarrativeScenarioAdapter",
    "NarrativeAuthorLedPolicy",
    "NarrativeReActEnvironment",
    "NarrativeToolResult",
    "NarrativeWritingAgent",
    "NarrativeWritingSession",
    "NarrativeWorkbenchConfig",
    "NarrativeWorkflowState",
    "NarrativeTaskNode",
    "NarrativeTaskResult",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleConfig",
    "OperatorMessage",
    "OperatorToolCall",
    "PromptComposer",
    "PromptRegistry",
    "ReferenceMaterial",
    "ParallelToolExecutor",
    "RAGVectorNarrativeRetrievalPolicy",
    "RuleBasedNarrativeRepairPolicy",
    "RuleBasedSourceAnalysisPolicy",
    "ScoreBasedBranchSelectionPolicy",
    "SQLiteFTSNarrativeRetrievalPolicy",
    "MemoryGovernancePolicy",
    "WorkbenchDecision",
    "WorkbenchOperatorPolicy",
    "LoadAnalysisTool",
    "SaveNarrativeArtifactsTool",
    "ScanWorkspaceTool",
    "build_author_request_from_files",
    "build_narrative_scenario",
    "build_narrative_writing_agent",
    "build_state_from_analysis",
    "compose_system_prompt",
    "default_max_steps",
    "has_llm_configuration",
    "load_reference_directory",
    "load_reference_file",
    "NarrativeRAGIndexingService",
    "narrative_state_documents",
    "read_text_file",
]
