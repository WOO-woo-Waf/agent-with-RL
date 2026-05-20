"""Package-wide LLM calling, JSON parsing, and audit infrastructure."""

from agent_rl.llm.audit import (
    JsonlAuditWriter,
    LLMInteractionAuditRecord,
    LLMTokenUsageAuditRecord,
    build_interaction_record,
    build_usage_record,
    extract_prompt_metadata,
    new_interaction_id,
    record_interaction,
    record_token_usage,
    summarize_messages,
)
from agent_rl.llm.clients import (
    EndpointPool,
    LLMEndpoint,
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
    has_llm_configuration,
)
from agent_rl.llm.contracts import ChatMessage, ChatModelClient
from agent_rl.llm.json import JsonBlobParser, JsonParseResult, extract_json_text

__all__ = [
    "ChatMessage",
    "ChatModelClient",
    "EndpointPool",
    "JsonBlobParser",
    "JsonParseResult",
    "JsonlAuditWriter",
    "LLMEndpoint",
    "LLMInteractionAuditRecord",
    "LLMTokenUsageAuditRecord",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleConfig",
    "build_interaction_record",
    "build_usage_record",
    "extract_json_text",
    "extract_prompt_metadata",
    "has_llm_configuration",
    "new_interaction_id",
    "record_interaction",
    "record_token_usage",
    "summarize_messages",
]
