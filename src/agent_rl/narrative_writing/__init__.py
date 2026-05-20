"""OOAD implementation of the narrative-writing Agent scenario."""

from agent_rl.narrative_writing.agent import NarrativeWritingAgent
from agent_rl.narrative_writing.ingestion import (
    build_author_request_from_files,
    load_reference_directory,
    load_reference_file,
    read_text_file,
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
    "NarrativeRunResult",
    "NarrativeScenarioAdapter",
    "NarrativeWritingAgent",
    "ReferenceMaterial",
    "build_author_request_from_files",
    "load_reference_directory",
    "load_reference_file",
    "read_text_file",
]
