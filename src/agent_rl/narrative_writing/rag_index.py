"""Narrative state to generic RAG index adapters."""

from __future__ import annotations

from agent_rl.domains.narrative import NarrativeTaskState
from agent_rl.rag import RAGDocument, RAGModelService


class NarrativeRAGIndexingService:
    """Indexes narrative state objects into a replaceable RAG service."""

    def __init__(self, rag_service: RAGModelService, *, collection_id: str = "narrative") -> None:
        self.rag_service = rag_service
        self.collection_id = collection_id

    def index_state(self, state: NarrativeTaskState, *, batch_size: int | None = None) -> int:
        documents = narrative_state_documents(state)
        return self.rag_service.index_documents(documents, collection_id=self.collection_id, batch_size=batch_size)


def narrative_state_documents(state: NarrativeTaskState) -> list[RAGDocument]:
    """Convert source, story state, and memory into RAG documents."""

    documents: list[RAGDocument] = []
    for chunk in state.source_chunks:
        documents.append(
            RAGDocument(
                document_id=chunk.chunk_id,
                story_id=state.story_id,
                evidence_type="source_chunk",
                source=f"source_chunks:{chunk.source_type}",
                text=chunk.text,
                chapter_index=chunk.chapter_index,
                metadata={"document_id": chunk.document_id, "chunk_index": chunk.chunk_index},
            )
        )
    for character in state.characters:
        documents.append(
            RAGDocument(
                document_id=character.character_id,
                story_id=state.story_id,
                evidence_type="character_profile",
                source="characters",
                text=f"{character.name}: traits={character.stable_traits}; goals={character.current_goals}; voice={character.voice_profile}; knowledge={character.knowledge_boundary}",
                related_entities=[character.character_id],
            )
        )
    for thread in state.plot_threads:
        documents.append(
            RAGDocument(
                document_id=thread.thread_id,
                story_id=state.story_id,
                evidence_type="plot_thread",
                source="plot_threads",
                text=f"{thread.name}: stage={thread.stage}; open={thread.open_questions}; next={thread.next_expected_beats}; blocked={thread.blocked_beats}",
                related_entities=list(thread.related_character_ids),
                related_plot_threads=[thread.thread_id],
            )
        )
    for event in state.events:
        documents.append(
            RAGDocument(
                document_id=event.event_id,
                story_id=state.story_id,
                evidence_type="event",
                source="events",
                text=event.summary,
                related_entities=list(event.participants),
                related_plot_threads=list(event.plot_thread_ids),
                chapter_index=event.chapter_index,
                metadata={"canonical": event.is_canonical},
            )
        )
    for rule in state.world_rules:
        documents.append(
            RAGDocument(
                document_id=rule.rule_id,
                story_id=state.story_id,
                evidence_type="world_rule",
                source="world_rules",
                text=rule.rule_text,
                related_entities=list(rule.applies_to),
                metadata={"stability": rule.stability, "rule_type": rule.rule_type},
            )
        )
    for snippet in state.style_snippets:
        documents.append(
            RAGDocument(
                document_id=snippet.snippet_id,
                story_id=state.story_id,
                evidence_type="style_snippet",
                source="style_snippets",
                text=snippet.text,
                chapter_index=snippet.chapter_index,
                metadata={"snippet_type": snippet.snippet_type, "style_tags": list(snippet.style_tags)},
            )
        )
    for atom in state.memory_atoms:
        if not atom.canonical or atom.status == "deprecated":
            continue
        documents.append(
            RAGDocument(
                document_id=atom.memory_id,
                story_id=state.story_id,
                evidence_type=atom.memory_type,
                source="memory_atoms",
                text=atom.text,
                related_entities=list(atom.related_entities),
                metadata={"importance": atom.importance, "freshness": atom.freshness, "state_version_no": atom.state_version_no},
            )
        )
    for block in state.compressed_memory:
        documents.append(
            RAGDocument(
                document_id=block.block_id,
                story_id=state.story_id,
                evidence_type="compressed_memory",
                source=f"compressed:{block.block_type}",
                text=block.summary + "\n" + "\n".join(block.key_points),
                metadata={"scope": block.scope, "preserved_ids": list(block.preserved_ids)},
            )
        )
    return [document for document in documents if document.text.strip()]
