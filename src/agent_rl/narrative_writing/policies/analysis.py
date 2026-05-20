"""Local source-analysis policy for narrative reference materials."""

from __future__ import annotations

from typing import Sequence

from agent_rl.domains.narrative import (
    CharacterCard,
    MemoryAtom,
    NarrativeEvent,
    NarrativeSourceAnalysis,
    PlotThreadState,
    SourceChunk,
    SourceDocument,
    StyleProfile,
    StyleSnippet,
    WorldRule,
)
from agent_rl.narrative_writing.requests import ReferenceMaterial
from agent_rl.narrative_writing.utils import candidate_names, new_id, split_author_items, split_paragraphs


class RuleBasedSourceAnalysisPolicy:
    """Analyzes references without external services.

    This is the baseline implementation for tests and local learning. LLM,
    vector, or graph-backed analyzers can replace it through the analysis port.
    """

    def __init__(self, max_chunk_chars: int = 900) -> None:
        self.max_chunk_chars = max_chunk_chars

    def analyze(
        self,
        references: Sequence[ReferenceMaterial],
        *,
        task_id: str,
        story_id: str,
        goal: str,
        writing_direction: str = "",
    ) -> NarrativeSourceAnalysis:
        documents = _build_source_documents(references)
        chunks = [
            chunk
            for reference, document in zip(references, documents)
            for chunk in _build_source_chunks(reference, document, self.max_chunk_chars)
        ]
        analysis = NarrativeSourceAnalysis(
            analysis_id=new_id("analysis"),
            task_id=task_id,
            story_id=story_id,
            source_documents=documents,
            source_chunks=chunks,
            characters=_infer_characters(references),
            events=_infer_events(chunks),
            plot_threads=_infer_plot_threads(goal=goal, writing_direction=writing_direction),
            world_rules=_infer_world_rules(references),
            style_profile=_infer_style_profile(references),
            style_snippets=_infer_style_snippets(references),
            memory_atoms=_infer_reference_memory(references),
            coverage={
                "reference_count": float(len(references)),
                "source_document_count": float(len(documents)),
                "source_chunk_count": float(len(chunks)),
            },
        )
        analysis.coverage.update(
            {
                "character_count": float(len(analysis.characters)),
                "event_count": float(len(analysis.events)),
                "style_snippet_count": float(len(analysis.style_snippets)),
                "memory_atom_count": float(len(analysis.memory_atoms)),
            }
        )
        analysis.trace.append(
            {
                "policy": self.__class__.__name__,
                "max_chunk_chars": self.max_chunk_chars,
                "coverage": dict(analysis.coverage),
            }
        )
        return analysis


def _build_source_documents(references: Sequence[ReferenceMaterial]) -> list[SourceDocument]:
    return [
        SourceDocument(
            document_id=new_id("source"),
            title=reference.title,
            source_type=reference.source_type,
            author=reference.author,
            text_hash=str(abs(hash(reference.text))),
            metadata={"text_preview": reference.text[:240], "char_count": len(reference.text)},
        )
        for reference in references
    ]


def _build_source_chunks(reference: ReferenceMaterial, document: SourceDocument, max_chunk_chars: int) -> list[SourceChunk]:
    paragraphs = split_paragraphs(reference.text)
    if not paragraphs and reference.text.strip():
        paragraphs = [reference.text.strip()]
    chunks: list[SourceChunk] = []
    buffer: list[str] = []
    chapter_index = 1
    cursor = 0
    for paragraph in paragraphs:
        if _looks_like_chapter_heading(paragraph):
            chapter_index += 1 if chunks or buffer else 0
        candidate = "\n".join([*buffer, paragraph]) if buffer else paragraph
        if buffer and len(candidate) > max_chunk_chars:
            chunks.append(_chunk(document, reference.source_type, buffer, chapter_index, len(chunks), cursor))
            cursor += sum(len(item) for item in buffer)
            buffer = [paragraph]
        else:
            buffer.append(paragraph)
    if buffer:
        chunks.append(_chunk(document, reference.source_type, buffer, chapter_index, len(chunks), cursor))
    return chunks


def _chunk(
    document: SourceDocument,
    source_type: str,
    paragraphs: list[str],
    chapter_index: int,
    chunk_index: int,
    start_offset: int,
) -> SourceChunk:
    text = "\n".join(paragraphs)
    return SourceChunk(
        chunk_id=new_id("chunk"),
        document_id=document.document_id,
        source_type=source_type,
        text=text,
        chapter_index=chapter_index,
        chunk_index=chunk_index,
        start_offset=start_offset,
        end_offset=start_offset + len(text),
        metadata={"paragraph_count": len(paragraphs)},
    )


def _infer_characters(references: Sequence[ReferenceMaterial]) -> list[CharacterCard]:
    names: list[str] = []
    for reference in references:
        for token in candidate_names(reference.text):
            if token not in names:
                names.append(token)
    if not names:
        names = ["主角"]
    return [
        CharacterCard(
            character_id=new_id("char"),
            name=name,
            stable_traits=["目标明确", "行动受既有经历影响"],
            current_goals=["推进当前章节目标"],
            knowledge_boundary=["只知道参考材料中已揭示的信息"],
            voice_profile=["保持与参考文本一致的语气"],
            dialogue_do=["围绕当前冲突说话"],
            dialogue_do_not=["突然解释全部真相"],
            forbidden_actions=["无铺垫地改变核心立场"],
        )
        for name in names[:5]
    ]


def _infer_events(chunks: Sequence[SourceChunk]) -> list[NarrativeEvent]:
    events: list[NarrativeEvent] = []
    for chunk in chunks[:12]:
        summary = _first_sentence(chunk.text)[:120]
        if not summary:
            continue
        events.append(
            NarrativeEvent(
                event_id=new_id("event"),
                summary=summary,
                event_type="source_observed_event",
                chapter_index=chunk.chapter_index,
                participants=candidate_names(chunk.text),
                is_canonical=chunk.source_type == "target_continuation",
            )
        )
    return events


def _infer_plot_threads(*, goal: str, writing_direction: str) -> list[PlotThreadState]:
    target = writing_direction or goal
    return [
        PlotThreadState(
            thread_id=new_id("plot"),
            name="主线推进",
            thread_type="main",
            status="open",
            stage="continuation",
            stakes=target,
            open_questions=["当前线索会把角色带向哪里？"],
            next_expected_beats=split_author_items(target) or [target],
        )
    ]


def _infer_world_rules(references: Sequence[ReferenceMaterial]) -> list[WorldRule]:
    if not references:
        return []
    return [
        WorldRule(
            rule_id=new_id("rule"),
            rule_text="续写必须尊重参考材料中已经确认的人物关系、世界规则和事件因果。",
            stability="confirmed",
            forbidden_implications=["无证据推翻既有 canon"],
        )
    ]


def _infer_style_profile(references: Sequence[ReferenceMaterial]) -> StyleProfile:
    texts = [reference.text for reference in references if reference.text.strip()]
    avg_len = 0.0
    if texts:
        sentences = [part for text in texts for part in text.replace("！", "。").replace("？", "。").split("。") if part]
        avg_len = sum(len(sentence) for sentence in sentences) / max(1, len(sentences))
    return StyleProfile(
        profile_id=new_id("style"),
        narrative_pov="third_or_source_consistent",
        sentence_length_distribution={"avg": avg_len},
        dialogue_ratio=0.25,
        forbidden_patterns=["忽然无理由转折", "总结式说教"],
    )


def _infer_style_snippets(references: Sequence[ReferenceMaterial]) -> list[StyleSnippet]:
    snippets: list[StyleSnippet] = []
    for reference in references:
        for index, paragraph in enumerate(split_paragraphs(reference.text)[:4]):
            snippets.append(
                StyleSnippet(
                    snippet_id=new_id("style-snippet"),
                    text=paragraph[:220],
                    snippet_type="source_style",
                    style_tags=["reference", reference.source_type],
                    chapter_index=index + 1,
                )
            )
    return snippets


def _infer_reference_memory(references: Sequence[ReferenceMaterial]) -> list[MemoryAtom]:
    memories: list[MemoryAtom] = []
    for reference in references:
        canonical = reference.source_type == "target_continuation"
        for paragraph in split_paragraphs(reference.text)[:12]:
            memories.append(
                MemoryAtom(
                    memory_id=new_id("memory"),
                    memory_type="source_excerpt",
                    text=paragraph[:360],
                    canonical=canonical,
                    importance=0.7 if canonical else 0.45,
                    freshness=0.8,
                    state_version_no=0,
                )
            )
    return memories


def _looks_like_chapter_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("第") and any(marker in stripped[:12] for marker in ("章", "节", "回"))


def _first_sentence(text: str) -> str:
    normalized = text.replace("！", "。").replace("？", "。")
    for part in normalized.split("。"):
        candidate = part.strip()
        if candidate:
            return candidate
    return text.strip()
