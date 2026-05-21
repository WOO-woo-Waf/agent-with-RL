"""LLM-backed chunk/chapter/global analysis policy."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from agent_rl.domains.narrative import (
    CharacterCard,
    ChapterAnalysisResult,
    ChunkAnalysisResult,
    GlobalStoryAnalysisResult,
    MemoryAtom,
    NarrativeEvent,
    NarrativeSourceAnalysis,
    PlotThreadState,
    SourceChunk,
    StyleProfile,
    StyleSnippet,
    WorldRule,
)
from agent_rl.llm import ChatModelClient, JsonBlobParser
from agent_rl.narrative_writing.policies.analysis import RuleBasedSourceAnalysisPolicy
from agent_rl.narrative_writing.ports import NarrativeAnalysisRepository
from agent_rl.narrative_writing.prompting import PromptComposer
from agent_rl.narrative_writing.requests import ReferenceMaterial
from agent_rl.narrative_writing.utils import new_id, unique


class LLMDeepNarrativeAnalysisPolicy:
    """Analyzes source text through chunk, chapter, and global LLM passes.

    The policy follows the old project's useful shape but keeps this package's
    OOAD boundary: LLM calls are injected, persistence is a repository port, and
    rule-based analysis remains the fallback path for local tests and failures.
    """

    def __init__(
        self,
        client: ChatModelClient,
        *,
        repository: NarrativeAnalysisRepository | None = None,
        fallback_policy: RuleBasedSourceAnalysisPolicy | None = None,
        prompt_composer: PromptComposer | None = None,
        max_chunk_chars: int = 1800,
        chunk_concurrency: int = 1,
    ) -> None:
        self.client = client
        self.repository = repository
        self.fallback_policy = fallback_policy or RuleBasedSourceAnalysisPolicy(max_chunk_chars=max_chunk_chars)
        self.prompt_composer = prompt_composer or PromptComposer()
        self.parser = JsonBlobParser()
        self.chunk_concurrency = max(1, int(chunk_concurrency))

    def analyze(
        self,
        references: Sequence[ReferenceMaterial],
        *,
        task_id: str,
        story_id: str,
        goal: str,
        writing_direction: str = "",
    ) -> NarrativeSourceAnalysis:
        base = self.fallback_policy.analyze(
            references,
            task_id=task_id,
            story_id=story_id,
            goal=goal,
            writing_direction=writing_direction,
        )
        fallback_reasons: list[str] = []
        chunk_analyses = self._analyze_chunks(
            chunks=base.source_chunks,
            task_id=task_id,
            story_id=story_id,
            goal=goal,
            writing_direction=writing_direction,
            fallback_reasons=fallback_reasons,
        )
        chapter_analyses = self._analyze_chapters(
            chunk_analyses,
            task_id=task_id,
            story_id=story_id,
            goal=goal,
            writing_direction=writing_direction,
            fallback_reasons=fallback_reasons,
        )
        global_analysis = self._analyze_global(
            chapter_analyses,
            task_id=task_id,
            story_id=story_id,
            goal=goal,
            writing_direction=writing_direction,
            fallback_reasons=fallback_reasons,
        )
        enriched = self._merge_into_source_analysis(
            base,
            chunk_analyses=chunk_analyses,
            chapter_analyses=chapter_analyses,
            global_analysis=global_analysis,
            fallback_reasons=fallback_reasons,
        )
        if self.repository is not None:
            self.repository.save_source_analysis(enriched)
        return enriched

    def _analyze_chunks(
        self,
        *,
        chunks: Sequence[SourceChunk],
        task_id: str,
        story_id: str,
        goal: str,
        writing_direction: str,
        fallback_reasons: list[str],
    ) -> list[ChunkAnalysisResult]:
        if self.chunk_concurrency <= 1 or len(chunks) <= 1:
            results: list[ChunkAnalysisResult] = []
            for chunk in chunks:
                results.append(
                    self._analyze_one_chunk(
                        chunk,
                        task_id=task_id,
                        story_id=story_id,
                        goal=goal,
                        writing_direction=writing_direction,
                        previous_context="\n".join(item.summary for item in results[-8:] if item.summary),
                        fallback_reasons=fallback_reasons,
                    )
                )
            return results

        results: list[ChunkAnalysisResult | None] = [None] * len(chunks)

        def run(index: int) -> tuple[int, ChunkAnalysisResult]:
            return index, self._analyze_one_chunk(
                chunks[index],
                task_id=task_id,
                story_id=story_id,
                goal=goal,
                writing_direction=writing_direction,
                previous_context="",
                fallback_reasons=fallback_reasons,
            )

        with ThreadPoolExecutor(max_workers=min(self.chunk_concurrency, len(chunks))) as executor:
            futures = {executor.submit(run, index): index for index in range(len(chunks))}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result_index, result = future.result()
                    results[result_index] = result
                except Exception as exc:
                    fallback_reasons.append(f"parallel chunk {chunks[index].chunk_id} fallback: {exc}")
                    results[index] = _fallback_chunk_result(chunks[index], str(exc))
        return [item for item in results if item is not None]

    def _analyze_one_chunk(
        self,
        chunk: SourceChunk,
        *,
        task_id: str,
        story_id: str,
        goal: str,
        writing_direction: str,
        previous_context: str,
        fallback_reasons: list[str],
    ) -> ChunkAnalysisResult:
        try:
            payload = self._call_json(
                "novel_chunk_analysis",
                {
                    "task_id": task_id,
                    "story_id": story_id,
                    "goal": goal,
                    "writing_direction": writing_direction,
                    "chunk": _chunk_payload(chunk),
                    "previous_context": previous_context[:8000],
                    "output_schema": _chunk_schema(),
                },
            )
            return _chunk_result_from_payload(chunk, payload)
        except Exception as exc:
            fallback_reasons.append(f"novel_chunk_analysis {chunk.chunk_id} fallback: {exc}")
            return _fallback_chunk_result(chunk, str(exc))

    def _analyze_chapters(
        self,
        chunk_analyses: Sequence[ChunkAnalysisResult],
        *,
        task_id: str,
        story_id: str,
        goal: str,
        writing_direction: str,
        fallback_reasons: list[str],
    ) -> list[ChapterAnalysisResult]:
        by_chapter: dict[int, list[ChunkAnalysisResult]] = defaultdict(list)
        for item in chunk_analyses:
            by_chapter[int(item.chapter_index or 1)].append(item)
        results: list[ChapterAnalysisResult] = []
        for chapter_index in sorted(by_chapter):
            rows = by_chapter[chapter_index]
            try:
                payload = self._call_json(
                    "novel_chapter_analysis",
                    {
                        "task_id": task_id,
                        "story_id": story_id,
                        "goal": goal,
                        "writing_direction": writing_direction,
                        "chapter_index": chapter_index,
                        "chunk_analyses": [_compact_chunk_analysis(item) for item in rows],
                        "output_schema": _chapter_schema(),
                    },
                )
                results.append(_chapter_result_from_payload(chapter_index, rows, payload))
            except Exception as exc:
                fallback_reasons.append(f"novel_chapter_analysis chapter {chapter_index} fallback: {exc}")
                results.append(_fallback_chapter_result(chapter_index, rows, str(exc)))
        return results

    def _analyze_global(
        self,
        chapter_analyses: Sequence[ChapterAnalysisResult],
        *,
        task_id: str,
        story_id: str,
        goal: str,
        writing_direction: str,
        fallback_reasons: list[str],
    ) -> GlobalStoryAnalysisResult:
        try:
            payload = self._call_json(
                "novel_global_analysis",
                {
                    "task_id": task_id,
                    "story_id": story_id,
                    "goal": goal,
                    "writing_direction": writing_direction,
                    "chapter_analyses": [_compact_chapter_analysis(item) for item in chapter_analyses],
                    "output_schema": _global_schema(),
                },
            )
            return _global_result_from_payload(story_id, chapter_analyses, payload)
        except Exception as exc:
            fallback_reasons.append(f"novel_global_analysis fallback: {exc}")
            return _fallback_global_result(story_id, chapter_analyses, str(exc))

    def _call_json(self, purpose: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self.prompt_composer.compose_system_prompt(purpose=purpose)
        messages = [
            {"role": "system", "content": prompt.system_content},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        raw = self.client.complete(messages, purpose=purpose, json_mode=True)
        parsed = self.parser.parse(raw)
        if not isinstance(parsed.data, dict):
            raise ValueError(f"{purpose} returned non-object JSON")
        data = dict(parsed.data)
        data.setdefault("_parse_metadata", {"repaired": parsed.repaired, "source": parsed.source})
        data.setdefault("_prompt_metadata", prompt.metadata)
        return data

    def _merge_into_source_analysis(
        self,
        base: NarrativeSourceAnalysis,
        *,
        chunk_analyses: list[ChunkAnalysisResult],
        chapter_analyses: list[ChapterAnalysisResult],
        global_analysis: GlobalStoryAnalysisResult,
        fallback_reasons: list[str],
    ) -> NarrativeSourceAnalysis:
        base.chunk_analyses = chunk_analyses
        base.chapter_analyses = chapter_analyses
        base.global_analysis = global_analysis
        base.characters = _characters_from_global(global_analysis) or base.characters
        base.events = _events_from_chapters(chapter_analyses) or base.events
        base.plot_threads = _plot_threads_from_global(global_analysis) or base.plot_threads
        base.world_rules = _world_rules_from_global(global_analysis) or base.world_rules
        base.style_profile = _style_profile_from_global(global_analysis) or base.style_profile
        base.style_snippets = _style_snippets_from_chunks(chunk_analyses) or base.style_snippets
        base.memory_atoms = _memory_from_analysis(global_analysis, chapter_analyses) or base.memory_atoms
        base.coverage.update(
            {
                "llm_chunk_analysis_count": float(len(chunk_analyses)),
                "llm_chapter_analysis_count": float(len(chapter_analyses)),
                "llm_global_analysis_count": 1.0,
                "llm_fallback_count": float(len(fallback_reasons)),
            }
        )
        base.trace.append(
            {
                "policy": self.__class__.__name__,
                "chunk_concurrency": self.chunk_concurrency,
                "fallback_reasons": list(fallback_reasons),
                "repository": self.repository.__class__.__name__ if self.repository is not None else "",
            }
        )
        return base


def _chunk_payload(chunk: SourceChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_type": chunk.source_type,
        "chapter_index": chunk.chapter_index,
        "chunk_index": chunk.chunk_index,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "text": chunk.text,
        "metadata": dict(chunk.metadata),
    }


def _chunk_result_from_payload(chunk: SourceChunk, payload: dict[str, Any]) -> ChunkAnalysisResult:
    evidence = _dict(payload.get("evidence"))
    return ChunkAnalysisResult(
        analysis_id=new_id("chunk-analysis"),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chapter_index=_int(payload.get("chapter_index"), chunk.chapter_index),
        chunk_index=chunk.chunk_index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        summary=str(payload.get("summary") or evidence.get("embedding_summary") or "")[:1000],
        key_events=_str_list(payload.get("events"), keys=("summary", "event", "text"), limit=20),
        open_questions=_str_list(payload.get("open_questions"), limit=12),
        character_mentions=_str_list(payload.get("characters"), keys=("name", "summary"), limit=24),
        world_rule_candidates=_str_list(payload.get("world_facts"), keys=("rule_text", "text"), limit=16),
        plot_thread_candidates=_str_list(payload.get("plot_threads"), keys=("name", "summary"), limit=16),
        scene_state=_dict(payload.get("scene")),
        style_features=_dict(payload.get("style")),
        source_evidence=_evidence_rows(evidence),
        retrieval_keywords=_str_list(evidence.get("retrieval_keywords"), limit=32),
        confidence=_confidence(payload),
        raw_payload=payload,
    )


def _chapter_result_from_payload(
    chapter_index: int,
    chunks: Sequence[ChunkAnalysisResult],
    payload: dict[str, Any],
) -> ChapterAnalysisResult:
    return ChapterAnalysisResult(
        analysis_id=new_id("chapter-analysis"),
        chapter_index=_int(payload.get("chapter_index"), chapter_index) or chapter_index,
        chapter_title=str(payload.get("chapter_title") or f"Chapter {chapter_index}"),
        source_start_offset=min((item.start_offset for item in chunks), default=0),
        source_end_offset=max((item.end_offset for item in chunks), default=0),
        chunk_ids=[item.chunk_id for item in chunks],
        chapter_summary=str(payload.get("chapter_summary") or "")[:1600],
        chapter_events=_str_list(payload.get("chapter_events"), limit=32),
        characters_involved=_str_list(payload.get("characters_involved"), limit=32),
        character_state_updates=_state_updates(payload.get("character_state_updates")),
        relationship_updates=_dict_rows(payload.get("relationship_updates"), limit=24),
        scene_sequence=_dict_rows(payload.get("scene_sequence"), limit=32),
        world_rules_confirmed=_str_list(payload.get("world_rules_confirmed"), limit=24),
        setting_concepts=_dict_rows(payload.get("setting_concepts"), limit=64),
        foreshadowing=_dict_rows(payload.get("foreshadowing"), fallback_key="seed_text", limit=24),
        open_questions=_str_list(payload.get("open_questions"), limit=24),
        scene_markers=_str_list(payload.get("scene_markers"), limit=24),
        style_profile_override=_dict(payload.get("style_profile_override")),
        chapter_synopsis=str(payload.get("chapter_synopsis") or payload.get("chapter_summary") or "")[:1000],
        retrieval_keywords=_str_list(payload.get("retrieval_keywords"), limit=48),
        coverage=_dict(payload.get("state_completeness")),
        raw_payload=payload,
    )


def _global_result_from_payload(
    story_id: str,
    chapters: Sequence[ChapterAnalysisResult],
    payload: dict[str, Any],
) -> GlobalStoryAnalysisResult:
    settings = _dict(payload.get("setting_systems"))
    return GlobalStoryAnalysisResult(
        analysis_id=new_id("global-analysis"),
        story_id=str(payload.get("story_id") or story_id),
        title=str(payload.get("title") or ""),
        story_synopsis=str(payload.get("story_synopsis") or payload.get("task_summary") or "")[:6000],
        chapter_count=len(chapters),
        character_registry=_dict_rows(payload.get("character_cards"), limit=200),
        relationship_graph=_dict_rows(payload.get("relationship_graph"), limit=300),
        plot_threads=_dict_rows(payload.get("plot_threads"), limit=120),
        world_rules=_dict_rows(payload.get("world_rules"), fallback_key="rule_text", limit=200),
        setting_systems={str(key): _dict_rows(value, fallback_key="name", limit=200) for key, value in settings.items()},
        locations=_dict_rows(payload.get("locations"), limit=160),
        objects=_dict_rows(payload.get("objects"), limit=160),
        organizations=_dict_rows(payload.get("organizations"), limit=120),
        foreshadowing_states=_dict_rows(payload.get("foreshadowing_states"), fallback_key="seed_text", limit=160),
        scene_case_library=_dict_rows(payload.get("narrative_cases"), fallback_key="summary", limit=120),
        retrieval_index_suggestions=_dict_rows(payload.get("retrieval_index_suggestions"), fallback_key="text", limit=240),
        continuity_constraints=_str_list(payload.get("continuation_constraints"), limit=120),
        style_profile=_dict(payload.get("style_bible") or payload.get("style_profile")),
        global_open_questions=_str_list(payload.get("global_open_questions"), limit=32),
        chapter_index_map=_dict(payload.get("chapter_index_map")) or _chapter_index_map(chapters),
        analysis_coverage=_dict(payload.get("state_completeness")),
        raw_payload=payload,
    )


def _fallback_chunk_result(chunk: SourceChunk, reason: str) -> ChunkAnalysisResult:
    summary = _first_non_empty(_sentences(chunk.text)[:2]) or chunk.text[:500]
    return ChunkAnalysisResult(
        analysis_id=new_id("chunk-analysis"),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chapter_index=chunk.chapter_index,
        chunk_index=chunk.chunk_index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        summary=summary,
        key_events=[summary] if summary else [],
        source_evidence=[{"evidence_type": "source_quote", "text": item} for item in _sentences(chunk.text)[:4]],
        confidence=0.2,
        fallback_reason=reason,
        raw_payload={"fallback_reason": reason},
    )


def _fallback_chapter_result(
    chapter_index: int,
    chunks: Sequence[ChunkAnalysisResult],
    reason: str,
) -> ChapterAnalysisResult:
    summary = " ".join(item.summary for item in chunks if item.summary)[:1000]
    return ChapterAnalysisResult(
        analysis_id=new_id("chapter-analysis"),
        chapter_index=chapter_index,
        chapter_title=f"Chapter {chapter_index}",
        source_start_offset=min((item.start_offset for item in chunks), default=0),
        source_end_offset=max((item.end_offset for item in chunks), default=0),
        chunk_ids=[item.chunk_id for item in chunks],
        chapter_summary=summary,
        chapter_synopsis=summary[:800],
        chapter_events=unique(event for item in chunks for event in item.key_events)[:32],
        characters_involved=unique(name for item in chunks for name in item.character_mentions)[:32],
        open_questions=unique(question for item in chunks for question in item.open_questions)[:24],
        coverage={"fallback_reason": reason, "confidence": 0.25},
        fallback_reason=reason,
        raw_payload={"fallback_reason": reason},
    )


def _fallback_global_result(
    story_id: str,
    chapters: Sequence[ChapterAnalysisResult],
    reason: str,
) -> GlobalStoryAnalysisResult:
    synopsis = "\n".join(
        f"Chapter {item.chapter_index}: {item.chapter_synopsis or item.chapter_summary}"
        for item in chapters
        if item.chapter_synopsis or item.chapter_summary
    )[:6000]
    return GlobalStoryAnalysisResult(
        analysis_id=new_id("global-analysis"),
        story_id=story_id,
        story_synopsis=synopsis,
        chapter_count=len(chapters),
        plot_threads=[{"thread_id": "plot-main", "name": "main plot", "stage": "open"}],
        global_open_questions=unique(question for item in chapters for question in item.open_questions)[:32],
        chapter_index_map=_chapter_index_map(chapters),
        analysis_coverage={"fallback_reason": reason, "confidence": 0.3},
        fallback_reason=reason,
        raw_payload={"fallback_reason": reason},
    )


def _characters_from_global(global_analysis: GlobalStoryAnalysisResult) -> list[CharacterCard]:
    cards: list[CharacterCard] = []
    for index, row in enumerate(global_analysis.character_registry, start=1):
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        cards.append(
            CharacterCard(
                character_id=str(row.get("character_id") or f"char-{index:03d}"),
                name=name,
                aliases=_str_list(row.get("aliases"), limit=8),
                role_type=str(row.get("role_type") or row.get("role") or ""),
                identity_tags=_str_list(row.get("identity_tags"), limit=12),
                stable_traits=_str_list(row.get("stable_traits"), limit=12),
                wounds_or_fears=_str_list(row.get("wounds_or_fears"), limit=12),
                values=_str_list(row.get("values"), limit=12),
                moral_boundaries=_str_list(row.get("moral_boundaries"), limit=12),
                current_goals=_str_list(row.get("current_goals") or row.get("goals"), limit=12),
                hidden_goals=_str_list(row.get("hidden_goals"), limit=12),
                knowledge_boundary=_str_list(row.get("knowledge_boundary"), limit=12),
                voice_profile=_str_list(row.get("voice_profile"), limit=12),
                dialogue_do=_str_list(row.get("dialogue_do"), limit=12),
                dialogue_do_not=_str_list(row.get("dialogue_do_not"), limit=12),
                gesture_patterns=_str_list(row.get("gesture_patterns"), limit=12),
                decision_patterns=_str_list(row.get("decision_patterns"), limit=12),
                forbidden_actions=_str_list(row.get("forbidden_actions"), limit=12),
                source_span_ids=_str_list(row.get("source_span_ids"), limit=24),
            )
        )
    return cards


def _events_from_chapters(chapters: Sequence[ChapterAnalysisResult]) -> list[NarrativeEvent]:
    events: list[NarrativeEvent] = []
    order = 1
    for chapter in chapters:
        for text in chapter.chapter_events:
            events.append(
                NarrativeEvent(
                    event_id=new_id("event"),
                    summary=text,
                    event_type="llm_chapter_event",
                    chapter_index=chapter.chapter_index,
                    timeline_order=order,
                    participants=list(chapter.characters_involved),
                )
            )
            order += 1
    return events


def _plot_threads_from_global(global_analysis: GlobalStoryAnalysisResult) -> list[PlotThreadState]:
    threads: list[PlotThreadState] = []
    for index, row in enumerate(global_analysis.plot_threads, start=1):
        name = str(row.get("name") or row.get("thread_name") or "").strip()
        if not name:
            continue
        threads.append(
            PlotThreadState(
                thread_id=str(row.get("thread_id") or f"plot-{index:03d}"),
                name=name,
                stage=str(row.get("stage") or ""),
                stakes=str(row.get("stakes") or ""),
                open_questions=_str_list(row.get("open_questions"), limit=16),
                anchor_event_ids=_str_list(row.get("anchor_events"), limit=16),
                next_expected_beats=_str_list(row.get("next_expected_beats"), limit=16),
            )
        )
    return threads


def _world_rules_from_global(global_analysis: GlobalStoryAnalysisResult) -> list[WorldRule]:
    rules: list[WorldRule] = []
    for index, row in enumerate(global_analysis.world_rules, start=1):
        text = str(row.get("rule_text") or row.get("text") or "").strip()
        if not text:
            continue
        rules.append(
            WorldRule(
                rule_id=str(row.get("rule_id") or f"rule-{index:03d}"),
                rule_text=text,
                rule_type=str(row.get("rule_type") or "soft"),
                stability=str(row.get("status") or "candidate"),  # type: ignore[arg-type]
                source_span_ids=_str_list(row.get("source_span_ids"), limit=24),
            )
        )
    return rules


def _style_profile_from_global(global_analysis: GlobalStoryAnalysisResult) -> StyleProfile | None:
    style = global_analysis.style_profile
    if not style:
        return None
    return StyleProfile(
        profile_id=new_id("style"),
        narrative_pov=str(style.get("narrative_pov") or style.get("pov") or ""),
        tense=str(style.get("tense") or ""),
        narrative_distance=str(style.get("narrative_distance") or ""),
        sentence_length_distribution=_dict(style.get("sentence_length_distribution")),
        paragraph_length_distribution=_dict(style.get("paragraph_length_distribution")),
        dialogue_ratio=_float(style.get("dialogue_ratio"), 0.0),
        description_mix=_dict(style.get("description_mix")),
        rhetoric_markers=_str_list(style.get("rhetoric_markers"), limit=24),
        lexical_fingerprint=_str_list(style.get("lexical_fingerprint"), limit=48),
        pacing_profile=_dict(style.get("pacing_profile")),
        forbidden_patterns=_str_list(style.get("negative_style_rules") or style.get("forbidden_patterns"), limit=24),
    )


def _style_snippets_from_chunks(chunks: Sequence[ChunkAnalysisResult]) -> list[StyleSnippet]:
    snippets: list[StyleSnippet] = []
    for chunk in chunks:
        for index, evidence in enumerate(chunk.source_evidence[:6], start=1):
            text = str(evidence.get("text") or "")
            if not text:
                continue
            snippets.append(
                StyleSnippet(
                    snippet_id=f"{chunk.chunk_id}-style-{index:03d}",
                    text=text[:600],
                    snippet_type=str(evidence.get("evidence_type") or "source_quote"),
                    style_tags=["llm_analysis"],
                    chapter_index=chunk.chapter_index,
                )
            )
    return snippets


def _memory_from_analysis(
    global_analysis: GlobalStoryAnalysisResult,
    chapters: Sequence[ChapterAnalysisResult],
) -> list[MemoryAtom]:
    atoms: list[MemoryAtom] = []
    if global_analysis.story_synopsis:
        atoms.append(
            MemoryAtom(
                memory_id=new_id("memory"),
                memory_type="global_story_synopsis",
                text=global_analysis.story_synopsis,
                importance=0.95,
                freshness=1.0,
            )
        )
    for chapter in chapters:
        text = chapter.chapter_synopsis or chapter.chapter_summary
        if text:
            atoms.append(
                MemoryAtom(
                    memory_id=new_id("memory"),
                    memory_type="chapter_synopsis",
                    text=text,
                    importance=0.8,
                    freshness=0.8,
                    related_entities=list(chapter.characters_involved),
                )
            )
    return atoms


def _chunk_schema() -> dict[str, Any]:
    return {
        "summary": "string",
        "scene": {"location": "string", "time": "string", "atmosphere": ["string"], "scene_function": "string"},
        "characters": [{"name": "string", "goal": "string", "emotion": "string", "knowledge": ["string"]}],
        "events": [{"summary": "string", "cause": "string", "effect": "string", "participants": ["string"]}],
        "relationship_updates": ["string"],
        "world_facts": ["string"],
        "plot_threads": ["string"],
        "foreshadowing": ["string"],
        "open_questions": ["string"],
        "style": {"pov": "string", "sentence_rhythm": "string", "dialogue_style": "string"},
        "evidence": {"source_quotes": ["string"], "style_snippets": ["string"], "retrieval_keywords": ["string"]},
        "state_completeness": {"covered_dimensions": ["string"], "missing_dimensions": ["string"], "confidence": 0.0},
    }


def _chapter_schema() -> dict[str, Any]:
    return {
        "chapter_index": "number",
        "chapter_title": "string",
        "chapter_summary": "string",
        "chapter_synopsis": "string",
        "scene_sequence": [{"location": "string", "characters": ["string"], "goal": "string", "outcome": "string"}],
        "chapter_events": ["string"],
        "characters_involved": ["string"],
        "character_state_updates": {},
        "relationship_updates": ["string"],
        "plot_progress": ["string"],
        "world_rules_confirmed": ["string"],
        "setting_concepts": [{"name": "string", "definition": "string", "confidence": 0.0}],
        "foreshadowing": ["string"],
        "open_questions": ["string"],
        "scene_markers": ["string"],
        "style_profile_override": {},
        "retrieval_keywords": ["string"],
        "state_completeness": {"confidence": 0.0},
    }


def _global_schema() -> dict[str, Any]:
    return {
        "story_id": "string",
        "title": "string",
        "story_synopsis": "string",
        "character_cards": [{"character_id": "string", "name": "string", "current_goals": ["string"]}],
        "relationship_graph": [{"source": "string", "target": "string", "status": "string"}],
        "plot_threads": [{"thread_id": "string", "name": "string", "stage": "string", "open_questions": ["string"]}],
        "world_rules": [{"rule_id": "string", "rule_text": "string", "status": "candidate|confirmed"}],
        "setting_systems": {},
        "locations": [],
        "objects": [],
        "organizations": [],
        "timeline": ["string"],
        "foreshadowing_states": [{"seed_text": "string", "status": "candidate|planted|revealed"}],
        "style_bible": {},
        "narrative_cases": [],
        "continuation_constraints": ["string"],
        "retrieval_index_suggestions": [{"evidence_type": "string", "text": "string", "keywords": ["string"]}],
        "state_completeness": {"overall_score": 0.0, "missing_dimensions": ["string"]},
    }


def _compact_chunk_analysis(item: ChunkAnalysisResult) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "chapter_index": item.chapter_index,
        "summary": item.summary,
        "key_events": item.key_events,
        "open_questions": item.open_questions,
        "character_mentions": item.character_mentions,
        "world_rule_candidates": item.world_rule_candidates,
        "plot_thread_candidates": item.plot_thread_candidates,
        "scene_state": item.scene_state,
        "style_features": item.style_features,
        "retrieval_keywords": item.retrieval_keywords,
    }


def _compact_chapter_analysis(item: ChapterAnalysisResult) -> dict[str, Any]:
    return {
        "chapter_index": item.chapter_index,
        "chapter_title": item.chapter_title,
        "chapter_summary": item.chapter_summary,
        "chapter_synopsis": item.chapter_synopsis,
        "chapter_events": item.chapter_events,
        "characters_involved": item.characters_involved,
        "character_state_updates": item.character_state_updates,
        "relationship_updates": item.relationship_updates,
        "scene_sequence": item.scene_sequence,
        "world_rules_confirmed": item.world_rules_confirmed,
        "setting_concepts": item.setting_concepts,
        "foreshadowing": item.foreshadowing,
        "open_questions": item.open_questions,
        "retrieval_keywords": item.retrieval_keywords,
    }


def _chapter_index_map(chapters: Sequence[ChapterAnalysisResult]) -> dict[str, Any]:
    return {
        str(item.chapter_index): {
            "chapter_title": item.chapter_title,
            "chapter_summary": item.chapter_summary,
            "chapter_synopsis": item.chapter_synopsis,
            "open_questions": item.open_questions,
        }
        for item in chapters
    }


def _evidence_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("source_quotes", "style_snippets", "scene_cases"):
        for text in _str_list(evidence.get(key), limit=10):
            rows.append({"evidence_type": key[:-1], "text": text})
    return rows


def _dict_rows(value: Any, *, fallback_key: str = "summary", limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _list(value):
        if isinstance(item, dict):
            rows.append(dict(item))
        else:
            text = str(item or "").strip()
            if text:
                rows.append({fallback_key: text})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _state_updates(value: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for key, raw in _dict(value).items():
        clean = str(key).strip()
        if clean:
            result[clean] = _str_list(raw, limit=24)
    return result


def _str_list(value: Any, *, keys: Sequence[str] = ("summary", "name", "text"), limit: int | None = None) -> list[str]:
    result: list[str] = []
    for item in _list(value):
        if isinstance(item, dict):
            text = next((str(item.get(key) or "").strip() for key in keys if str(item.get(key) or "").strip()), "")
            if not text:
                text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            text = str(item or "").strip()
        if text:
            result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _confidence(payload: dict[str, Any]) -> float:
    completeness = _dict(payload.get("state_completeness"))
    return _float(payload.get("confidence") or completeness.get("confidence"), 0.7)


def _sentences(text: str) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n")
    for mark in ("。", "！", "？", ".", "!", "?"):
        normalized = normalized.replace(mark, mark + "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _first_non_empty(items: Sequence[str]) -> str:
    return " ".join(item for item in items if item).strip()
