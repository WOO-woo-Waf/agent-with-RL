"""Local task-aware retrieval policies."""

from __future__ import annotations

from dataclasses import dataclass

from agent_rl.domains.narrative import EvidencePack, NarrativeEvidence, NarrativeQuery, NarrativeTaskState
from agent_rl.narrative_writing.utils import new_id, tokenize


SOURCE_TYPE_WEIGHTS = {
    "target_continuation": 1.0,
    "crossover_linkage": 0.85,
    "reference_character": 0.8,
    "reference_world": 0.78,
    "same_author_world_style": 0.72,
    "reference_style": 0.68,
    "reference_plot": 0.68,
}


@dataclass(frozen=True)
class RetrievalQuota:
    """Per-channel evidence limits for the local composite retriever."""

    author: int = 4
    character: int = 5
    plot: int = 5
    world: int = 4
    style: int = 4
    source: int = 6
    scene_case: int = 3


class CompositeNarrativeRetrievalPolicy:
    """Multi-channel local retrieval with source-type weighting and quotas."""

    def __init__(self, quota: RetrievalQuota | None = None) -> None:
        self.quota = quota or RetrievalQuota()

    def retrieve(self, state: NarrativeTaskState, query: NarrativeQuery) -> EvidencePack:
        query_terms = tokenize(query.query_text)
        pack = EvidencePack(pack_id=new_id("evidence-pack"), query_id=query.query_id)

        author_candidates = _author_plan_evidence(state)
        character_candidates = _character_evidence(state, query_terms)
        plot_candidates = _plot_and_memory_evidence(state, query_terms)
        world_candidates = _world_evidence(state, query_terms)
        style_candidates = _style_evidence(state, query_terms)
        source_candidates = _source_chunk_evidence(state, query_terms)

        pack.author_plan_evidence.extend(_select(author_candidates, self.quota.author))
        pack.character_evidence.extend(_select(character_candidates, self.quota.character))
        pack.plot_evidence.extend(_select([*plot_candidates, *source_candidates], self.quota.plot + self.quota.source))
        pack.world_evidence.extend(_select(world_candidates, self.quota.world))
        pack.style_evidence.extend(_select(style_candidates, self.quota.style))
        pack.scene_case_evidence.extend(_select(_scene_case_evidence(state, query_terms), self.quota.scene_case))
        pack.retrieval_trace.append(
            {
                "policy": self.__class__.__name__,
                "query_terms": sorted(query_terms),
                "candidate_counts": {
                    "author": len(author_candidates),
                    "character": len(character_candidates),
                    "plot_memory": len(plot_candidates),
                    "source": len(source_candidates),
                    "world": len(world_candidates),
                    "style": len(style_candidates),
                    "scene_case": len(pack.scene_case_evidence),
                },
                "selected_counts": {
                    "author": len(pack.author_plan_evidence),
                    "character": len(pack.character_evidence),
                    "plot": len(pack.plot_evidence),
                    "world": len(pack.world_evidence),
                    "style": len(pack.style_evidence),
                    "scene_case": len(pack.scene_case_evidence),
                },
                "quota": self.quota.__dict__,
            }
        )
        return pack


class KeywordNarrativeRetrievalPolicy:
    """Small local retrieval policy using entity and keyword overlap."""

    def retrieve(self, state: NarrativeTaskState, query: NarrativeQuery) -> EvidencePack:
        query_terms = tokenize(query.query_text)
        pack = EvidencePack(pack_id=new_id("evidence-pack"), query_id=query.query_id)
        pack.author_plan_evidence.extend(
            NarrativeEvidence(
                evidence_id=new_id("ev-author"),
                evidence_type="author_constraint",
                source="author_constraints",
                text=constraint.text,
                usage_hint="must_follow" if constraint.violation_policy == "block_commit" else "prefer",
                score_author_plan=1.0,
                final_score=1.0,
            )
            for constraint in state.author_constraints
            if constraint.status == "confirmed"
        )
        pack.character_evidence.extend(
            _score_evidence(
                evidence_type="character_profile",
                source="characters",
                text=f"{character.name}: traits={character.stable_traits}; goals={character.current_goals}; voice={character.voice_profile}",
                query_terms=query_terms,
                related_entities=[character.character_id],
            )
            for character in state.characters
        )
        pack.plot_evidence.extend(
            _score_evidence(
                evidence_type="plot_thread",
                source="plot_threads",
                text=f"{thread.name}: stage={thread.stage}; open={thread.open_questions}; next={thread.next_expected_beats}",
                query_terms=query_terms,
                related_plot_threads=[thread.thread_id],
            )
            for thread in state.plot_threads
        )
        pack.world_evidence.extend(
            _score_evidence("world_rule", "world_rules", rule.rule_text, query_terms)
            for rule in state.world_rules
            if rule.stability in ("confirmed", "candidate")
        )
        pack.style_evidence.extend(
            _score_evidence("style_snippet", "style_snippets", snippet.text, query_terms)
            for snippet in state.style_snippets
        )
        pack.plot_evidence.extend(
            _score_evidence("compressed_memory", "compressed_memory", f"{memory.summary}; {memory.key_points}", query_terms)
            for memory in state.compressed_memory
        )
        pack.plot_evidence.extend(
            _score_evidence("source_memory", "memory_atoms", memory.text, query_terms)
            for memory in state.memory_atoms
            if memory.canonical
        )
        pack.retrieval_trace.append(
            {
                "policy": self.__class__.__name__,
                "query_terms": sorted(query_terms),
                "selected_counts": {
                    "author": len(pack.author_plan_evidence),
                    "character": len(pack.character_evidence),
                    "plot": len(pack.plot_evidence),
                    "world": len(pack.world_evidence),
                    "style": len(pack.style_evidence),
                },
            }
        )
        _trim_pack(pack)
        return pack


def _author_plan_evidence(state: NarrativeTaskState) -> list[NarrativeEvidence]:
    return [
        NarrativeEvidence(
            evidence_id=new_id("ev-author"),
            evidence_type="author_constraint",
            source="author_constraints",
            text=constraint.text,
            usage_hint="must_follow" if constraint.violation_policy == "block_commit" else "prefer",
            score_author_plan=1.0,
            final_score=1.0,
        )
        for constraint in state.author_constraints
        if constraint.status == "confirmed"
    ]


def _character_evidence(state: NarrativeTaskState, query_terms: set[str]) -> list[NarrativeEvidence]:
    return [
        _score_evidence(
            evidence_type="character_profile",
            source="characters",
            text=f"{character.name}: traits={character.stable_traits}; goals={character.current_goals}; voice={character.voice_profile}; knowledge={character.knowledge_boundary}",
            query_terms=query_terms,
            related_entities=[character.character_id],
        )
        for character in state.characters
    ]


def _plot_and_memory_evidence(state: NarrativeTaskState, query_terms: set[str]) -> list[NarrativeEvidence]:
    candidates = [
        _score_evidence(
            evidence_type="plot_thread",
            source="plot_threads",
            text=f"{thread.name}: stage={thread.stage}; open={thread.open_questions}; next={thread.next_expected_beats}; blocked={thread.blocked_beats}",
            query_terms=query_terms,
            related_plot_threads=[thread.thread_id],
        )
        for thread in state.plot_threads
    ]
    candidates.extend(
        _score_evidence("compressed_memory", "compressed_memory", f"{memory.summary}; {memory.key_points}", query_terms)
        for memory in state.compressed_memory
    )
    candidates.extend(
        _score_evidence(
            "source_memory",
            "memory_atoms",
            memory.text,
            query_terms,
            source_weight=1.0 if memory.canonical else 0.72,
        )
        for memory in state.memory_atoms
    )
    candidates.extend(
        _score_evidence(
            "source_event",
            "events",
            event.summary,
            query_terms,
            related_entities=event.participants,
            related_plot_threads=event.plot_thread_ids,
        )
        for event in state.events
        if event.is_canonical
    )
    return candidates


def _world_evidence(state: NarrativeTaskState, query_terms: set[str]) -> list[NarrativeEvidence]:
    return [
        _score_evidence("world_rule", "world_rules", rule.rule_text, query_terms)
        for rule in state.world_rules
        if rule.stability in ("confirmed", "candidate")
    ]


def _style_evidence(state: NarrativeTaskState, query_terms: set[str]) -> list[NarrativeEvidence]:
    return [
        _score_evidence("style_snippet", "style_snippets", snippet.text, query_terms)
        for snippet in state.style_snippets
    ]


def _source_chunk_evidence(state: NarrativeTaskState, query_terms: set[str]) -> list[NarrativeEvidence]:
    return [
        _score_evidence(
            "source_chunk",
            f"source_chunks:{chunk.source_type}",
            chunk.text,
            query_terms,
            chapter_index=chunk.chapter_index,
            source_weight=SOURCE_TYPE_WEIGHTS.get(chunk.source_type, 0.6),
        )
        for chunk in state.source_chunks
    ]


def _scene_case_evidence(state: NarrativeTaskState, query_terms: set[str]) -> list[NarrativeEvidence]:
    cases: list[NarrativeEvidence] = []
    for event in state.events:
        if not event.summary.strip():
            continue
        cases.append(
            _score_evidence(
                "scene_case",
                "events",
                event.summary,
                query_terms,
                related_entities=event.participants,
                related_plot_threads=event.plot_thread_ids,
                chapter_index=event.chapter_index,
                source_weight=1.0 if event.is_canonical else 0.7,
            )
        )
    return cases


def _score_evidence(
    evidence_type: str,
    source: str,
    text: str,
    query_terms: set[str],
    related_entities: list[str] | None = None,
    related_plot_threads: list[str] | None = None,
    chapter_index: int | None = None,
    source_weight: float = 1.0,
) -> NarrativeEvidence:
    terms = tokenize(text)
    overlap = len(terms & query_terms)
    score = overlap / max(1, len(query_terms))
    final_score = min(1.0, (score + 0.05) * source_weight)
    return NarrativeEvidence(
        evidence_id=new_id("ev"),
        evidence_type=evidence_type,
        source=source,
        text=text,
        usage_hint="context",
        related_entities=related_entities or [],
        related_plot_threads=related_plot_threads or [],
        chapter_index=chapter_index,
        score_structural=score,
        final_score=final_score,
    )


def _select(items: list[NarrativeEvidence], limit: int) -> list[NarrativeEvidence]:
    deduped: dict[tuple[str, str], NarrativeEvidence] = {}
    for item in items:
        key = (item.evidence_type, item.text)
        current = deduped.get(key)
        if current is None or item.final_score > current.final_score:
            deduped[key] = item
    selected = sorted(deduped.values(), key=lambda item: item.final_score, reverse=True)
    return selected[:limit]


def _trim_pack(pack: EvidencePack, limit: int = 6) -> None:
    for items in (
        pack.style_evidence,
        pack.character_evidence,
        pack.plot_evidence,
        pack.world_evidence,
        pack.author_plan_evidence,
        pack.scene_case_evidence,
    ):
        items.sort(key=lambda item: item.final_score, reverse=True)
        del items[limit:]
