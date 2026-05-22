"""Local task-aware retrieval policies."""

from __future__ import annotations

from dataclasses import dataclass

from agent_rl.domains.narrative import EvidencePack, NarrativeEvidence, NarrativeQuery, NarrativeTaskState
from agent_rl.narrative_writing.ports import NarrativeMemoryRepository
from agent_rl.narrative_writing.utils import new_id, tokenize
from agent_rl.rag import RAGModelService, VectorSearchResult


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


class SQLiteFTSNarrativeRetrievalPolicy:
    """Retrieves persisted memory evidence before local in-state fallback."""

    def __init__(
        self,
        memory_repository: NarrativeMemoryRepository,
        *,
        fallback: CompositeNarrativeRetrievalPolicy | None = None,
        limit: int = 12,
    ) -> None:
        self.memory_repository = memory_repository
        self.fallback = fallback or CompositeNarrativeRetrievalPolicy()
        self.limit = limit

    def retrieve(self, state: NarrativeTaskState, query: NarrativeQuery) -> EvidencePack:
        pack = self.fallback.retrieve(state, query)
        persisted = self.memory_repository.search(state.story_id, query.query_text, limit=self.limit)
        pack.plot_evidence = _select([*persisted, *pack.plot_evidence], self.limit)
        pack.retrieval_trace.append(
            {
                "policy": self.__class__.__name__,
                "persisted_memory_count": len(persisted),
                "memory_repository": self.memory_repository.__class__.__name__,
            }
        )
        return pack


class RAGVectorNarrativeRetrievalPolicy:
    """Merges vector RAG service results with local structural evidence."""

    def __init__(
        self,
        rag_service: RAGModelService,
        *,
        fallback: CompositeNarrativeRetrievalPolicy | None = None,
        collection_id: str = "narrative",
        limit: int = 12,
        rerank: bool = True,
    ) -> None:
        self.rag_service = rag_service
        self.fallback = fallback or CompositeNarrativeRetrievalPolicy()
        self.collection_id = collection_id
        self.limit = limit
        self.rerank = rerank

    def retrieve(self, state: NarrativeTaskState, query: NarrativeQuery) -> EvidencePack:
        pack = self.fallback.retrieve(state, query)
        try:
            rows = self.rag_service.search(
                query.query_text,
                story_id=state.story_id,
                evidence_types=query.required_evidence_types or None,
                collection_id=self.collection_id,
                limit=self.limit,
                rerank=self.rerank,
            )
        except Exception as exc:  # noqa: BLE001 - retrieval should degrade to local evidence.
            pack.retrieval_trace.append(
                {
                    "policy": self.__class__.__name__,
                    "status": "failed",
                    "reason": str(exc),
                    "fallback": self.fallback.__class__.__name__,
                }
            )
            return pack
        _merge_vector_rows(pack, rows)
        pack.retrieval_trace.append(
            {
                "policy": self.__class__.__name__,
                "status": "succeeded",
                "vector_result_count": len(rows),
                "collection_id": self.collection_id,
                "rerank": self.rerank,
            }
        )
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


def _merge_vector_rows(pack: EvidencePack, rows: list[VectorSearchResult]) -> None:
    for row in rows:
        evidence = NarrativeEvidence(
            evidence_id=row.evidence_id,
            evidence_type=row.evidence_type,
            source=row.source or "rag_vector",
            text=row.text,
            usage_hint="vector_rag",
            related_entities=list(row.related_entities),
            related_plot_threads=list(row.related_plot_threads),
            chapter_index=row.chapter_index,
            score_vector=round(float(row.score), 4),
            final_score=round(float(row.score), 4),
        )
        target = _target_bucket(pack, row.evidence_type)
        for index, existing in enumerate(target):
            if existing.evidence_id == evidence.evidence_id or (existing.evidence_type, existing.text) == (evidence.evidence_type, evidence.text):
                existing.score_vector = max(existing.score_vector, evidence.score_vector)
                existing.final_score = max(existing.final_score, _fused_score(existing, evidence.score_vector))
                existing.source = evidence.source
                target[index] = existing
                break
        else:
            target.append(evidence)
        target.sort(key=lambda item: item.final_score, reverse=True)


def _target_bucket(pack: EvidencePack, evidence_type: str) -> list[NarrativeEvidence]:
    if evidence_type in {"style", "style_snippet", "dialogue", "action", "environment"}:
        return pack.style_evidence
    if evidence_type in {"character", "character_profile"}:
        return pack.character_evidence
    if evidence_type in {"world", "world_rule", "world_fact"}:
        return pack.world_evidence
    if evidence_type in {"author_plan", "author_constraint"}:
        return pack.author_plan_evidence
    if evidence_type in {"plot", "plot_thread", "event", "episodic_event", "memory", "source_memory", "compressed_memory"}:
        return pack.plot_evidence
    return pack.scene_case_evidence


def _fused_score(existing: NarrativeEvidence, vector_score: float) -> float:
    if existing.score_author_plan > 0:
        return min(1.0, max(existing.score_author_plan, existing.final_score, vector_score))
    score = 0.65 * max(vector_score, 0.0) + 0.25 * max(existing.score_structural, existing.final_score, 0.0) + 0.10 * max(existing.score_graph, 0.0)
    return round(min(max(score, existing.final_score, vector_score), 1.0), 4)
