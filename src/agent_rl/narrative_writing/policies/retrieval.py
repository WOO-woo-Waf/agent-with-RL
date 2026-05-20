"""Local task-aware retrieval policy."""

from __future__ import annotations

from agent_rl.domains.narrative import EvidencePack, NarrativeEvidence, NarrativeQuery, NarrativeTaskState
from agent_rl.narrative_writing.utils import new_id, tokenize


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


def _score_evidence(
    evidence_type: str,
    source: str,
    text: str,
    query_terms: set[str],
    related_entities: list[str] | None = None,
    related_plot_threads: list[str] | None = None,
) -> NarrativeEvidence:
    terms = tokenize(text)
    overlap = len(terms & query_terms)
    score = overlap / max(1, len(query_terms))
    return NarrativeEvidence(
        evidence_id=new_id("ev"),
        evidence_type=evidence_type,
        source=source,
        text=text,
        usage_hint="context",
        related_entities=related_entities or [],
        related_plot_threads=related_plot_threads or [],
        score_structural=score,
        final_score=score,
    )


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
