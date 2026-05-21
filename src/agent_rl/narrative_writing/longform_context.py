"""Long-form narrative memory layers and context selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent_rl.domains.narrative import (
    ChapterBlueprint,
    MemoryAtom,
    NarrativeTaskState,
    PromptContextSection,
    StateChangeProposal,
    WorkingMemoryContext,
)
from agent_rl.narrative_writing.requests import AuthorRequest
from agent_rl.narrative_writing.utils import new_id, tokenize

MemoryLayer = Literal["near", "mid", "global"]


@dataclass(frozen=True)
class MemoryCandidate:
    """One selectable state item for long-form context assembly."""

    candidate_id: str
    layer: MemoryLayer
    source_type: str
    text: str
    importance: float = 0.0
    freshness: float = 0.0
    relevance: float = 0.0
    canonical: bool = True
    continuity_risk: str = "low"
    author_priority: str = "normal"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def score(self) -> float:
        layer_bias = {"near": 0.35, "mid": 0.2, "global": 0.1}[self.layer]
        risk_bias = {"high": 0.25, "medium": 0.12, "low": 0.0}.get(self.continuity_risk, 0.0)
        author_bias = {"must": 0.35, "high": 0.22, "normal": 0.0, "low": -0.1}.get(self.author_priority, 0.0)
        canon_bias = 0.08 if self.canonical else -0.08
        return self.importance * 0.3 + self.freshness * 0.15 + self.relevance * 0.3 + layer_bias + risk_bias + author_bias + canon_bias


class LongformContextSelector:
    """Selects near, mid, and global story memory under an explicit budget."""

    def __init__(self, *, max_near: int = 8, max_mid: int = 8, max_global: int = 5) -> None:
        self.max_near = max_near
        self.max_mid = max_mid
        self.max_global = max_global

    def collect(
        self,
        state: NarrativeTaskState,
        request: AuthorRequest,
        blueprint: ChapterBlueprint | None = None,
    ) -> list[MemoryCandidate]:
        query_terms = tokenize(" ".join([request.request, request.writing_direction, *(request.constraints)]))
        candidates: list[MemoryCandidate] = []
        candidates.extend(_near_candidates(state, query_terms))
        candidates.extend(_mid_candidates(state, query_terms))
        candidates.extend(_global_candidates(state, query_terms, blueprint))
        return candidates

    def select(
        self,
        state: NarrativeTaskState,
        request: AuthorRequest,
        blueprint: ChapterBlueprint | None = None,
    ) -> dict[MemoryLayer, list[MemoryCandidate]]:
        candidates = self.collect(state, request, blueprint)
        grouped: dict[MemoryLayer, list[MemoryCandidate]] = {"near": [], "mid": [], "global": []}
        for layer, limit in (("near", self.max_near), ("mid", self.max_mid), ("global", self.max_global)):
            layer_items = [item for item in candidates if item.layer == layer and item.text.strip()]
            grouped[layer] = sorted(layer_items, key=lambda item: item.score, reverse=True)[:limit]
        return grouped

    def build_sections(
        self,
        state: NarrativeTaskState,
        request: AuthorRequest,
        blueprint: ChapterBlueprint | None = None,
    ) -> list[PromptContextSection]:
        selected = self.select(state, request, blueprint)
        sections: list[PromptContextSection] = []
        for order, (layer, label, budget) in enumerate(
            (
                ("near", "近场可续写剧情", 2600),
                ("mid", "章节与角色抽象剧情", 2400),
                ("global", "全局压缩剧情", 1800),
            ),
            start=1,
        ):
            text = "\n".join(f"- [{item.source_type}] {item.text}" for item in selected[layer])
            if text.strip():
                sections.append(
                    PromptContextSection(
                        section_id=f"longform-{layer}",
                        label=label,
                        source_type=f"longform_{layer}",
                        text=text,
                        priority=88 - order,
                        order=75 + order,
                        budget_chars=budget,
                        metadata={
                            "layer": layer,
                            "candidate_count": len(selected[layer]),
                            "top_scores": [round(item.score, 4) for item in selected[layer][:5]],
                        },
                    )
                )
        return sections

    def attach_to_context(
        self,
        context: WorkingMemoryContext,
        state: NarrativeTaskState,
        request: AuthorRequest,
        blueprint: ChapterBlueprint | None = None,
    ) -> WorkingMemoryContext:
        existing_ids = {section.section_id for section in context.sections}
        for section in self.build_sections(state, request, blueprint):
            if section.section_id not in existing_ids:
                context.sections.append(section)
        context.metadata["longform_layers"] = {
            "near": self.max_near,
            "mid": self.max_mid,
            "global": self.max_global,
        }
        context.metadata["estimated_tokens"] = context.estimated_tokens
        return context


class DraftCompressionTool:
    """Creates pre-commit memory atoms and compressed blocks from pending changes."""

    def compress(self, state: NarrativeTaskState, changes: list[StateChangeProposal]) -> dict[str, object]:
        high_priority = [change for change in changes if change.confidence >= 0.75]
        continuity_risks = [
            change.summary
            for change in changes
            if change.update_type in {"character_state", "relationship", "world_fact", "plot_progress"}
        ][:6]
        state.metadata["pending_memory_compression"] = {
            "change_count": len(changes),
            "high_priority_count": len(high_priority),
            "continuity_risks": continuity_risks,
            "candidate_block_summary": "；".join(change.summary for change in changes[:4]),
        }
        return {
            "memory_atoms_pending": len(changes),
            "high_priority_items": len(high_priority),
            "continuity_risks": continuity_risks,
        }


def _near_candidates(state: NarrativeTaskState, query_terms: set[str]) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for chunk in state.source_chunks[-12:]:
        candidates.append(
            MemoryCandidate(
                candidate_id=chunk.chunk_id,
                layer="near",
                source_type="source_chunk",
                text=chunk.text[:500],
                importance=0.65,
                freshness=0.8,
                relevance=_relevance(chunk.text, query_terms),
                metadata={"chapter_index": chunk.chapter_index},
            )
        )
    for snippet in state.style_snippets[:6]:
        candidates.append(
            MemoryCandidate(
                candidate_id=snippet.snippet_id,
                layer="near",
                source_type="style_snippet",
                text=snippet.text,
                importance=0.55,
                freshness=0.7,
                relevance=0.4 + _relevance(snippet.text, query_terms) * 0.4,
                metadata={"chapter_index": snippet.chapter_index},
            )
        )
    return candidates


def _mid_candidates(state: NarrativeTaskState, query_terms: set[str]) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for event in state.events[-24:]:
        candidates.append(
            MemoryCandidate(
                candidate_id=event.event_id,
                layer="mid",
                source_type="event",
                text=event.summary,
                importance=0.75,
                freshness=0.65,
                relevance=_relevance(event.summary, query_terms),
                continuity_risk="medium",
                metadata={"chapter_index": event.chapter_index},
            )
        )
    for thread in state.plot_threads:
        text = f"{thread.name}: stage={thread.stage}; open={thread.open_questions}; next={thread.next_expected_beats}; blocked={thread.blocked_beats}"
        candidates.append(
            MemoryCandidate(
                candidate_id=thread.thread_id,
                layer="mid",
                source_type="plot_thread",
                text=text,
                importance=0.82,
                freshness=0.7,
                relevance=_relevance(text, query_terms),
                continuity_risk="high",
            )
        )
    for atom in state.memory_atoms[-24:]:
        candidates.append(_memory_atom_candidate(atom, query_terms))
    return candidates


def _global_candidates(
    state: NarrativeTaskState,
    query_terms: set[str],
    blueprint: ChapterBlueprint | None,
) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for character in state.characters:
        text = f"{character.name}: traits={character.stable_traits}; goals={character.current_goals}; boundaries={character.knowledge_boundary}; voice={character.voice_profile}"
        candidates.append(
            MemoryCandidate(
                candidate_id=character.character_id,
                layer="global",
                source_type="character_card",
                text=text,
                importance=0.8,
                freshness=0.5,
                relevance=_relevance(text, query_terms),
                continuity_risk="high",
            )
        )
    for rule in state.world_rules:
        candidates.append(
            MemoryCandidate(
                candidate_id=rule.rule_id,
                layer="global",
                source_type="world_rule",
                text=rule.rule_text,
                importance=0.78,
                freshness=0.45,
                relevance=_relevance(rule.rule_text, query_terms),
                continuity_risk="high" if rule.rule_type == "hard" else "medium",
            )
        )
    for block in state.compressed_memory[-12:]:
        candidates.append(
            MemoryCandidate(
                candidate_id=block.block_id,
                layer="global",
                source_type=f"compressed:{block.block_type}",
                text=block.summary,
                importance=0.72,
                freshness=0.5,
                relevance=_relevance(block.summary, query_terms),
                metadata={"scope": block.scope, "preserved_ids": list(block.preserved_ids)},
            )
        )
    if state.style_profile is not None:
        candidates.append(
            MemoryCandidate(
                candidate_id=state.style_profile.profile_id,
                layer="global",
                source_type="style_profile",
                text=(
                    f"pov={state.style_profile.narrative_pov}; distance={state.style_profile.narrative_distance}; "
                    f"dialogue_ratio={state.style_profile.dialogue_ratio}; forbidden={state.style_profile.forbidden_patterns}"
                ),
                importance=0.68,
                freshness=0.4,
                relevance=0.45,
            )
        )
    if blueprint is not None:
        candidates.append(
            MemoryCandidate(
                candidate_id=blueprint.blueprint_id,
                layer="global",
                source_type="confirmed_blueprint" if blueprint.confirmed else "pending_blueprint",
                text=f"{blueprint.chapter_goal}; target={blueprint.target_total_chars}; beats={blueprint.required_beats}",
                importance=0.9,
                freshness=1.0,
                relevance=1.0,
                author_priority="must",
            )
        )
    return candidates


def _memory_atom_candidate(atom: MemoryAtom, query_terms: set[str]) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=atom.memory_id,
        layer="mid" if atom.freshness >= 0.55 else "global",
        source_type=f"memory:{atom.memory_type}",
        text=atom.text,
        importance=atom.importance,
        freshness=atom.freshness,
        relevance=_relevance(atom.text, query_terms),
        canonical=atom.canonical,
    )


def _relevance(text: str, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    text_terms = tokenize(text)
    if not text_terms:
        return 0.0
    return len(text_terms & query_terms) / max(1, len(query_terms))
