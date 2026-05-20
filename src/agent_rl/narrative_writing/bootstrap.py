"""Factory functions for bootstrapping a narrative task state."""

from __future__ import annotations

from typing import Sequence

from agent_rl.domains.narrative import (
    AuthorConstraint,
    CharacterCard,
    MemoryAtom,
    NarrativeTaskState,
    PlotThreadState,
    SourceDocument,
    StyleProfile,
    StyleSnippet,
    WorldRule,
)
from agent_rl.narrative_writing.requests import AuthorRequest, ReferenceMaterial
from agent_rl.narrative_writing.utils import candidate_names, is_negative_constraint, new_id, split_author_items, split_paragraphs


def build_initial_state(request: AuthorRequest) -> NarrativeTaskState:
    state = NarrativeTaskState(task_id=request.task_id, story_id=request.story_id, goal=request.request)
    state.source_documents.extend(build_source_documents(request.references))
    state.author_constraints.extend(build_author_constraints(request.constraints))
    state.characters.extend(infer_characters(request.references))
    state.plot_threads.extend(infer_plot_threads(request))
    state.world_rules.extend(infer_world_rules(request.references))
    state.style_profile = infer_style_profile(request.references)
    state.style_snippets.extend(infer_style_snippets(request.references))
    state.memory_atoms.extend(infer_reference_memory(request.references))
    state.metadata["writing_direction"] = request.writing_direction
    return state


def build_source_documents(references: Sequence[ReferenceMaterial]) -> list[SourceDocument]:
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


def build_author_constraints(constraints: Sequence[str]) -> list[AuthorConstraint]:
    return [
        AuthorConstraint(
            constraint_id=new_id("constraint"),
            text=constraint,
            constraint_type="forbidden_development" if is_negative_constraint(constraint) else "chapter_goal",
            priority="high" if any(marker in constraint for marker in ("必须", "不要", "禁止", "不能")) else "normal",
            violation_policy="block_commit" if is_negative_constraint(constraint) else "warn",
        )
        for constraint in constraints
        if constraint.strip()
    ]


def infer_characters(references: Sequence[ReferenceMaterial]) -> list[CharacterCard]:
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
            voice_profile=["保持与参考文本一致的语气"],
            dialogue_do=["围绕当前冲突说话"],
            dialogue_do_not=["突然解释全部真相"],
            forbidden_actions=["无铺垫地改变核心立场"],
        )
        for name in names[:5]
    ]


def infer_plot_threads(request: AuthorRequest) -> list[PlotThreadState]:
    goal = request.writing_direction or request.request
    return [
        PlotThreadState(
            thread_id=new_id("plot"),
            name="主线推进",
            thread_type="main",
            status="open",
            stage="continuation",
            stakes=goal,
            open_questions=["当前线索会把角色带向哪里？"],
            next_expected_beats=split_author_items(goal) or [goal],
        )
    ]


def infer_world_rules(references: Sequence[ReferenceMaterial]) -> list[WorldRule]:
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


def infer_style_profile(references: Sequence[ReferenceMaterial]) -> StyleProfile:
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


def infer_style_snippets(references: Sequence[ReferenceMaterial]) -> list[StyleSnippet]:
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


def infer_reference_memory(references: Sequence[ReferenceMaterial]) -> list[MemoryAtom]:
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
