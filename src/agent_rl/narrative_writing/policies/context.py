"""Working-context assembly for narrative generation."""

from __future__ import annotations

from agent_rl.domains.narrative import (
    ChapterPlan,
    EvidencePack,
    NarrativeTaskState,
    PromptContextSection,
    WorkingMemoryContext,
)
from agent_rl.narrative_writing.longform_context import LongformContextSelector
from agent_rl.narrative_writing.requests import AuthorRequest
from agent_rl.narrative_writing.utils import new_id


class BudgetedNarrativeContextPolicy:
    """Builds ordered, budgeted context sections from state and evidence."""

    def __init__(self, char_budget: int = 12000) -> None:
        self.char_budget = char_budget
        self.longform_selector = LongformContextSelector()

    def build(
        self,
        state: NarrativeTaskState,
        evidence_pack: EvidencePack,
        plan: ChapterPlan,
        request: AuthorRequest,
    ) -> WorkingMemoryContext:
        sections = [
            _section(
                "author-request",
                "作者请求",
                "author",
                "\n".join(
                    item
                    for item in (
                        request.request,
                        f"写作方向：{request.writing_direction}" if request.writing_direction else "",
                        _join("硬约束", request.constraints),
                    )
                    if item
                ),
                priority=100,
                order=10,
            ),
            _section(
                "chapter-plan",
                "章节计划",
                "chapter_plan",
                "\n".join(
                    item
                    for item in (
                        f"目标：{plan.objective}",
                        _join("必写节拍", plan.required_beats),
                        _join("连续性要求", plan.continuity_must_keep),
                    )
                    if item
                ),
                priority=95,
                order=20,
            ),
            _section(
                "author-evidence",
                "作者计划证据",
                "evidence",
                _evidence_text(evidence_pack.author_plan_evidence),
                priority=92,
                order=30,
            ),
            _section(
                "plot-evidence",
                "剧情与记忆证据",
                "evidence",
                _evidence_text(evidence_pack.plot_evidence),
                priority=82,
                order=40,
            ),
            _section(
                "character-evidence",
                "人物证据",
                "evidence",
                _evidence_text(evidence_pack.character_evidence),
                priority=78,
                order=50,
            ),
            _section(
                "world-evidence",
                "世界规则",
                "evidence",
                _evidence_text(evidence_pack.world_evidence),
                priority=75,
                order=60,
            ),
            _section(
                "style-evidence",
                "风格参照",
                "evidence",
                _evidence_text(evidence_pack.style_evidence),
                priority=70,
                order=70,
            ),
            _section(
                "state-summary",
                "当前状态摘要",
                "state",
                _state_summary(state),
                priority=65,
                order=80,
                budget_chars=1800,
            ),
        ]
        sections = [section for section in sections if section.text.strip()]
        context = WorkingMemoryContext(
            context_id=new_id("context"),
            evidence_pack_id=evidence_pack.pack_id,
            sections=sections,
            char_budget=self.char_budget,
            metadata={
                "context_policy": self.__class__.__name__,
                "section_count": len(sections),
            },
        )
        blueprint = state.chapter_blueprints[-1] if state.chapter_blueprints else None
        context = self.longform_selector.attach_to_context(context, state, request, blueprint)
        context.metadata["estimated_tokens"] = context.estimated_tokens
        return context


def _section(
    section_id: str,
    label: str,
    source_type: str,
    text: str,
    *,
    priority: int,
    order: int,
    budget_chars: int = 2200,
) -> PromptContextSection:
    return PromptContextSection(
        section_id=section_id,
        label=label,
        source_type=source_type,
        text=text,
        priority=priority,
        order=order,
        budget_chars=budget_chars,
    )


def _join(label: str, items) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return ""
    return f"{label}：" + "；".join(values)


def _evidence_text(items) -> str:
    return "\n".join(f"- [{item.evidence_type}] {item.text}" for item in items if item.text.strip())


def _state_summary(state: NarrativeTaskState) -> str:
    parts = [
        f"task_id={state.task_id}; story_id={state.story_id}; state_version={state.state_version_no}",
        _join("人物", [character.name for character in state.characters]),
        _join("剧情线", [thread.name for thread in state.plot_threads]),
        _join("近期压缩记忆", [memory.summary for memory in state.compressed_memory[:3]]),
    ]
    return "\n".join(part for part in parts if part)
