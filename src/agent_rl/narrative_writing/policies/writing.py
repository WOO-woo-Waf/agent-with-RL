"""Template writing policy."""

from __future__ import annotations

import json
from typing import Any

from agent_rl.domains.narrative import ChapterPlan, DraftCandidate, EvidencePack, NarrativeTaskState, WorkingMemoryContext
from agent_rl.llm import ChatModelClient, JsonBlobParser
from agent_rl.narrative_writing.prompting import PromptComposer


class TemplateNarrativeWriterPolicy:
    """Deterministic writer useful for local runs, tests, and later LLM replacement."""

    def generate(
        self,
        state: NarrativeTaskState,
        plan: ChapterPlan,
        evidence_pack: EvidencePack,
        working_context: WorkingMemoryContext | None = None,
    ) -> DraftCandidate:
        protagonist = state.characters[0].name if state.characters else "主角"
        constraints = [evidence.text for evidence in evidence_pack.author_plan_evidence[:3]]
        plot_evidence = [evidence.text for evidence in evidence_pack.plot_evidence[:3]]
        style_evidence = [evidence.text for evidence in evidence_pack.style_evidence[:2]]

        paragraphs = [
            f"第 {plan.chapter_index} 章草稿：{plan.objective}",
            f"{protagonist}带着上一轮事件留下的压力进入新的场景。{_style_summary(state)}",
        ]
        for index, beat in enumerate(plan.required_beats, start=1):
            paragraphs.append(f"场景 {index}：{beat}。人物行动必须承接既有因果，不越过已确认的知识边界。")
        if constraints:
            paragraphs.append("作者硬约束：" + "；".join(constraints))
        if plot_evidence:
            paragraphs.append("剧情承接：" + "；".join(plot_evidence))
        if style_evidence:
            paragraphs.append("风格参照：" + " / ".join(style_evidence))
        if working_context is not None:
            paragraphs.append(f"上下文装配：{len(working_context.sections)} 个片段，约 {working_context.estimated_tokens} tokens。")
        paragraphs.append("结尾保留一个未解问题，让下一章仍有推进空间。")

        return DraftCandidate(
            draft_id=f"draft-{abs(hash(tuple(paragraphs))) % 10**12}",
            content="\n\n".join(paragraphs),
            planned_beat_ids=list(plan.required_beats),
            style_targets=style_evidence,
            continuity_notes=plan.continuity_must_keep,
            metadata={
                "writer_policy": self.__class__.__name__,
                "working_context_id": working_context.context_id if working_context else "",
                "context_section_count": len(working_context.sections) if working_context else 0,
            },
        )


class LLMNarrativeWriterPolicy:
    """LLM-backed writer with deterministic template fallback."""

    def __init__(
        self,
        client: ChatModelClient,
        *,
        prompt_composer: PromptComposer | None = None,
        fallback: TemplateNarrativeWriterPolicy | None = None,
        parser: JsonBlobParser | None = None,
    ) -> None:
        self.client = client
        self.prompt_composer = prompt_composer or PromptComposer()
        self.fallback = fallback or TemplateNarrativeWriterPolicy()
        self.parser = parser or JsonBlobParser()

    def generate(
        self,
        state: NarrativeTaskState,
        plan: ChapterPlan,
        evidence_pack: EvidencePack,
        working_context: WorkingMemoryContext | None = None,
    ) -> DraftCandidate:
        prompt = self.prompt_composer.compose_system_prompt(purpose="draft_generation")
        messages = [
            {"role": "system", "content": prompt.system_content},
            {"role": "user", "content": json.dumps(_draft_payload(state, plan, evidence_pack, working_context), ensure_ascii=False)},
        ]
        try:
            raw = self.client.complete(messages, purpose="draft_generation", json_mode=True)
            parsed = self.parser.parse(raw)
            draft = _draft_from_payload(parsed.data, plan)
            draft.metadata.update(
                {
                    "writer_policy": self.__class__.__name__,
                    "prompt_metadata": dict(prompt.metadata),
                    "llm_json_repaired": parsed.repaired,
                    "llm_parse_source": parsed.source,
                    "llm_fallback_used": False,
                    "working_context_id": working_context.context_id if working_context else "",
                    "context_section_count": len(working_context.sections) if working_context else 0,
                }
            )
            return draft
        except Exception as exc:  # noqa: BLE001 - fallback must catch model/parse/client errors.
            fallback_draft = self.fallback.generate(state, plan, evidence_pack, working_context)
            fallback_draft.metadata.update(
                {
                    "writer_policy": self.__class__.__name__,
                    "llm_fallback_used": True,
                    "llm_error": str(exc),
                    "fallback_writer_policy": self.fallback.__class__.__name__,
                    "prompt_metadata": dict(prompt.metadata),
                }
            )
            return fallback_draft


def _style_summary(state: NarrativeTaskState) -> str:
    if state.style_profile is None:
        return "风格保持克制，并优先服务剧情推进。"
    avg = state.style_profile.sentence_length_distribution.get("avg", 0.0)
    if avg:
        return f"句子节奏参考原文平均长度约 {avg:.1f} 字。"
    return "风格保持参考材料的叙事距离和节奏。"


def _draft_payload(
    state: NarrativeTaskState,
    plan: ChapterPlan,
    evidence_pack: EvidencePack,
    working_context: WorkingMemoryContext | None,
) -> dict[str, Any]:
    return {
        "task": {
            "task_id": state.task_id,
            "story_id": state.story_id,
            "state_version_no": state.state_version_no,
        },
        "chapter_plan": {
            "plan_id": plan.plan_id,
            "chapter_index": plan.chapter_index,
            "objective": plan.objective,
            "required_beats": list(plan.required_beats),
            "continuity_must_keep": list(plan.continuity_must_keep),
            "target_word_count": plan.target_word_count,
        },
        "working_context": working_context.render_for_model() if working_context else "",
        "evidence_trace": list(evidence_pack.retrieval_trace),
        "output_schema": {
            "content": "string",
            "planned_beat_ids": ["string"],
            "style_targets": ["string"],
            "continuity_notes": ["string"],
            "rationale": "short audit note",
        },
    }


def _draft_from_payload(payload: Any, plan: ChapterPlan) -> DraftCandidate:
    if not isinstance(payload, dict):
        raise ValueError("draft payload must be a JSON object")
    content = str(payload.get("content") or "").strip()
    if not content:
        raise ValueError("draft payload missing content")
    planned_beat_ids = _list_of_strings(payload.get("planned_beat_ids")) or list(plan.required_beats)
    style_targets = _list_of_strings(payload.get("style_targets"))
    continuity_notes = _list_of_strings(payload.get("continuity_notes")) or list(plan.continuity_must_keep)
    return DraftCandidate(
        draft_id=str(payload.get("draft_id") or f"draft-{abs(hash(content)) % 10**12}"),
        content=content,
        planned_beat_ids=planned_beat_ids,
        style_targets=style_targets,
        continuity_notes=continuity_notes,
        metadata={"llm_rationale": str(payload.get("rationale") or "")},
    )


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
