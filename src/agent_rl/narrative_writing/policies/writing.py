"""Template writing policy."""

from __future__ import annotations

from agent_rl.domains.narrative import ChapterPlan, DraftCandidate, EvidencePack, NarrativeTaskState


class TemplateNarrativeWriterPolicy:
    """Deterministic writer useful for local runs, tests, and later LLM replacement."""

    def generate(self, state: NarrativeTaskState, plan: ChapterPlan, evidence_pack: EvidencePack) -> DraftCandidate:
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
        paragraphs.append("结尾保留一个未解问题，让下一章仍有推进空间。")

        return DraftCandidate(
            draft_id=f"draft-{abs(hash(tuple(paragraphs))) % 10**12}",
            content="\n\n".join(paragraphs),
            planned_beat_ids=list(plan.required_beats),
            style_targets=style_evidence,
            continuity_notes=plan.continuity_must_keep,
            metadata={"writer_policy": self.__class__.__name__},
        )


def _style_summary(state: NarrativeTaskState) -> str:
    if state.style_profile is None:
        return "风格保持克制，并优先服务剧情推进。"
    avg = state.style_profile.sentence_length_distribution.get("avg", 0.0)
    if avg:
        return f"句子节奏参考原文平均长度约 {avg:.1f} 字。"
    return "风格保持参考材料的叙事距离和节奏。"
