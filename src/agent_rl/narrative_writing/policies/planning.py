"""Chapter blueprint and chapter plan policy."""

from __future__ import annotations

from agent_rl.domains.narrative import ChapterBlueprint, ChapterBlueprintSegment, ChapterPlan, EvidencePack, NarrativeTaskState
from agent_rl.narrative_writing.requests import AuthorRequest
from agent_rl.narrative_writing.utils import is_negative_constraint, new_id, split_author_items, unique


class RuleBasedPlanningPolicy:
    """Creates chapter-level plans from author direction and constraints."""

    def propose_blueprint(self, state: NarrativeTaskState, request: AuthorRequest) -> ChapterBlueprint:
        planning_text = " ".join(part for part in (request.writing_direction, request.blueprint_feedback) if part.strip())
        direction_items = split_author_items(planning_text)
        required = [item for item in direction_items if not is_negative_constraint(item)]
        forbidden = unique(
            item
            for item in (*direction_items, *request.constraints)
            if is_negative_constraint(item)
        )
        related_threads = [thread.thread_id for thread in state.plot_threads[:3]]
        target_total_chars = max(int(request.target_word_count or 0), 0)
        segments = _build_segments(
            required_beats=required or [planning_text.strip()],
            forbidden_beats=list(forbidden),
            target_total_chars=target_total_chars,
            related_threads=related_threads,
        )
        return ChapterBlueprint(
            blueprint_id=new_id("blueprint"),
            chapter_index=request.target_chapter_index,
            chapter_goal=planning_text.strip(),
            required_plot_threads=related_threads,
            required_beats=required or [planning_text.strip()],
            forbidden_beats=list(forbidden),
            expected_scene_count=max(1, len(segments)),
            pacing_target=_infer_pacing(request),
            ending_hook="保留一个未完全回答的问题，方便下一章继续推进。",
            target_total_chars=target_total_chars,
            segments=segments,
            requires_author_confirmation=True,
            confirmed=request.confirm_plan,
            revision_notes=[request.blueprint_feedback] if request.blueprint_feedback else [],
        )

    def build_chapter_plan(
        self,
        state: NarrativeTaskState,
        blueprint: ChapterBlueprint,
        evidence_pack: EvidencePack,
        request: AuthorRequest,
    ) -> ChapterPlan:
        required_beats = blueprint.required_beats or [blueprint.chapter_goal]
        continuity = [evidence.text for evidence in evidence_pack.plot_evidence[:3] if evidence.text.strip()]
        continuity.extend(evidence.text for evidence in evidence_pack.author_plan_evidence[:3] if evidence.text.strip())
        return ChapterPlan(
            plan_id=new_id("chapter-plan"),
            chapter_index=blueprint.chapter_index,
            objective=blueprint.chapter_goal,
            source_blueprint_id=blueprint.blueprint_id,
            target_word_count=request.target_word_count,
            required_beats=required_beats,
            scene_plan_ids=[new_id("scene-plan") for _ in range(max(1, blueprint.expected_scene_count or 1))],
            continuity_must_keep=continuity[:6],
            completion_criteria={
                "must_hit_required_beats": True,
                "must_avoid_forbidden_beats": blueprint.forbidden_beats,
                "ending_hook": blueprint.ending_hook,
            },
        )


def _infer_pacing(request: AuthorRequest) -> str:
    text = request.writing_direction + " " + " ".join(request.constraints)
    if "压抑" in text or "克制" in text:
        return "slow_tense"
    if "快" in text or "战斗" in text:
        return "fast_action"
    return "balanced"


def _build_segments(
    *,
    required_beats: list[str],
    forbidden_beats: list[str],
    target_total_chars: int,
    related_threads: list[str],
) -> list[ChapterBlueprintSegment]:
    beats = [beat for beat in required_beats if beat.strip()] or ["推进本章核心目标"]
    if target_total_chars >= 12000 and len(beats) < 4:
        segment_count = 4
    else:
        segment_count = max(1, len(beats))
    base_targets = _split_target_chars(target_total_chars, segment_count)
    segments: list[ChapterBlueprintSegment] = []
    for index in range(segment_count):
        beat = beats[index] if index < len(beats) else beats[-1]
        segments.append(
            ChapterBlueprintSegment(
                segment_id=new_id("blueprint-segment"),
                title=f"段落 {index + 1}",
                goal=beat,
                target_chars=base_targets[index],
                required_beats=[beat],
                forbidden_beats=list(forbidden_beats),
                plot_thread_ids=list(related_threads),
                entry_state="承接上一段的角色状态和信息边界" if index else "承接既有 canon 和作者方向",
                exit_state="为下一段保留推进空间",
            )
        )
    return segments


def _split_target_chars(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if total <= 0:
        return [0 for _ in range(count)]
    base = total // count
    remainder = total % count
    return [base + (1 if index < remainder else 0) for index in range(count)]
