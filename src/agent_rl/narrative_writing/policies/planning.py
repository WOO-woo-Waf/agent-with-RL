"""Chapter blueprint and chapter plan policy."""

from __future__ import annotations

from agent_rl.domains.narrative import ChapterBlueprint, ChapterPlan, EvidencePack, NarrativeTaskState
from agent_rl.narrative_writing.requests import AuthorRequest
from agent_rl.narrative_writing.utils import is_negative_constraint, new_id, split_author_items, unique


class RuleBasedPlanningPolicy:
    """Creates chapter-level plans from author direction and constraints."""

    def propose_blueprint(self, state: NarrativeTaskState, request: AuthorRequest) -> ChapterBlueprint:
        direction_items = split_author_items(request.writing_direction)
        required = [item for item in direction_items if not is_negative_constraint(item)]
        forbidden = unique(
            item
            for item in (*direction_items, *request.constraints)
            if is_negative_constraint(item)
        )
        related_threads = [thread.thread_id for thread in state.plot_threads[:3]]
        return ChapterBlueprint(
            blueprint_id=new_id("blueprint"),
            chapter_index=request.target_chapter_index,
            chapter_goal=request.writing_direction.strip(),
            required_plot_threads=related_threads,
            required_beats=required or [request.writing_direction.strip()],
            forbidden_beats=list(forbidden),
            expected_scene_count=2,
            pacing_target=_infer_pacing(request),
            ending_hook="保留一个未完全回答的问题，方便下一章继续推进。",
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
