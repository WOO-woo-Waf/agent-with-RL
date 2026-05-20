"""Rule-based state-change extraction policy."""

from __future__ import annotations

from agent_rl.domains.narrative import DraftCandidate, NarrativeTaskState, StateChangeProposal
from agent_rl.narrative_writing.utils import new_id


class RuleBasedExtractorPolicy:
    """Extracts coarse events and plot progress proposals from generated text."""

    def extract(self, state: NarrativeTaskState, draft: DraftCandidate) -> list[StateChangeProposal]:
        summary = draft.content.splitlines()[0][:120] if draft.content else "空草稿"
        changes = [
            StateChangeProposal(
                change_id=new_id("change"),
                update_type="narrative_event",
                summary=summary,
                canonical_key=f"event:{draft.draft_id}",
                details={"draft_id": draft.draft_id, "content_preview": draft.content[:240]},
                confidence=0.7,
            )
        ]
        if draft.planned_beat_ids:
            changes.append(
                StateChangeProposal(
                    change_id=new_id("change"),
                    update_type="plot_progress",
                    summary="推进章节计划节拍：" + "；".join(draft.planned_beat_ids[:4]),
                    canonical_key=f"plot-progress:{draft.draft_id}",
                    details={"planned_beats": list(draft.planned_beat_ids)},
                    confidence=0.65,
                )
            )
        return changes
