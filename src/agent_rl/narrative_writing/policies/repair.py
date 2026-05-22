"""Draft repair and branch selection policies."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from agent_rl.domains.narrative import (
    DraftBranch,
    DraftCandidate,
    DraftRepairPlan,
    DraftRevisionCandidate,
    EvaluationReport,
    NarrativeTaskState,
)
from agent_rl.narrative_writing.utils import new_id


class RuleBasedNarrativeRepairPolicy:
    """Creates a conservative local draft revision from blocker reports."""

    def repair(
        self,
        state: NarrativeTaskState,
        draft: DraftCandidate,
        reports: Sequence[EvaluationReport],
        *,
        attempt_no: int,
    ) -> DraftRevisionCandidate:
        blockers = [
            issue
            for report in reports
            for issue in report.issues
            if issue.severity == "blocker"
        ]
        instructions = [
            issue.suggested_repair or issue.summary
            for issue in blockers
            if (issue.suggested_repair or issue.summary)
        ]
        plan = DraftRepairPlan(
            repair_id=new_id("repair"),
            source_draft_id=draft.draft_id,
            blocker_summaries=[issue.summary for issue in blockers],
            repair_instructions=instructions,
            attempt_no=attempt_no,
            metadata={"policy": self.__class__.__name__, "story_id": state.story_id},
        )
        repaired_content = draft.content.rstrip()
        if instructions:
            repaired_content += "\n\n[repair]\n" + "\n".join(f"- {item}" for item in instructions)
        else:
            repaired_content += "\n\n[repair]\n- Strengthen state change extraction and canon alignment."
        repaired = replace(
            draft,
            draft_id=f"{draft.draft_id}-repair-{attempt_no}",
            content=repaired_content,
            metadata={
                **dict(draft.metadata),
                "repaired": True,
                "repair_id": plan.repair_id,
                "repair_attempt_no": attempt_no,
            },
        )
        return DraftRevisionCandidate(
            revision_id=new_id("draft-revision"),
            source_draft_id=draft.draft_id,
            draft=repaired,
            repair_plan=plan,
            metadata={"policy": self.__class__.__name__},
        )


class ScoreBasedBranchSelectionPolicy:
    """Selects the highest scored candidate branch."""

    def select_branch(self, branches: Sequence[DraftBranch]) -> DraftBranch | None:
        if not branches:
            return None
        return max(branches, key=lambda branch: branch.evaluation.score if branch.evaluation else 0.0)


__all__ = ["RuleBasedNarrativeRepairPolicy", "ScoreBasedBranchSelectionPolicy"]
