"""Composite narrative evaluation policy."""

from __future__ import annotations

from typing import Sequence

from agent_rl.domains.narrative import DraftCandidate, EvaluationIssue, EvaluationReport, NarrativeTaskState, StateChangeProposal
from agent_rl.narrative_writing.utils import is_negative_constraint, new_id


class CompositeNarrativeEvaluatorPolicy:
    """Evaluates author alignment, character boundaries, style, retrieval, and changes."""

    def evaluate(
        self,
        state: NarrativeTaskState,
        draft: DraftCandidate,
        changes: Sequence[StateChangeProposal],
    ) -> list[EvaluationReport]:
        return [
            self._author_alignment(state, draft),
            self._character_consistency(state, draft),
            self._style_check(state, draft),
            self._retrieval_coverage(state),
            self._change_quality(changes),
        ]

    def _author_alignment(self, state: NarrativeTaskState, draft: DraftCandidate) -> EvaluationReport:
        issues: list[EvaluationIssue] = []
        for constraint in state.author_constraints:
            if constraint.violation_policy == "block_commit" and _mentions_forbidden(draft.content, constraint.text):
                issues.append(
                    EvaluationIssue(
                        issue_id=new_id("issue"),
                        issue_type="author_constraint",
                        severity="blocker",
                        summary=f"草稿可能违反作者约束：{constraint.text}",
                        expected_constraint=constraint.text,
                        suggested_repair="重写相关段落，显式避开该发展。",
                    )
                )
        return _report("author_alignment", issues, passed_score=1.0)

    def _character_consistency(self, state: NarrativeTaskState, draft: DraftCandidate) -> EvaluationReport:
        issues: list[EvaluationIssue] = []
        for character in state.characters:
            for forbidden in character.dialogue_do_not + character.forbidden_actions:
                if forbidden and forbidden in draft.content:
                    issues.append(
                        EvaluationIssue(
                            issue_id=new_id("issue"),
                            issue_type="character_boundary",
                            severity="blocker",
                            summary=f"{character.name} 触碰禁用行为或台词：{forbidden}",
                            expected_constraint=forbidden,
                            suggested_repair="替换为符合角色卡的动作或台词。",
                        )
                    )
        return _report("character_consistency", issues, passed_score=0.9)

    def _style_check(self, state: NarrativeTaskState, draft: DraftCandidate) -> EvaluationReport:
        issues: list[EvaluationIssue] = []
        if state.style_profile is not None:
            for pattern in state.style_profile.forbidden_patterns:
                if pattern and pattern in draft.content:
                    issues.append(
                        EvaluationIssue(
                            issue_id=new_id("issue"),
                            issue_type="style_forbidden_pattern",
                            severity="warning",
                            summary=f"命中禁用风格模式：{pattern}",
                            expected_constraint=pattern,
                            suggested_repair="替换为风格画像允许的表达。",
                        )
                    )
        return _report("style_drift", issues, passed_score=0.9, warning_score=0.75)

    def _retrieval_coverage(self, state: NarrativeTaskState) -> EvaluationReport:
        pack = state.evidence_pack
        issues: list[EvaluationIssue] = []
        if pack is None or not pack.all_evidence():
            issues.append(
                EvaluationIssue(
                    issue_id=new_id("issue"),
                    issue_type="retrieval_coverage",
                    severity="warning",
                    summary="本轮没有可用证据，草稿可能脱离 canon 或风格参考。",
                    suggested_repair="补充参考小说、作者约束或记忆后重新检索。",
                )
            )
        return _report("retrieval_coverage", issues, passed_score=1.0, warning_score=0.5)

    def _change_quality(self, changes: Sequence[StateChangeProposal]) -> EvaluationReport:
        issues: list[EvaluationIssue] = []
        if not changes:
            issues.append(
                EvaluationIssue(
                    issue_id=new_id("issue"),
                    issue_type="no_state_change",
                    severity="blocker",
                    summary="草稿没有抽取出任何可提交状态变化。",
                    suggested_repair="要求生成器输出明确事件、剧情推进或角色变化。",
                )
            )
        return _report("state_change_quality", issues, passed_score=0.9)


def _mentions_forbidden(content: str, constraint: str) -> bool:
    if not is_negative_constraint(constraint):
        return False
    if constraint in content:
        return False
    forbidden = constraint.replace("不要", "").replace("禁止", "").replace("不能", "").replace("避免", "").strip()
    return bool(forbidden and forbidden in content)


def _report(
    report_type: str,
    issues: list[EvaluationIssue],
    passed_score: float,
    warning_score: float = 0.0,
) -> EvaluationReport:
    if any(issue.severity == "blocker" for issue in issues):
        status = "failed"
        score = 0.0
    elif issues:
        status = "warning"
        score = warning_score
    else:
        status = "passed"
        score = passed_score
    return EvaluationReport(
        report_id=new_id("report"),
        report_type=report_type,
        status=status,
        overall_score=score,
        issues=issues,
    )
