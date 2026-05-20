"""Evaluation helpers for narrative retrieval quality."""

from __future__ import annotations

from agent_rl.domains.narrative import EvaluationIssue, EvaluationReport, EvidencePack, NarrativeQuery
from agent_rl.narrative_writing.utils import new_id


class BasicRetrievalEvaluationPolicy:
    """Scores whether retrieved evidence covers the current writing query."""

    def evaluate(self, evidence_pack: EvidencePack, query: NarrativeQuery) -> EvaluationReport:
        evidence = evidence_pack.all_evidence()
        actual_types = {item.evidence_type for item in evidence}
        expected_types = set(query.required_evidence_types)
        normalized_actual = set(actual_types)
        if "source_memory" in normalized_actual:
            normalized_actual.add("compressed_memory")
        missing = sorted(expected_types - normalized_actual)
        issues = [
            EvaluationIssue(
                issue_id=new_id("issue"),
                issue_type="retrieval_missing_evidence_type",
                severity="warning",
                summary=f"检索缺少证据类型：{evidence_type}",
                expected_constraint=evidence_type,
                suggested_repair="补充对应索引或放宽检索查询。",
            )
            for evidence_type in missing
        ]
        if not evidence:
            issues.append(
                EvaluationIssue(
                    issue_id=new_id("issue"),
                    issue_type="retrieval_empty",
                    severity="warning",
                    summary="本轮没有检索到任何证据。",
                    suggested_repair="检查参考材料、query 构造或检索策略。",
                )
            )
        coverage = 1.0
        if expected_types:
            coverage = (len(expected_types) - len(missing)) / len(expected_types)
        score = max(0.0, min(1.0, 0.35 + 0.65 * coverage))
        if not evidence:
            score = 0.0
        return EvaluationReport(
            report_id=new_id("report"),
            report_type="retrieval_evaluation",
            status="passed" if not issues else "warning",
            overall_score=score,
            issues=issues,
            metrics={
                "evidence_count": float(len(evidence)),
                "required_type_count": float(len(expected_types)),
                "missing_type_count": float(len(missing)),
                "coverage": coverage,
            },
        )
