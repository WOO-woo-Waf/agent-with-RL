from agent_rl.domains.narrative import (
    EvaluationIssue,
    EvaluationReport,
    EvidencePack,
    NarrativeEvidence,
    NarrativeTaskState,
)


def test_evidence_pack_flattens_partitioned_evidence() -> None:
    pack = EvidencePack(
        pack_id="pack-1",
        query_id="query-1",
        character_evidence=[
            NarrativeEvidence(
                evidence_id="ev-character",
                evidence_type="character_profile",
                source="state",
                text="角色在压力下说话克制。",
            )
        ],
        plot_evidence=[
            NarrativeEvidence(
                evidence_id="ev-plot",
                evidence_type="plot_thread",
                source="state",
                text="密信伏笔尚未回收。",
            )
        ],
    )

    assert [item.evidence_id for item in pack.all_evidence()] == ["ev-character", "ev-plot"]


def test_narrative_task_state_detects_blocking_reports() -> None:
    state = NarrativeTaskState(
        task_id="task-1",
        story_id="story-1",
        goal="续写下一章",
        reports=[
            EvaluationReport(
                report_id="report-1",
                report_type="character_consistency",
                status="failed",
                issues=[
                    EvaluationIssue(
                        issue_id="issue-1",
                        issue_type="knowledge_boundary",
                        severity="blocker",
                        summary="角色知道了尚未揭示的秘密。",
                    )
                ],
            )
        ],
    )

    assert len(state.blocking_reports()) == 1
