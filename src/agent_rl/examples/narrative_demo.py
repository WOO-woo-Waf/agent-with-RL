"""Run a local narrative-writing Agent without external services."""

from __future__ import annotations

from agent_rl.narrative_writing import AuthorRequest, NarrativeWritingAgent, ReferenceMaterial


def run_demo() -> None:
    agent = NarrativeWritingAgent()
    reference = ReferenceMaterial(
        title="示例参考",
        text=(
            "林舟站在旧仓库门口，雨水沿着铁皮棚落下。他没有立刻开口，"
            "只是看着那封被揉皱的信。对方越是沉默，他越觉得答案藏在沉默之后。"
        ),
        source_type="target_continuation",
    )

    preview = agent.run(
        AuthorRequest(
            request="规划并续写下一章",
            references=(reference,),
            writing_direction="下一章必须找到密信线索；不要让主角立刻原谅对方；节奏压抑一点",
            constraints=("不要让主角立刻原谅对方",),
            confirm_plan=False,
        )
    )
    print(preview.assistant_message)

    result = agent.run(
        AuthorRequest(
            request="规划并续写下一章",
            references=(reference,),
            writing_direction="下一章必须找到密信线索；不要让主角立刻原谅对方；节奏压抑一点",
            constraints=("不要让主角立刻原谅对方",),
            confirm_plan=True,
        )
    )
    print(f"outcome={result.trajectory.outcome} committed={result.committed}")
    if result.draft is not None:
        print(result.draft.content)


if __name__ == "__main__":
    run_demo()
