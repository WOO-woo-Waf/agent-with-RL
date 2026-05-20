from agent_rl import Action, Goal, MultiAgentCoordinator, PlanAndExecuteAgent, SequencePolicy
from agent_rl.examples.gridworld import GridWorldEnv


def test_plan_and_execute_records_plan() -> None:
    agent = PlanAndExecuteAgent(
        lambda goal, env: [Action("right"), Action("right"), Action("down"), Action("down")]
    )

    trajectory = agent.run(Goal("Reach target"), GridWorldEnv(width=3, height=3, target=(2, 2)))

    assert trajectory.outcome == "terminated"
    assert trajectory.metadata["plan"] == ["right", "right", "down", "down"]


def test_multi_agent_coordinator_marks_agent_id() -> None:
    coordinator = MultiAgentCoordinator(
        policies={
            "horizontal": SequencePolicy([Action("right"), Action("right")]),
            "vertical": SequencePolicy([Action("down"), Action("down")]),
        },
        router=lambda state, actions: "horizontal" if state.step_index < 2 else "vertical",
    )

    trajectory = coordinator.run(
        Goal("Reach target"),
        GridWorldEnv(width=3, height=3, target=(2, 2)),
        max_steps=4,
    )

    assert trajectory.outcome == "terminated"
    assert [step.agent_id for step in trajectory.steps] == [
        "horizontal",
        "horizontal",
        "vertical",
        "vertical",
    ]
