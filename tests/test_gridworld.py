from agent_rl import Goal
from agent_rl.examples.gridworld import GreedyGridPolicy, GridWorldEnv
from agent_rl.runtime import AgentRuntime


def test_greedy_grid_policy_reaches_target() -> None:
    trajectory = AgentRuntime(GreedyGridPolicy()).run(
        Goal("Reach target"),
        GridWorldEnv(width=3, height=3, target=(2, 2)),
        max_steps=10,
    )

    assert trajectory.outcome == "terminated"
    assert len(trajectory.steps) == 4
    assert trajectory.total_reward > 0


def test_gridworld_truncates_when_max_steps_is_too_low() -> None:
    trajectory = AgentRuntime(GreedyGridPolicy()).run(
        Goal("Reach target"),
        GridWorldEnv(width=3, height=3, target=(2, 2), max_steps=2),
        max_steps=10,
    )

    assert trajectory.outcome == "truncated"
    assert len(trajectory.steps) == 2
