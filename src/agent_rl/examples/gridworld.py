"""Tiny RL environment that follows the project Environment protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent_rl.concepts import Action, AgentState, Goal, Observation, Reward, Transition
from agent_rl.runtime import AgentRuntime


Position = tuple[int, int]


@dataclass
class GridWorldEnv:
    width: int = 4
    height: int = 4
    start: Position = (0, 0)
    target: Position = (3, 3)
    max_steps: int = 20

    def __post_init__(self) -> None:
        self.position = self.start
        self.steps = 0
        self._last = self._observe()

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, object]]:
        self.position = self.start
        self.steps = 0
        self._last = self._observe()
        return self._last, {"seed": seed}

    def step(self, action: Action) -> Transition:
        previous = self._last
        x, y = self.position
        if action.name == "up":
            y -= 1
        elif action.name == "down":
            y += 1
        elif action.name == "left":
            x -= 1
        elif action.name == "right":
            x += 1

        self.position = (min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1))
        self.steps += 1
        next_observation = self._observe()
        self._last = next_observation

        reached_target = self.position == self.target
        truncated = self.steps >= self.max_steps and not reached_target
        reward = Reward(1.0 if reached_target else -0.01, {"task": 1.0 if reached_target else 0.0})
        return Transition(
            observation=previous,
            action=action,
            next_observation=next_observation,
            reward=reward,
            terminated=reached_target,
            truncated=truncated,
            info={"position": self.position},
        )

    def available_actions(self) -> Sequence[Action]:
        return (
            Action("up"),
            Action("down"),
            Action("left"),
            Action("right"),
        )

    def close(self) -> None:
        return None

    def _observe(self) -> Observation:
        return Observation(
            {"position": self.position, "target": self.target},
            source="gridworld",
        )


class GreedyGridPolicy:
    """Move along Manhattan distance toward the target."""

    def select_action(self, state: AgentState, actions: Sequence[Action]):
        position = state.observation.payload["position"]
        target = state.observation.payload["target"]
        px, py = position
        tx, ty = target
        if px < tx:
            action = Action("right")
        elif px > tx:
            action = Action("left")
        elif py < ty:
            action = Action("down")
        elif py > ty:
            action = Action("up")
        else:
            action = Action("stop", kind="control")
        from agent_rl.concepts import Decision

        return Decision(action=action, rationale="reduce Manhattan distance")


def run_demo() -> None:
    goal = Goal("Reach the target cell", ("position equals target",))
    trajectory = AgentRuntime(GreedyGridPolicy()).run(goal, GridWorldEnv())
    print(f"outcome={trajectory.outcome} steps={len(trajectory.steps)} reward={trajectory.total_reward:.2f}")


if __name__ == "__main__":
    run_demo()
