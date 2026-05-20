"""Agent architecture variants mapped to the same core model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from agent_rl.core.concepts import Action, AgentState, Decision, Environment, Goal, Policy, Trajectory
from agent_rl.core.memory import InMemoryStore
from agent_rl.core.policies import SequencePolicy
from agent_rl.core.runtime import AgentRuntime


class ReActAgent:
    """Single-level online controller: observe, choose one action, observe again."""

    def __init__(self, policy: Policy) -> None:
        self.runtime = AgentRuntime(policy=policy, memory=InMemoryStore())

    def run(self, goal: Goal, env: Environment, max_steps: int = 20) -> Trajectory:
        return self.runtime.run(goal, env, max_steps=max_steps)


class PlanAndExecuteAgent:
    """Hierarchical controller: plan first, then execute the plan as macro policy."""

    def __init__(self, planner: Callable[[Goal, Environment], Sequence[Action]]) -> None:
        self.planner = planner

    def run(self, goal: Goal, env: Environment, max_steps: int = 20) -> Trajectory:
        plan = list(self.planner(goal, env))
        runtime = AgentRuntime(policy=SequencePolicy(plan), memory=InMemoryStore())
        trajectory = runtime.run(goal, env, max_steps=max_steps)
        trajectory.metadata["plan"] = [action.name for action in plan]
        return trajectory


class MultiAgentCoordinator:
    """Composite controller that delegates each decision to a named sub-policy."""

    def __init__(
        self,
        policies: dict[str, Policy],
        router: Callable[[AgentState, Sequence[Action]], str],
    ) -> None:
        if not policies:
            raise ValueError("MultiAgentCoordinator requires at least one policy")
        self.policies = policies
        self.router = router

    def as_policy(self) -> Policy:
        return _CoordinatedPolicy(self.policies, self.router)

    def run(self, goal: Goal, env: Environment, max_steps: int = 20) -> Trajectory:
        runtime = AgentRuntime(policy=self.as_policy(), memory=InMemoryStore())
        return runtime.run(goal, env, max_steps=max_steps)


class _CoordinatedPolicy:
    def __init__(
        self,
        policies: dict[str, Policy],
        router: Callable[[AgentState, Sequence[Action]], str],
    ) -> None:
        self._policies = policies
        self._router = router

    def select_action(self, state: AgentState, actions: Sequence[Action]) -> Decision:
        agent_id = self._router(state, actions)
        policy = self._policies[agent_id]
        decision = policy.select_action(state, actions)
        action = Action(
            name=decision.action.name,
            payload=decision.action.payload,
            kind=decision.action.kind,
            agent_id=agent_id,
            metadata=decision.action.metadata,
        )
        return Decision(
            action=action,
            rationale=f"{agent_id}: {decision.rationale}",
            confidence=decision.confidence,
            metadata=decision.metadata,
        )
