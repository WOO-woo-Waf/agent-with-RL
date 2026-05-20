"""Template-method runtime for executing policy/environment control loops."""

from __future__ import annotations

from agent_rl.concepts import (
    AgentState,
    Environment,
    Goal,
    Guardrail,
    MemoryStore,
    Policy,
    Trajectory,
    TrajectoryStep,
    utc_now,
)
from agent_rl.memory import InMemoryStore


class AgentRuntime:
    """Runs the classic observe-decide-act-evaluate loop."""

    def __init__(
        self,
        policy: Policy,
        memory: MemoryStore | None = None,
        guardrails: tuple[Guardrail, ...] = (),
    ) -> None:
        self.policy = policy
        self.memory = memory or InMemoryStore()
        self.guardrails = guardrails

    def run(self, goal: Goal, env: Environment, max_steps: int = 20) -> Trajectory:
        observation, reset_info = env.reset()
        trajectory = Trajectory(goal=goal, metadata={"reset_info": dict(reset_info)})

        for step_index in range(max_steps):
            state = AgentState(
                goal=goal,
                observation=observation,
                memory=self.memory,
                step_index=step_index,
            )
            decision = self.policy.select_action(state, env.available_actions())
            if any(not guardrail.allowed(state, decision) for guardrail in self.guardrails):
                trajectory.outcome = "blocked"
                break
            if decision.action.kind == "control" and decision.action.name == "stop":
                trajectory.outcome = "stopped"
                break

            step = TrajectoryStep(
                index=step_index,
                observation=observation,
                action=decision.action,
                agent_id=decision.action.agent_id,
                rationale=decision.rationale,
            )
            transition = env.step(decision.action)
            step.reward = transition.reward
            step.next_observation = transition.next_observation
            step.ended_at = utc_now()
            step.metadata.update(dict(transition.info))
            trajectory.append(step)
            self.memory.append("trajectory", step)

            observation = transition.next_observation
            if transition.terminated:
                trajectory.outcome = "terminated"
                break
            if transition.truncated:
                trajectory.outcome = "truncated"
                break
        else:
            trajectory.outcome = "max_steps"

        env.close()
        return trajectory
