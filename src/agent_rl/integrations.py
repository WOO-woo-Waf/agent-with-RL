"""Optional adapters to authoritative open-source/official implementations."""

from __future__ import annotations

from typing import Any, Sequence

from agent_rl.concepts import Action, Environment, Observation, Reward, Tool, Transition


class OptionalDependencyMissing(ImportError):
    """Raised when an optional framework adapter is used without its package."""


class GymnasiumEnvAdapter:
    """Adapter for Gymnasium-style environments.

    The wrapped env is expected to expose reset(), step(), action_space, and close().
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self._last_observation: Observation | None = None

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        raw_observation, info = self.env.reset(seed=seed)
        observation = Observation(raw_observation, source="gymnasium", metadata=dict(info))
        self._last_observation = observation
        return observation, dict(info)

    def step(self, action: Action) -> Transition:
        if self._last_observation is None:
            raise RuntimeError("reset() must be called before step()")
        raw_next, raw_reward, terminated, truncated, info = self.env.step(action.payload)
        next_observation = Observation(raw_next, source="gymnasium", metadata=dict(info))
        transition = Transition(
            observation=self._last_observation,
            action=action,
            next_observation=next_observation,
            reward=Reward(float(raw_reward)),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )
        self._last_observation = next_observation
        return transition

    def available_actions(self) -> Sequence[Action]:
        action_space = getattr(self.env, "action_space", None)
        if action_space is None or not hasattr(action_space, "sample"):
            return ()
        return (Action(name="sample", payload=action_space.sample()),)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


class PettingZooAECAdapter:
    """Thin adapter for PettingZoo AEC environments.

    It exposes the current active agent as metadata and expects actions whose
    payload is already valid for env.step(payload).
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self._agent_iter: Any | None = None
        self._current_agent: str | None = None
        self._last_observation: Observation | None = None

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        self.env.reset(seed=seed)
        self._agent_iter = iter(self.env.agent_iter())
        self._current_agent = next(self._agent_iter)
        raw_observation, reward, termination, truncation, info = self.env.last()
        observation = Observation(
            raw_observation,
            source="pettingzoo",
            metadata={
                "agent": self._current_agent,
                "last_reward": reward,
                "termination": termination,
                "truncation": truncation,
                **dict(info),
            },
        )
        self._last_observation = observation
        return observation, dict(observation.metadata)

    def step(self, action: Action) -> Transition:
        if self._agent_iter is None or self._last_observation is None:
            raise RuntimeError("reset() must be called before step()")
        acting_agent = self._current_agent
        self.env.step(action.payload)
        try:
            self._current_agent = next(self._agent_iter)
        except StopIteration:
            next_observation = Observation(None, source="pettingzoo", metadata={"agent": None})
            return Transition(
                self._last_observation,
                action,
                next_observation,
                Reward(0.0),
                terminated=True,
                info={"acting_agent": acting_agent},
            )

        raw_observation, reward, termination, truncation, info = self.env.last()
        next_observation = Observation(
            raw_observation,
            source="pettingzoo",
            metadata={"agent": self._current_agent, **dict(info)},
        )
        transition = Transition(
            self._last_observation,
            action,
            next_observation,
            Reward(float(reward)),
            terminated=bool(termination),
            truncated=bool(truncation),
            info={"acting_agent": acting_agent, **dict(info)},
        )
        self._last_observation = next_observation
        return transition

    def available_actions(self) -> Sequence[Action]:
        if self._current_agent is None:
            return ()
        space = self.env.action_space(self._current_agent)
        return (Action(name="sample", payload=space.sample(), agent_id=self._current_agent),)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def openai_function_tool(tool: Tool) -> Any:
    """Expose a local Tool through OpenAI Agents SDK when installed."""

    try:
        from agents import function_tool
    except ImportError as exc:
        raise OptionalDependencyMissing("Install the 'openai-agents' package first") from exc

    @function_tool(name_override=tool.name, description_override=tool.description)
    def _wrapped(payload: dict[str, Any]) -> Any:
        return tool.invoke(payload).payload

    return _wrapped


def assert_environment_contract(env: Environment) -> None:
    """Runtime contract check for local or adapted environments."""

    for method_name in ("reset", "step", "available_actions", "close"):
        if not hasattr(env, method_name):
            raise TypeError(f"Environment is missing {method_name}()")
