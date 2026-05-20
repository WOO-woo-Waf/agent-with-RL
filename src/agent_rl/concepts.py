"""Core OOAD concepts shared by RL agents and LLM agent runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


JsonMap = Mapping[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Goal:
    """Human-readable task goal plus optional measurable success criteria."""

    description: str
    success_criteria: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """What the agent can currently observe, not necessarily the full state."""

    payload: Any
    source: str = "environment"
    metadata: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """A high-level action: environment move, tool call, message, handoff, or stop."""

    name: str
    payload: Any = None
    kind: str = "environment"
    agent_id: str = "default"
    metadata: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class Reward:
    """Scalar reward with optional dimensions for richer evaluation."""

    value: float
    dimensions: Mapping[str, float] = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """One environment transition after applying an action."""

    observation: Observation
    action: Action
    next_observation: Observation
    reward: Reward
    terminated: bool = False
    truncated: bool = False
    info: JsonMap = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class Decision:
    """Policy output before the runtime applies guardrails and execution."""

    action: Action
    rationale: str = ""
    confidence: float | None = None
    metadata: JsonMap = field(default_factory=dict)


@dataclass
class AgentState:
    """Engineering approximation of a POMDP belief state."""

    goal: Goal
    observation: Observation
    memory: "MemoryStore"
    step_index: int = 0
    scratchpad: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    """Replayable record of one control-loop step."""

    index: int
    observation: Observation
    action: Action
    reward: Reward | None = None
    next_observation: Observation | None = None
    agent_id: str = "default"
    rationale: str = ""
    started_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete task trace for evaluation, replay, and later policy learning."""

    goal: Goal
    steps: list[TrajectoryStep] = field(default_factory=list)
    outcome: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(step.reward.value for step in self.steps if step.reward is not None)

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)


@runtime_checkable
class MemoryStore(Protocol):
    """Memory abstraction. Implementations can be in-memory, vector, SQL, etc."""

    def get(self, key: str, default: Any = None) -> Any:
        ...

    def set(self, key: str, value: Any) -> None:
        ...

    def append(self, key: str, value: Any) -> None:
        ...


@runtime_checkable
class Policy(Protocol):
    """Strategy pattern: choose an action from current agent state."""

    def select_action(self, state: AgentState, actions: Sequence[Action]) -> Decision:
        ...


@runtime_checkable
class Environment(Protocol):
    """Gymnasium-compatible shape expressed in project domain objects."""

    def reset(self, seed: int | None = None) -> tuple[Observation, JsonMap]:
        ...

    def step(self, action: Action) -> Transition:
        ...

    def available_actions(self) -> Sequence[Action]:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class Tool(Protocol):
    """Tool action executor used by LLM agents and option-style macro actions."""

    name: str
    description: str

    def invoke(self, payload: Any) -> Observation:
        ...


@runtime_checkable
class Guardrail(Protocol):
    """Policy constraint that can veto unsafe or invalid decisions."""

    def allowed(self, state: AgentState, decision: Decision) -> bool:
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Converts a trajectory into reward/evaluation signals."""

    def evaluate(self, trajectory: Trajectory) -> Reward:
        ...
