"""Reusable policy strategies."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4
from typing import Sequence

from agent_rl.concepts import Action, AgentState, Decision


class SequencePolicy:
    """Deterministic policy that emits a predefined plan one action at a time."""

    def __init__(self, actions: Sequence[Action], fallback: Action | None = None) -> None:
        self._actions = list(actions)
        self._fallback = fallback or Action(name="stop", kind="control")
        self._cursor_key = f"sequence_policy:{uuid4()}:cursor"

    def select_action(self, state: AgentState, actions: Sequence[Action]) -> Decision:
        cursor = int(state.memory.get(self._cursor_key, 0))
        if cursor < len(self._actions):
            action = self._actions[cursor]
            state.memory.set(self._cursor_key, cursor + 1)
            return Decision(action=action, rationale="next action from sequence")
        return Decision(action=self._fallback, rationale="sequence exhausted")


class GreedyActionPolicy:
    """Simple strategy wrapper around a scoring function."""

    def __init__(
        self,
        score: Callable[[AgentState, Action], float],
        fallback: Action | None = None,
    ) -> None:
        self._score = score
        self._fallback = fallback or Action(name="stop", kind="control")

    def select_action(self, state: AgentState, actions: Sequence[Action]) -> Decision:
        if not actions:
            return Decision(action=self._fallback, rationale="no available actions")
        best = max(actions, key=lambda action: self._score(state, action))
        return Decision(action=best, rationale="highest scoring action")
