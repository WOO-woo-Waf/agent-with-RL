"""Application service that runs the narrative-writing Agent use case."""

from __future__ import annotations

from agent_rl.core.memory import InMemoryStore
from agent_rl.core.runtime import AgentRuntime
from agent_rl.domains.narrative import NarrativeTaskState
from agent_rl.narrative_writing.policies import BasicAuthorInteractionPolicy
from agent_rl.narrative_writing.ports import AuthorInteractionPolicy
from agent_rl.narrative_writing.react import (
    NarrativeAuthorLedPolicy,
    NarrativeReActEnvironment,
    narrative_goal_from_request,
)
from agent_rl.narrative_writing.requests import AuthorRequest, NarrativeRunResult
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter


class NarrativeWritingAgent:
    """Usable narrative-writing Agent with author interaction gates."""

    def __init__(
        self,
        scenario: NarrativeScenarioAdapter | None = None,
        interaction_policy: AuthorInteractionPolicy | None = None,
    ) -> None:
        self.scenario = scenario or NarrativeScenarioAdapter()
        self.interaction_policy = interaction_policy or BasicAuthorInteractionPolicy()

    def run(self, request: AuthorRequest, state: NarrativeTaskState | None = None) -> NarrativeRunResult:
        env = NarrativeReActEnvironment(
            request,
            scenario=self.scenario,
            task_state=state,
            interaction_policy=self.interaction_policy,
        )
        runtime = AgentRuntime(policy=NarrativeAuthorLedPolicy(), memory=InMemoryStore())
        trajectory = runtime.run(narrative_goal_from_request(request), env, max_steps=12)
        return NarrativeRunResult(
            state=env.task_state,
            trajectory=trajectory,
            questions=env.workflow.questions,
            assistant_message=env.workflow.last_message,
            requires_confirmation=trajectory.outcome == "needs_confirmation",
            proposed_blueprint=env.workflow.proposed_blueprint,
            draft=env.workflow.draft,
            committed=env.workflow.committed,
        )
