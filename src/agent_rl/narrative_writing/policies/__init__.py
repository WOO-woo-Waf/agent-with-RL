"""Default local policy implementations for the narrative-writing Agent."""

from agent_rl.narrative_writing.policies.evaluation import CompositeNarrativeEvaluatorPolicy
from agent_rl.narrative_writing.policies.extraction import RuleBasedExtractorPolicy
from agent_rl.narrative_writing.policies.interaction import BasicAuthorInteractionPolicy
from agent_rl.narrative_writing.policies.memory import SimpleNarrativeMemoryPolicy
from agent_rl.narrative_writing.policies.planning import RuleBasedPlanningPolicy
from agent_rl.narrative_writing.policies.retrieval import KeywordNarrativeRetrievalPolicy
from agent_rl.narrative_writing.policies.writing import TemplateNarrativeWriterPolicy

__all__ = [
    "BasicAuthorInteractionPolicy",
    "CompositeNarrativeEvaluatorPolicy",
    "KeywordNarrativeRetrievalPolicy",
    "RuleBasedExtractorPolicy",
    "RuleBasedPlanningPolicy",
    "SimpleNarrativeMemoryPolicy",
    "TemplateNarrativeWriterPolicy",
]
