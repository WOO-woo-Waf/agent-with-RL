"""Domain model for narrative/novel-writing agent scenarios.

The objects here are scenario concepts. They can be used by an Agent runtime
through adapters, policies, retrieval services, and evaluators without making
the core Agent/RL package depend on any specific writing application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


Status = Literal["candidate", "confirmed", "contested", "deprecated"]
Severity = Literal["info", "warning", "blocker"]


@dataclass(frozen=True)
class SourceSpan:
    """A traceable span in the original novel, generated text, or author note."""

    span_id: str
    source_id: str
    source_type: str
    chapter_index: int | None = None
    start_offset: int = 0
    end_offset: int = 0
    text_preview: str = ""


@dataclass
class SourceDocument:
    """Input material such as target novel, style reference, or author notes."""

    document_id: str
    title: str
    source_type: str = "target_continuation"
    author: str = ""
    language: str = "zh"
    text_hash: str = ""
    source_span_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldRule:
    """Canonical or candidate world constraint used by generation and validation."""

    rule_id: str
    rule_text: str
    rule_scope: str = "global"
    rule_type: str = "hard"
    stability: Status = "candidate"
    applies_to: list[str] = field(default_factory=list)
    forbidden_implications: list[str] = field(default_factory=list)
    required_implications: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)


@dataclass
class LocationState:
    """A place with continuity, atmosphere, access rules, and event history."""

    location_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    location_type: str = ""
    atmosphere_tags: list[str] = field(default_factory=list)
    access_rules: list[str] = field(default_factory=list)
    known_event_ids: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)


@dataclass
class ObjectState:
    """Important item that can carry ownership, foreshadowing, or plot state."""

    object_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    owner_character_id: str = ""
    current_location_id: str = ""
    functions: list[str] = field(default_factory=list)
    plot_relevance: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)


@dataclass
class CharacterCard:
    """Stable character identity, voice, behavior constraints, and boundaries."""

    character_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    role_type: str = ""
    identity_tags: list[str] = field(default_factory=list)
    stable_traits: list[str] = field(default_factory=list)
    flaws: list[str] = field(default_factory=list)
    wounds_or_fears: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    moral_boundaries: list[str] = field(default_factory=list)
    current_goals: list[str] = field(default_factory=list)
    hidden_goals: list[str] = field(default_factory=list)
    knowledge_boundary: list[str] = field(default_factory=list)
    voice_profile: list[str] = field(default_factory=list)
    dialogue_do: list[str] = field(default_factory=list)
    dialogue_do_not: list[str] = field(default_factory=list)
    gesture_patterns: list[str] = field(default_factory=list)
    decision_patterns: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)


@dataclass
class CharacterDynamicState:
    """Character state at a narrative time point."""

    character_id: str
    chapter_index: int | None = None
    emotional_state: str = ""
    physical_state: str = ""
    current_location_id: str = ""
    active_goal: str = ""
    known_facts: list[str] = field(default_factory=list)
    believed_facts: list[str] = field(default_factory=list)
    secrets_held: list[str] = field(default_factory=list)
    recent_decisions: list[str] = field(default_factory=list)
    arc_stage: str = ""


@dataclass
class RelationshipState:
    """Directional relationship state between two characters."""

    relationship_id: str
    source_character_id: str
    target_character_id: str
    relationship_type: str = ""
    public_status: str = ""
    private_status: str = ""
    trust_level: float = 0.0
    tension_level: float = 0.0
    emotional_tags: list[str] = field(default_factory=list)
    unresolved_conflicts: list[str] = field(default_factory=list)
    next_expected_shift: str = ""


@dataclass
class NarrativeEvent:
    """Canonical or candidate event with causal and state-change links."""

    event_id: str
    summary: str
    event_type: str = ""
    chapter_index: int | None = None
    scene_id: str = ""
    timeline_order: int | None = None
    location_id: str = ""
    participants: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    revealed_facts: list[str] = field(default_factory=list)
    changed_state_refs: list[str] = field(default_factory=list)
    plot_thread_ids: list[str] = field(default_factory=list)
    is_canonical: bool = True
    source_span_ids: list[str] = field(default_factory=list)


@dataclass
class PlotThreadState:
    """Main/sub/relationship/mystery plot line with progress constraints."""

    thread_id: str
    name: str
    thread_type: str = "main"
    status: str = "open"
    stage: str = ""
    stakes: str = ""
    open_questions: list[str] = field(default_factory=list)
    anchor_event_ids: list[str] = field(default_factory=list)
    next_expected_beats: list[str] = field(default_factory=list)
    blocked_beats: list[str] = field(default_factory=list)
    resolution_conditions: list[str] = field(default_factory=list)
    related_character_ids: list[str] = field(default_factory=list)


@dataclass
class Beat:
    """Small narrative unit that serves a scene or plot function."""

    beat_id: str
    summary: str
    narrative_function: str
    beat_type: str = ""
    required: bool = False
    status: str = "planned"
    involved_character_ids: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class ForeshadowingState:
    """Planted clue, reinforcement, reveal policy, and payoff state."""

    foreshadowing_id: str
    seed_text: str
    status: str = "candidate"
    planted_at_chapter: int | None = None
    expected_payoff_chapter: int | None = None
    related_object_ids: list[str] = field(default_factory=list)
    related_character_ids: list[str] = field(default_factory=list)
    related_plot_thread_ids: list[str] = field(default_factory=list)
    reveal_policy: str = ""


@dataclass
class SceneState:
    """Scene state used as the main unit for generation and validation."""

    scene_id: str
    chapter_index: int
    scene_index: int
    objective: str
    scene_type: str = ""
    location_id: str = ""
    pov_character_id: str = ""
    time_label: str = ""
    entry_state: str = ""
    exit_state: str = ""
    conflict_id: str = ""
    involved_character_ids: list[str] = field(default_factory=list)
    beat_ids: list[str] = field(default_factory=list)
    emotional_curve: list[str] = field(default_factory=list)
    style_requirements: list[str] = field(default_factory=list)


@dataclass
class StyleProfile:
    """Structured style baseline, not a free-form prompt sentence."""

    profile_id: str
    narrative_pov: str = ""
    tense: str = ""
    narrative_distance: str = ""
    sentence_length_distribution: Mapping[str, float] = field(default_factory=dict)
    paragraph_length_distribution: Mapping[str, float] = field(default_factory=dict)
    dialogue_ratio: float = 0.0
    description_mix: Mapping[str, float] = field(default_factory=dict)
    rhetoric_markers: list[str] = field(default_factory=list)
    lexical_fingerprint: list[str] = field(default_factory=list)
    pacing_profile: dict[str, Any] = field(default_factory=dict)
    forbidden_patterns: list[str] = field(default_factory=list)


@dataclass
class StyleSnippet:
    """Traceable style exemplar for RAG, few-shot prompting, and style eval."""

    snippet_id: str
    text: str
    snippet_type: str = ""
    normalized_template: str = ""
    style_tags: list[str] = field(default_factory=list)
    speaker_or_pov: str = ""
    scene_type: str = ""
    chapter_index: int | None = None
    source_span_id: str = ""


@dataclass
class AuthorConstraint:
    """Confirmed author intent with priority and violation behavior."""

    constraint_id: str
    text: str
    constraint_type: str
    priority: str = "normal"
    status: Status = "confirmed"
    applies_to_chapters: list[int] = field(default_factory=list)
    applies_to_characters: list[str] = field(default_factory=list)
    applies_to_threads: list[str] = field(default_factory=list)
    violation_policy: str = "block_commit"


@dataclass
class ChapterBlueprint:
    """Author/system chapter-level plan that generation must satisfy."""

    blueprint_id: str
    chapter_index: int
    chapter_goal: str
    required_plot_threads: list[str] = field(default_factory=list)
    required_character_arcs: list[str] = field(default_factory=list)
    required_beats: list[str] = field(default_factory=list)
    forbidden_beats: list[str] = field(default_factory=list)
    expected_scene_count: int | None = None
    pacing_target: str = ""
    ending_hook: str = ""


@dataclass
class MemoryAtom:
    """Minimal memory record that can be retrieved, promoted, or compressed."""

    memory_id: str
    memory_type: str
    text: str
    canonical: bool = True
    importance: float = 0.0
    freshness: float = 0.0
    related_entities: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)
    state_version_no: int | None = None


@dataclass
class CompressedMemoryBlock:
    """Concept-aware compression result with preserved/dropped provenance."""

    block_id: str
    block_type: str
    scope: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    preserved_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    compression_ratio: float = 0.0
    valid_until_state_version: int | None = None


@dataclass
class NarrativeQuery:
    """Task-derived query for narrative retrieval, not just raw user text."""

    query_id: str
    query_text: str
    query_type: str
    target_chapter_index: int | None = None
    scene_type: str = ""
    pov_character_id: str = ""
    involved_character_ids: list[str] = field(default_factory=list)
    plot_thread_ids: list[str] = field(default_factory=list)
    required_evidence_types: list[str] = field(default_factory=list)
    token_budget: int = 0


@dataclass
class NarrativeEvidence:
    """Unified evidence item across source text, state, graph, author plan, memory."""

    evidence_id: str
    evidence_type: str
    source: str
    text: str
    usage_hint: str = ""
    related_entities: list[str] = field(default_factory=list)
    related_plot_threads: list[str] = field(default_factory=list)
    chapter_index: int | None = None
    score_vector: float = 0.0
    score_graph: float = 0.0
    score_structural: float = 0.0
    score_author_plan: float = 0.0
    final_score: float = 0.0


@dataclass
class EvidencePack:
    """Partitioned evidence context for prompt building and audit."""

    pack_id: str
    query_id: str
    style_evidence: list[NarrativeEvidence] = field(default_factory=list)
    character_evidence: list[NarrativeEvidence] = field(default_factory=list)
    plot_evidence: list[NarrativeEvidence] = field(default_factory=list)
    world_evidence: list[NarrativeEvidence] = field(default_factory=list)
    author_plan_evidence: list[NarrativeEvidence] = field(default_factory=list)
    scene_case_evidence: list[NarrativeEvidence] = field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)

    def all_evidence(self) -> list[NarrativeEvidence]:
        return [
            *self.style_evidence,
            *self.character_evidence,
            *self.plot_evidence,
            *self.world_evidence,
            *self.author_plan_evidence,
            *self.scene_case_evidence,
        ]


@dataclass
class ChapterPlan:
    """Runtime plan for one continuation execution."""

    plan_id: str
    chapter_index: int
    objective: str
    source_blueprint_id: str = ""
    target_word_count: int | None = None
    required_beats: list[str] = field(default_factory=list)
    scene_plan_ids: list[str] = field(default_factory=list)
    continuity_must_keep: list[str] = field(default_factory=list)
    completion_criteria: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftCandidate:
    """Generated text before extraction, validation, and commit."""

    draft_id: str
    content: str
    planned_beat_ids: list[str] = field(default_factory=list)
    style_targets: list[str] = field(default_factory=list)
    continuity_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateChangeProposal:
    """Candidate state update extracted from generated or author-provided text."""

    change_id: str
    update_type: str
    summary: str
    canonical_key: str
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source_span_ids: list[str] = field(default_factory=list)
    conflict_mark: bool = False
    conflict_reason: str = ""
    related_entities: list[str] = field(default_factory=list)


@dataclass
class EvaluationIssue:
    """One validation finding that can block, warn, or inform repair."""

    issue_id: str
    issue_type: str
    severity: Severity
    summary: str
    expected_constraint: str = ""
    evidence: str = ""
    suggested_repair: str = ""


@dataclass
class EvaluationReport:
    """Unified quality report across character, plot, style, world, retrieval."""

    report_id: str
    report_type: str
    status: str
    overall_score: float = 0.0
    issues: list[EvaluationIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def blocks_commit(self) -> bool:
        return any(issue.severity == "blocker" for issue in self.issues)


@dataclass
class NarrativeTaskState:
    """Scenario state root for a novel-writing Agent task."""

    task_id: str
    story_id: str
    goal: str
    source_documents: list[SourceDocument] = field(default_factory=list)
    world_rules: list[WorldRule] = field(default_factory=list)
    locations: list[LocationState] = field(default_factory=list)
    objects: list[ObjectState] = field(default_factory=list)
    characters: list[CharacterCard] = field(default_factory=list)
    character_states: list[CharacterDynamicState] = field(default_factory=list)
    relationships: list[RelationshipState] = field(default_factory=list)
    events: list[NarrativeEvent] = field(default_factory=list)
    plot_threads: list[PlotThreadState] = field(default_factory=list)
    beats: list[Beat] = field(default_factory=list)
    foreshadowing: list[ForeshadowingState] = field(default_factory=list)
    scenes: list[SceneState] = field(default_factory=list)
    style_profile: StyleProfile | None = None
    style_snippets: list[StyleSnippet] = field(default_factory=list)
    author_constraints: list[AuthorConstraint] = field(default_factory=list)
    chapter_blueprints: list[ChapterBlueprint] = field(default_factory=list)
    memory_atoms: list[MemoryAtom] = field(default_factory=list)
    compressed_memory: list[CompressedMemoryBlock] = field(default_factory=list)
    evidence_pack: EvidencePack | None = None
    chapter_plan: ChapterPlan | None = None
    draft: DraftCandidate | None = None
    pending_changes: list[StateChangeProposal] = field(default_factory=list)
    reports: list[EvaluationReport] = field(default_factory=list)
    state_version_no: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def blocking_reports(self) -> list[EvaluationReport]:
        return [report for report in self.reports if report.blocks_commit]
