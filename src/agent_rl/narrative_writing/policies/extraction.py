"""Rule-based state-change extraction policy."""

from __future__ import annotations

import json
from typing import Any

from agent_rl.domains.narrative import DraftCandidate, NarrativeTaskState, StateChangeProposal
from agent_rl.llm import ChatModelClient, JsonBlobParser
from agent_rl.narrative_writing.prompting import PromptComposer
from agent_rl.narrative_writing.utils import new_id


class RuleBasedExtractorPolicy:
    """Extracts coarse events and plot progress proposals from generated text."""

    def extract(self, state: NarrativeTaskState, draft: DraftCandidate) -> list[StateChangeProposal]:
        summary = draft.content.splitlines()[0][:120] if draft.content else "空草稿"
        changes = [
            StateChangeProposal(
                change_id=new_id("change"),
                update_type="narrative_event",
                summary=summary,
                canonical_key=f"event:{draft.draft_id}",
                details={"draft_id": draft.draft_id, "content_preview": draft.content[:240]},
                confidence=0.7,
            )
        ]
        if draft.planned_beat_ids:
            changes.append(
                StateChangeProposal(
                    change_id=new_id("change"),
                    update_type="plot_progress",
                    summary="推进章节计划节拍：" + "；".join(draft.planned_beat_ids[:4]),
                    canonical_key=f"plot-progress:{draft.draft_id}",
                    details={"planned_beats": list(draft.planned_beat_ids)},
                    confidence=0.65,
                )
            )
        return changes


class LLMNarrativeExtractorPolicy:
    """LLM-backed state-change extractor with rule-based fallback."""

    def __init__(
        self,
        client: ChatModelClient,
        *,
        prompt_composer: PromptComposer | None = None,
        fallback: RuleBasedExtractorPolicy | None = None,
        parser: JsonBlobParser | None = None,
    ) -> None:
        self.client = client
        self.prompt_composer = prompt_composer or PromptComposer()
        self.fallback = fallback or RuleBasedExtractorPolicy()
        self.parser = parser or JsonBlobParser()

    def extract(self, state: NarrativeTaskState, draft: DraftCandidate) -> list[StateChangeProposal]:
        prompt = self.prompt_composer.compose_system_prompt(purpose="state_extraction")
        messages = [
            {"role": "system", "content": prompt.system_content},
            {"role": "user", "content": json.dumps(_extraction_payload(state, draft), ensure_ascii=False)},
        ]
        try:
            raw = self.client.complete(messages, purpose="state_extraction", json_mode=True)
            parsed = self.parser.parse(raw)
            changes = _changes_from_payload(parsed.data, draft)
            for change in changes:
                change.details.setdefault("extractor_policy", self.__class__.__name__)
                change.details.setdefault("prompt_metadata", dict(prompt.metadata))
                change.details.setdefault("llm_json_repaired", parsed.repaired)
            return changes
        except Exception as exc:  # noqa: BLE001 - fallback must catch model/parse/client errors.
            changes = self.fallback.extract(state, draft)
            for change in changes:
                change.details["extractor_policy"] = self.__class__.__name__
                change.details["llm_fallback_used"] = True
                change.details["llm_error"] = str(exc)
                change.details["fallback_extractor_policy"] = self.fallback.__class__.__name__
                change.details["prompt_metadata"] = dict(prompt.metadata)
            return changes


def _extraction_payload(state: NarrativeTaskState, draft: DraftCandidate) -> dict[str, Any]:
    return {
        "task": {
            "task_id": state.task_id,
            "story_id": state.story_id,
            "state_version_no": state.state_version_no,
        },
        "draft": {
            "draft_id": draft.draft_id,
            "content": draft.content,
            "planned_beat_ids": list(draft.planned_beat_ids),
            "continuity_notes": list(draft.continuity_notes),
        },
        "known_state": {
            "characters": [character.name for character in state.characters],
            "plot_threads": [thread.name for thread in state.plot_threads],
            "world_rules": [rule.rule_text for rule in state.world_rules],
        },
        "output_schema": {
            "changes": [
                {
                    "update_type": "narrative_event|plot_progress|character_state|world_fact|relationship|style_note",
                    "summary": "string",
                    "canonical_key": "stable dedupe key",
                    "details": "object",
                    "confidence": 0.0,
                    "related_entities": ["string"],
                }
            ]
        },
    }


def _changes_from_payload(payload: Any, draft: DraftCandidate) -> list[StateChangeProposal]:
    raw_changes = payload.get("changes") if isinstance(payload, dict) else payload
    if not isinstance(raw_changes, list):
        raise ValueError("extraction payload must contain a changes list")
    changes: list[StateChangeProposal] = []
    for item in raw_changes:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        update_type = str(item.get("update_type") or "").strip()
        if not summary or not update_type:
            continue
        details = item.get("details")
        changes.append(
            StateChangeProposal(
                change_id=str(item.get("change_id") or new_id("change")),
                update_type=update_type,
                summary=summary,
                canonical_key=str(item.get("canonical_key") or f"{update_type}:{draft.draft_id}:{len(changes)}"),
                details=details if isinstance(details, dict) else {"raw_details": details},
                confidence=_float_in_range(item.get("confidence"), default=0.5),
                related_entities=_list_of_strings(item.get("related_entities")),
            )
        )
    if not changes:
        raise ValueError("extraction payload did not contain usable changes")
    return changes


def _float_in_range(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
