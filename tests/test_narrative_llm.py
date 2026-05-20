import json

from agent_rl.narrative_writing import AuthorRequest, ReferenceMaterial
from agent_rl.llm import JsonBlobParser
from agent_rl.narrative_writing.policies import LLMNarrativeExtractorPolicy, LLMNarrativeWriterPolicy
from agent_rl.narrative_writing.scenario import NarrativeScenarioAdapter


class FakeClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, bool]] = []

    def complete(self, messages, *, purpose: str, json_mode: bool = True) -> str:
        self.calls.append((purpose, json_mode))
        response = self.responses[purpose]
        if isinstance(response, Exception):
            raise response
        return response


def _prepared_state():
    request = AuthorRequest(
        request="续写下一章",
        references=(ReferenceMaterial(title="参考", text="林舟握着密信，没有立刻解释。"),),
        writing_direction="继续推进密信线索",
        confirm_plan=True,
    )
    scenario = NarrativeScenarioAdapter()
    state = scenario.build_initial_state(request)
    blueprint = scenario.propose_plan(state, request)
    query = scenario.build_query(state, request)
    pack = scenario.retrieve_context(state, query)
    plan = scenario.build_chapter_plan(state, blueprint, pack, request)
    context = scenario.build_working_context(state, plan, pack, request)
    return state, plan, pack, context


def test_json_blob_parser_extracts_fenced_json() -> None:
    parsed = JsonBlobParser().parse('prefix\n```json\n{"content": "草稿"}\n```')

    assert parsed.data == {"content": "草稿"}


def test_llm_writer_policy_uses_model_payload() -> None:
    state, plan, pack, context = _prepared_state()
    client = FakeClient(
        {
            "draft_generation": json.dumps(
                {
                    "content": "林舟把密信收进掌心，仍然没有原谅对方。",
                    "planned_beat_ids": ["找到密信"],
                    "style_targets": ["克制"],
                    "continuity_notes": ["不提前解释真相"],
                    "rationale": "承接证据。",
                },
                ensure_ascii=False,
            )
        }
    )

    draft = LLMNarrativeWriterPolicy(client).generate(state, plan, pack, context)

    assert "密信" in draft.content
    assert draft.metadata["writer_policy"] == "LLMNarrativeWriterPolicy"
    assert draft.metadata["llm_fallback_used"] is False
    assert client.calls == [("draft_generation", True)]


def test_llm_writer_policy_falls_back_on_bad_json() -> None:
    state, plan, pack, context = _prepared_state()
    client = FakeClient({"draft_generation": "not json"})

    draft = LLMNarrativeWriterPolicy(client).generate(state, plan, pack, context)

    assert draft.content
    assert draft.metadata["llm_fallback_used"] is True
    assert draft.metadata["fallback_writer_policy"] == "TemplateNarrativeWriterPolicy"


def test_llm_extractor_policy_uses_model_changes() -> None:
    state, plan, pack, context = _prepared_state()
    draft = LLMNarrativeWriterPolicy(
        FakeClient({"draft_generation": '{"content":"林舟发现密信。","planned_beat_ids":["发现密信"]}'})
    ).generate(state, plan, pack, context)
    client = FakeClient(
        {
            "state_extraction": json.dumps(
                {
                    "changes": [
                        {
                            "update_type": "narrative_event",
                            "summary": "林舟发现密信。",
                            "canonical_key": "event:secret-letter",
                            "confidence": 0.91,
                            "related_entities": ["林舟"],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        }
    )

    changes = LLMNarrativeExtractorPolicy(client).extract(state, draft)

    assert changes[0].summary == "林舟发现密信。"
    assert changes[0].confidence == 0.91
    assert changes[0].details["extractor_policy"] == "LLMNarrativeExtractorPolicy"


def test_llm_extractor_policy_falls_back_on_client_error() -> None:
    state, plan, pack, context = _prepared_state()
    draft = LLMNarrativeWriterPolicy(
        FakeClient({"draft_generation": '{"content":"林舟发现密信。","planned_beat_ids":["发现密信"]}'})
    ).generate(state, plan, pack, context)
    client = FakeClient({"state_extraction": RuntimeError("model down")})

    changes = LLMNarrativeExtractorPolicy(client).extract(state, draft)

    assert changes
    assert changes[0].details["llm_fallback_used"] is True
