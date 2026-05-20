import json

from agent_rl.llm import (
    EndpointPool,
    LLMEndpoint,
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
    has_llm_configuration,
)


class FakeOpenAICompatibleClient(OpenAICompatibleChatClient):
    def __init__(self, config: OpenAICompatibleConfig, payloads):
        super().__init__(config=config, endpoint_pool=EndpointPool())
        self.payloads = list(payloads)
        self.requests = []

    def _post_chat_completion(self, endpoint, messages, *, json_mode):
        self.requests.append((endpoint, messages, json_mode))
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _response(content: str, total_tokens: int = 9):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": total_tokens},
    }


def test_openai_compatible_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "model-x")
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "2")

    config = OpenAICompatibleConfig.from_env()

    assert config.configured is True
    assert has_llm_configuration(config) is True
    assert config.max_attempts == 2


def test_endpoint_pool_rotates_start_point() -> None:
    pool = EndpointPool()
    endpoints = [LLMEndpoint("a", "k"), LLMEndpoint("b", "k")]

    assert [item.api_base for item in pool.iter_from(endpoints)] == ["a", "b"]
    assert [item.api_base for item in pool.iter_from(endpoints)] == ["b", "a"]


def test_openai_compatible_client_records_audit_and_usage(tmp_path, monkeypatch) -> None:
    audit_path = tmp_path / "interactions.jsonl"
    usage_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("LLM_AUDIT_LOG_PATH", str(audit_path))
    monkeypatch.setenv("LLM_USAGE_LOG_PATH", str(usage_path))
    config = OpenAICompatibleConfig(
        api_base="https://example.test/v1",
        api_key="key",
        model_name="model-x",
        max_attempts=1,
        base_backoff_s=0,
    )
    client = FakeOpenAICompatibleClient(config, [_response('{"content":"ok"}')])
    messages = [
        {
            "role": "system",
            "content": "# Prompt Metadata\nprompt_profile: default\ntask_prompt_id: draft_generation",
        },
        {"role": "user", "content": "{}"},
    ]

    content = client.complete(messages, purpose="draft_generation", json_mode=True)

    assert content == '{"content":"ok"}'
    assert client.requests[0][2] is True
    assert "JSON mode contract" in client.requests[0][1][0]["content"]
    audit_records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    usage_records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event_type"] for record in audit_records] == ["llm_request_started", "llm_request_succeeded"]
    assert audit_records[-1]["task_prompt_id"] == "draft_generation"
    assert usage_records[-1]["total_tokens"] == 9


def test_openai_compatible_client_retries_retryable_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_AUDIT_LOG_PATH", str(tmp_path / "interactions.jsonl"))
    monkeypatch.setenv("LLM_USAGE_LOG_PATH", str(tmp_path / "usage.jsonl"))
    config = OpenAICompatibleConfig(
        api_base="https://example.test/v1",
        api_key="key",
        model_name="model-x",
        max_attempts=2,
        base_backoff_s=0,
    )
    client = FakeOpenAICompatibleClient(config, [TimeoutError("timeout"), _response('{"content":"ok"}')])

    content = client.complete([{"role": "user", "content": "{}"}], purpose="draft_generation", json_mode=True)

    assert content == '{"content":"ok"}'
    assert len(client.requests) == 2


def test_openai_compatible_client_requires_config() -> None:
    client = OpenAICompatibleChatClient(OpenAICompatibleConfig())

    try:
        client.complete([{"role": "user", "content": "{}"}], purpose="draft_generation", json_mode=True)
    except RuntimeError as exc:
        assert "LLM configuration is incomplete" in str(exc)
    else:
        raise AssertionError("expected incomplete config failure")
