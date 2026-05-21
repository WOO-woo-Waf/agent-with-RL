import os

from agent_rl.config import (
    env_snapshot,
    expand_env_value,
    find_env_file,
    get_env_bool,
    get_env_float,
    get_env_int,
    load_env_file,
    load_project_env,
    parse_env_file,
)


def test_load_project_env_reads_values_without_overriding(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_MODEL=deepseek-v4-flash\nCUSTOM_VALUE=${LLM_MODEL}-x\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "existing")

    loaded = load_project_env(env_path)

    assert loaded == env_path
    assert os.environ["LLM_MODEL"] == "existing"
    assert os.environ["CUSTOM_VALUE"] == "existing-x"


def test_load_project_env_can_override(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_MODEL=deepseek-v4-flash\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "existing")

    load_project_env(env_path, override=True)

    assert os.environ["LLM_MODEL"] == "deepseek-v4-flash"


def test_find_env_file_searches_upward(tmp_path) -> None:
    root_env = tmp_path / ".env"
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    root_env.write_text("LLM_MODEL=deepseek-v4-flash\n", encoding="utf-8")

    assert find_env_file(start=nested) == root_env


def test_parse_env_file_supports_export_and_quotes(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        'export LLM_API_BASE="https://api.deepseek.com"\nLLM_MODEL=\'deepseek-v4-flash\'\n',
        encoding="utf-8",
    )

    values = parse_env_file(env_path)

    assert values["LLM_API_BASE"] == "https://api.deepseek.com"
    assert values["LLM_MODEL"] == "deepseek-v4-flash"


def test_load_env_file_expands_existing_env_values(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("BASE_DIR=${HOME}/agent-rl\n", encoding="utf-8")
    monkeypatch.setenv("HOME", "/home/example")

    load_env_file(env_path)

    assert os.environ["BASE_DIR"] == "/home/example/agent-rl"


def test_typed_env_helpers(monkeypatch) -> None:
    monkeypatch.setenv("A_INT", "12")
    monkeypatch.setenv("A_FLOAT", "1.25")
    monkeypatch.setenv("A_BOOL", "false")

    assert get_env_int("A_INT", 0) == 12
    assert get_env_float("A_FLOAT", 0.0) == 1.25
    assert get_env_bool("A_BOOL", True) is False


def test_env_snapshot_redacts_sensitive_values() -> None:
    snapshot = env_snapshot(
        ["LLM_API_KEY", "LLM_MODEL", "MISSING"],
        environ={"LLM_API_KEY": "secret", "LLM_MODEL": "deepseek-v4-flash"},
    )

    assert snapshot == {
        "LLM_API_KEY": "<set>",
        "LLM_MODEL": "deepseek-v4-flash",
        "MISSING": "<missing>",
    }


def test_expand_env_value_is_cross_platform_string_expansion() -> None:
    assert expand_env_value("${ROOT}\\data/${NAME}", {"ROOT": "D:\\buff", "NAME": "novel"}) == "D:\\buff\\data/novel"
