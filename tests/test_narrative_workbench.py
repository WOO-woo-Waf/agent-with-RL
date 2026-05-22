from io import StringIO

from agent_rl.narrative_writing.workbench import ConsoleIO, NarrativeInteractiveWorkbench, NarrativeWorkbenchConfig


def test_message_driven_workbench_starts_without_parameter_confirmation(tmp_path) -> None:
    reference = tmp_path / "chapter-001.txt"
    reference.write_text("Lin keeps the sealed letter hidden while rain falls on the warehouse.", encoding="utf-8")
    input_stream = StringIO(
        "\n".join(
            [
                f"Analyze {reference} and continue chapter 2, 600 words, do not reveal the sender. story id: workbench-story session id: workbench-session",
                "quit",
            ]
        )
        + "\n"
    )
    output_stream = StringIO()
    workbench = NarrativeInteractiveWorkbench(
        NarrativeWorkbenchConfig(
            state_root=str(tmp_path / "state"),
            conversation_root=str(tmp_path / "conversation"),
            memory_db=str(tmp_path / "memory.sqlite3"),
            evaluation_root=str(tmp_path / "evaluation"),
            artifact_root=str(tmp_path / "artifacts"),
            operator_root=str(tmp_path / "operator"),
            operator_session_id="operator-test",
            use_llm=False,
        ),
        io=ConsoleIO(input_stream, output_stream),
    )

    exit_code = workbench.run()

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "decide: start_session" in output
    assert "action: start narrative session" in output
    assert "blueprint_proposed" in output
    assert "next: say 'confirm'" in output
    assert (tmp_path / "state" / "workbench-story" / "sessions" / "workbench-session.json").exists()
    operator_state = (tmp_path / "operator" / "operator-test.json").read_text(encoding="utf-8")
    assert "workbench-session" in operator_state
    assert "tool_calls" in operator_state


def test_message_driven_workbench_confirms_exports_and_saves(tmp_path) -> None:
    reference = tmp_path / "chapter-001.txt"
    reference.write_text("Lin keeps the sealed letter hidden while rain falls on the warehouse.", encoding="utf-8")
    export_path = tmp_path / "out" / "chapter-002.txt"
    input_stream = StringIO(
        "\n".join(
            [
                f"Analyze {reference} and continue chapter 2, 600 words. story id: export-story session id: export-session",
                "confirm",
                f"export {export_path}",
                "quit",
            ]
        )
        + "\n"
    )
    output_stream = StringIO()
    workbench = NarrativeInteractiveWorkbench(
        NarrativeWorkbenchConfig(
            state_root=str(tmp_path / "state"),
            conversation_root=str(tmp_path / "conversation"),
            memory_db=str(tmp_path / "memory.sqlite3"),
            evaluation_root=str(tmp_path / "evaluation"),
            artifact_root=str(tmp_path / "artifacts"),
            operator_root=str(tmp_path / "operator"),
            operator_session_id="operator-export",
            use_llm=False,
        ),
        io=ConsoleIO(input_stream, output_stream),
    )

    exit_code = workbench.run()

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "decide: confirm_plan" in output
    assert "committed=True" in output
    assert export_path.exists()


def test_operator_session_resumes_active_narrative_session(tmp_path) -> None:
    reference = tmp_path / "chapter-001.txt"
    reference.write_text("Lin keeps the sealed letter hidden while rain falls on the warehouse.", encoding="utf-8")
    first_input = StringIO(
        "\n".join(
            [
                f"Analyze {reference} and continue chapter 2, 600 words. story id: resume-story session id: resume-session",
                "quit",
            ]
        )
        + "\n"
    )
    first_output = StringIO()
    config = NarrativeWorkbenchConfig(
        state_root=str(tmp_path / "state"),
        conversation_root=str(tmp_path / "conversation"),
        memory_db=str(tmp_path / "memory.sqlite3"),
        evaluation_root=str(tmp_path / "evaluation"),
        artifact_root=str(tmp_path / "artifacts"),
        operator_root=str(tmp_path / "operator"),
        operator_session_id="operator-resume",
        use_llm=False,
    )
    assert NarrativeInteractiveWorkbench(config, io=ConsoleIO(first_input, first_output)).run() == 0

    second_input = StringIO("status\nquit\n")
    second_output = StringIO()
    assert NarrativeInteractiveWorkbench(config, io=ConsoleIO(second_input, second_output)).run() == 0

    output = second_output.getvalue()
    assert "resumed operator session operator-resume" in output
    assert "active narrative session: resume-session" in output
    assert "decide: show_status" in output
