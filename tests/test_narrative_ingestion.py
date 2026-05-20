from agent_rl.narrative_writing import build_author_request_from_files, load_reference_directory, load_reference_file
from agent_rl.narrative_writing.bootstrap import build_initial_state


def test_load_reference_file_reads_utf8_text(tmp_path) -> None:
    reference_path = tmp_path / "chapter-01.txt"
    reference_path.write_text("林舟看见密信。\n雨声压低了仓库里的呼吸。", encoding="utf-8")

    reference = load_reference_file(reference_path)

    assert reference.title == "chapter-01"
    assert "密信" in reference.text
    assert reference.source_type == "target_continuation"


def test_load_reference_directory_keeps_filename_order(tmp_path) -> None:
    (tmp_path / "02.txt").write_text("第二段", encoding="utf-8")
    (tmp_path / "01.txt").write_text("第一段", encoding="utf-8")

    references = load_reference_directory(tmp_path)

    assert [reference.title for reference in references] == ["01", "02"]


def test_file_author_request_bootstraps_retrievable_source_memory(tmp_path) -> None:
    reference_path = tmp_path / "novel.txt"
    reference_path.write_text("林舟握着密信。\n沈姓角色没有立刻解释。", encoding="utf-8")
    request = build_author_request_from_files(
        request="续写下一章",
        reference_paths=(reference_path,),
        writing_direction="继续围绕密信推进",
        confirm_plan=True,
    )

    state = build_initial_state(request)

    assert state.source_documents
    assert len(state.memory_atoms) == 2
    assert all(memory.memory_type == "source_excerpt" for memory in state.memory_atoms)
    assert any("密信" in memory.text for memory in state.memory_atoms)
