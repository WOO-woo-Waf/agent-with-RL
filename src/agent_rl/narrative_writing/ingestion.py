"""Local ingestion helpers for author-provided narrative references."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from agent_rl.narrative_writing.requests import AuthorRequest, ReferenceMaterial


DEFAULT_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def read_text_file(path: str | Path, encodings: Iterable[str] = DEFAULT_TEXT_ENCODINGS) -> str:
    """Read a text file with common encodings used by Chinese novel drafts."""

    file_path = Path(path)
    encoding_candidates = tuple(encodings)
    last_error: UnicodeDecodeError | None = None
    for encoding in encoding_candidates:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            f"Unable to decode {file_path} with {encoding_candidates}",
        )
    return file_path.read_text()


def load_reference_file(
    path: str | Path,
    *,
    source_type: str = "target_continuation",
    title: str | None = None,
    author: str = "",
) -> ReferenceMaterial:
    """Convert one local novel/reference text file into a reference DTO."""

    file_path = Path(path)
    return ReferenceMaterial(
        title=title or file_path.stem,
        text=read_text_file(file_path),
        source_type=source_type,
        author=author,
    )


def load_reference_directory(
    path: str | Path,
    *,
    source_type: str = "target_continuation",
    pattern: str = "*.txt",
    author: str = "",
) -> tuple[ReferenceMaterial, ...]:
    """Load a directory of text references in filename order."""

    directory = Path(path)
    references = [
        load_reference_file(file_path, source_type=source_type, author=author)
        for file_path in sorted(directory.glob(pattern))
        if file_path.is_file()
    ]
    return tuple(references)


def build_author_request_from_files(
    *,
    request: str,
    reference_paths: Iterable[str | Path],
    writing_direction: str,
    constraints: Iterable[str] = (),
    story_id: str = "story-default",
    task_id: str = "task-default",
    source_type: str = "target_continuation",
    confirm_plan: bool = False,
    target_word_count: int = 1200,
) -> AuthorRequest:
    """Build an author request from local reference text files."""

    references = tuple(load_reference_file(path, source_type=source_type) for path in reference_paths)
    return AuthorRequest(
        request=request,
        story_id=story_id,
        task_id=task_id,
        references=references,
        writing_direction=writing_direction,
        constraints=tuple(constraints),
        confirm_plan=confirm_plan,
        target_word_count=target_word_count,
    )
