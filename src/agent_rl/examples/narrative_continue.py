"""Command-line entrypoint for planning and continuing a novel."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_rl.narrative_writing import (
    AuthorRequest,
    build_narrative_writing_agent,
    load_reference_directory,
    load_reference_file,
)


def main() -> None:
    args = _parse_args()
    references = []
    for path in args.reference:
        references.append(load_reference_file(path, source_type=args.source_type))
    for directory in args.reference_dir:
        references.extend(load_reference_directory(directory, source_type=args.source_type, pattern=args.pattern))
    request = AuthorRequest(
        request=args.request,
        story_id=args.story_id,
        task_id=args.task_id,
        references=tuple(references),
        writing_direction=args.direction,
        constraints=tuple(args.constraint),
        target_chapter_index=args.chapter_index,
        confirm_plan=args.confirm_plan,
        target_word_count=args.target_word_count,
    )
    agent = build_narrative_writing_agent(
        use_llm=args.llm,
        use_llm_analysis=not args.no_llm_analysis,
        fallback_to_local=not args.strict_llm,
        persist_analysis=not args.no_analysis_persistence,
        analysis_repository_root=args.analysis_repository_root,
    )
    result = agent.run(request)
    print(result.assistant_message)
    if result.questions:
        for question in result.questions:
            print(f"[question:{question.question_id}] {question.prompt}")
    if result.proposed_blueprint is not None:
        blueprint = result.proposed_blueprint
        print(f"[blueprint] {blueprint.blueprint_id}")
        print(f"required_beats={blueprint.required_beats}")
        print(f"forbidden_beats={blueprint.forbidden_beats}")
    if result.draft is not None:
        print(f"[draft] {result.draft.draft_id}")
        print(result.draft.content)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.draft.content, encoding="utf-8")
            print(f"[output] {output_path}")
    print(f"[result] outcome={result.trajectory.outcome} committed={result.committed}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and continue a novel with the narrative-writing Agent.")
    parser.add_argument("--reference", action="append", default=[], help="Reference novel text file. Can be repeated.")
    parser.add_argument("--reference-dir", action="append", default=[], help="Directory of reference txt files.")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern for --reference-dir.")
    parser.add_argument("--source-type", default="target_continuation", help="Reference source type.")
    parser.add_argument("--direction", required=True, help="Author writing direction for the next chapter.")
    parser.add_argument("--constraint", action="append", default=[], help="Hard or soft author constraint. Can be repeated.")
    parser.add_argument("--request", default="规划并续写下一章", help="High-level author request.")
    parser.add_argument("--story-id", default="story-default")
    parser.add_argument("--task-id", default="task-default")
    parser.add_argument("--chapter-index", type=int, default=1)
    parser.add_argument("--target-word-count", type=int, default=1200)
    parser.add_argument("--confirm-plan", action="store_true", help="Generate and commit draft after plan confirmation.")
    parser.add_argument("--llm", action="store_true", help="Use configured LLM writer/extractor policies.")
    parser.add_argument("--no-llm-analysis", action="store_true", help="Skip LLM chunk/chapter/global source analysis.")
    parser.add_argument("--no-analysis-persistence", action="store_true", help="Do not write analysis JSON/JSONL artifacts.")
    parser.add_argument(
        "--analysis-repository-root",
        default="artifacts/narrative",
        help="Root directory for local analysis JSON/JSONL artifacts.",
    )
    parser.add_argument("--strict-llm", action="store_true", help="Fail if --llm is set but LLM config is incomplete.")
    parser.add_argument("--output", default="", help="Write clean draft text to this file.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
