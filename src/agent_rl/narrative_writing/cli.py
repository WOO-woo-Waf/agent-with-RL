"""Command-line workbench for long-running narrative writing sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_rl.narrative_writing.factory import build_narrative_scenario
from agent_rl.narrative_writing.jobs import FileNarrativeJobRepository, NarrativeJob, NarrativeJobRunner
from agent_rl.narrative_writing.persistence import (
    FileNarrativeConversationRepository,
    FileNarrativeEvaluationRepository,
    FileNarrativeStateRepository,
    SQLiteNarrativeMemoryRepository,
)
from agent_rl.narrative_writing.requests import AuthorRequest, ReferenceMaterial
from agent_rl.narrative_writing.session import NarrativeWritingSession
from agent_rl.narrative_writing.serialization import to_jsonable
from agent_rl.narrative_writing.workbench import NarrativeInteractiveWorkbench, NarrativeWorkbenchConfig
from agent_rl.rag import RAGModelService, RAGServiceConfig


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "workbench"
    state_repo = FileNarrativeStateRepository(args.state_root)
    conversation_repo = FileNarrativeConversationRepository(args.conversation_root)
    memory_repo = SQLiteNarrativeMemoryRepository(args.memory_db)
    evaluation_repo = FileNarrativeEvaluationRepository(args.evaluation_root)
    job_repo = FileNarrativeJobRepository(args.job_root)
    rag_service, auto_rag_index = _rag_service_for_auto_index(args)
    scenario = _scenario_from_args(args, memory_repo=memory_repo, evaluation_repo=evaluation_repo)

    if args.command in {"workbench", "chat", "interactive"}:
        return NarrativeInteractiveWorkbench(_workbench_config_from_args(args)).run()

    if args.command == "enqueue-job":
        payload = _load_payload(args.payload_json)
        if args.max_steps is not None:
            payload["max_steps"] = args.max_steps
        job = NarrativeJob(
            job_id=args.job_id,
            job_type=args.job_type,
            session_id=args.session_id,
            story_id=args.story_id,
            payload=payload,
        )
        path = job_repo.enqueue(job)
        print(json.dumps({"job_id": job.job_id, "status": job.status, "path": str(path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-job":
        job = _job_runner(job_repo, state_repo, conversation_repo, memory_repo, evaluation_repo, rag_service, scenario, args, auto_rag_index).run(args.job_id)
        print(json.dumps(to_jsonable(job), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "run-next-job":
        job = _job_runner(job_repo, state_repo, conversation_repo, memory_repo, evaluation_repo, rag_service, scenario, args, auto_rag_index).run_next()
        print(json.dumps(to_jsonable(job) if job else {"status": "empty"}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "job-status":
        print(json.dumps(to_jsonable(job_repo.load(args.job_id)), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "start":
        request = AuthorRequest(
            request=args.request,
            session_id=args.session_id,
            story_id=args.story_id,
            task_id=args.task_id,
            references=tuple(_load_references(args.reference)),
            writing_direction=args.writing_direction,
            constraints=tuple(args.constraint),
            target_chapter_index=args.target_chapter_index,
            confirm_plan=args.confirm_plan,
            branch_count=args.branch_count,
            target_word_count=args.target_word_count,
            analysis_path=args.analysis_path,
            state_snapshot_path=args.state_snapshot_path,
            persist_artifacts=args.persist_artifacts,
            artifact_root=args.artifact_root,
        )
        session = NarrativeWritingSession(
            request,
            state_repository=state_repo,
            conversation_repository=conversation_repo,
            memory_repository=memory_repo,
            evaluation_repository=evaluation_repo,
            rag_service=rag_service,
            scenario=scenario,
            auto_rag_index=auto_rag_index,
            rag_collection_id=args.rag_collection_id,
            rag_index_batch_size=args.rag_index_batch_size,
        )
        result = session.run_until_pause(max_steps=args.max_steps)
        snapshot_path = session.save()
        _print_summary(session, result, snapshot_path=snapshot_path)
        return 0

    session = NarrativeWritingSession.resume(
        args.session_id,
        story_id=args.story_id,
        scenario=scenario,
        state_repository=state_repo,
        conversation_repository=conversation_repo,
        memory_repository=memory_repo,
        evaluation_repository=evaluation_repo,
        rag_service=rag_service,
        auto_rag_index=auto_rag_index,
        rag_collection_id=args.rag_collection_id,
        rag_index_batch_size=args.rag_index_batch_size,
    )

    if args.command == "status":
        _print_summary(session, session.result())
    elif args.command == "step":
        decision = session.step()
        snapshot_path = session.save()
        payload = _summary(session, session.result(), snapshot_path=snapshot_path)
        payload["decision"] = to_jsonable(decision)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "continue":
        result = session.run_until_pause(max_steps=args.max_steps)
        snapshot_path = session.save()
        _print_summary(session, result, snapshot_path=snapshot_path)
    elif args.command == "confirm-blueprint":
        session.apply_author_input(confirm_plan=True)
        result = session.run_until_pause(max_steps=args.max_steps)
        snapshot_path = session.save()
        _print_summary(session, result, snapshot_path=snapshot_path)
    elif args.command == "revise-blueprint":
        changes: dict[str, Any] = {"confirm_plan": False}
        if args.writing_direction:
            changes["writing_direction"] = args.writing_direction
        if args.feedback:
            changes["blueprint_feedback"] = args.feedback
        if args.constraint:
            changes["constraints"] = tuple(args.constraint)
        session.apply_author_input(**changes)
        result = session.run_until_pause(max_steps=args.max_steps)
        snapshot_path = session.save()
        _print_summary(session, result, snapshot_path=snapshot_path)
    elif args.command in {"select-branch", "accept-branch"}:
        session.apply_author_input(selected_branch_id=args.branch_id)
        result = session.run_until_pause(max_steps=args.max_steps)
        snapshot_path = session.save()
        _print_summary(session, result, snapshot_path=snapshot_path)
    elif args.command == "show-context":
        context = session.state.working_context
        print(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "context_id": context.context_id if context else "",
                    "estimated_tokens": context.estimated_tokens if context else 0,
                    "sections": [to_jsonable(section) for section in (context.sections if context else [])],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "show-branches":
        print(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "selected_branch_id": session.env.workflow.selected_branch_id,
                    "branches": [to_jsonable(branch) for branch in session.result().branches],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "rollback":
        result = session.rollback(reason=args.reason)
        _print_summary(session, result)
    elif args.command == "export-chapter":
        exported = session.export_chapter(args.output)
        payload = _summary(session, session.result(), snapshot_path=session.save())
        payload["exported_path"] = exported
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "invalidate-memory":
        memory_ids = session.invalidate_memory(text=args.text, memory_ids=tuple(args.memory_id), reason=args.reason)
        payload = _summary(session, session.result(), snapshot_path=session.save())
        payload["invalidated_memory_ids"] = memory_ids
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "index-rag":
        count = session.index_rag(RAGModelService.from_env(), collection_id=args.collection_id, batch_size=args.batch_size)
        payload = _summary(session, session.result(), snapshot_path=session.save())
        payload["indexed_count"] = count
        payload["collection_id"] = args.collection_id
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        parser.error(f"unsupported command: {args.command}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Narrative long-running agent workbench")
    parser.add_argument("--state-root", default="artifacts/narrative-state")
    parser.add_argument("--conversation-root", default="artifacts/narrative-conversation")
    parser.add_argument("--memory-db", default="artifacts/narrative-memory/memory.sqlite3")
    parser.add_argument("--evaluation-root", default="artifacts/narrative-evaluations")
    parser.add_argument("--job-root", default="artifacts/narrative-jobs")
    parser.add_argument("--operator-root", default="artifacts/narrative-operator-sessions")
    parser.add_argument("--operator-session-id", default="")
    parser.add_argument("--env-file", default="")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--use-llm", dest="use_llm", action="store_true", default=None)
    llm_group.add_argument("--no-llm", dest="use_llm", action="store_false")
    parser.add_argument("--strict-llm", action="store_true")
    parser.add_argument("--no-llm-analysis", dest="use_llm_analysis", action="store_false", default=None)
    parser.add_argument("--use-rag-vector", action="store_true")
    parser.add_argument("--auto-rag-index", action="store_true")
    parser.add_argument("--rag-collection-id", default="narrative")
    parser.add_argument("--rag-index-batch-size", type=int, default=None)
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("workbench", aliases=["chat", "interactive"])

    start = subparsers.add_parser("start")
    _session_identity_args(start, require_story=True)
    start.add_argument("--request", required=True)
    start.add_argument("--writing-direction", default="")
    start.add_argument("--constraint", action="append", default=[])
    start.add_argument("--reference", action="append", default=[])
    start.add_argument("--target-chapter-index", type=int, default=1)
    start.add_argument("--target-word-count", type=int, default=1200)
    start.add_argument("--confirm-plan", action="store_true")
    start.add_argument("--branch-count", type=int, default=1)
    start.add_argument("--analysis-path", default="")
    start.add_argument("--state-snapshot-path", default="")
    start.add_argument("--persist-artifacts", action="store_true")
    start.add_argument("--artifact-root", default="")
    start.add_argument("--max-steps", type=int, default=None)

    for name in ("status", "step"):
        command = subparsers.add_parser(name)
        _session_identity_args(command)

    cont = subparsers.add_parser("continue")
    _session_identity_args(cont)
    cont.add_argument("--max-steps", type=int, default=None)

    confirm = subparsers.add_parser("confirm-blueprint")
    _session_identity_args(confirm)
    confirm.add_argument("--max-steps", type=int, default=None)

    revise = subparsers.add_parser("revise-blueprint")
    _session_identity_args(revise)
    revise.add_argument("--writing-direction", default="")
    revise.add_argument("--feedback", default="")
    revise.add_argument("--constraint", action="append", default=[])
    revise.add_argument("--max-steps", type=int, default=None)

    select = subparsers.add_parser("select-branch")
    _session_identity_args(select)
    select.add_argument("--branch-id", required=True)
    select.add_argument("--max-steps", type=int, default=None)

    accept = subparsers.add_parser("accept-branch")
    _session_identity_args(accept)
    accept.add_argument("--branch-id", required=True)
    accept.add_argument("--max-steps", type=int, default=None)

    for name in ("show-context", "show-branches"):
        command = subparsers.add_parser(name)
        _session_identity_args(command)

    rollback = subparsers.add_parser("rollback")
    _session_identity_args(rollback)
    rollback.add_argument("--reason", default="")

    export = subparsers.add_parser("export-chapter")
    _session_identity_args(export)
    export.add_argument("--output", required=True)

    invalidate = subparsers.add_parser("invalidate-memory")
    _session_identity_args(invalidate)
    invalidate.add_argument("--text", default="")
    invalidate.add_argument("--memory-id", action="append", default=[])
    invalidate.add_argument("--reason", default="")

    index_rag = subparsers.add_parser("index-rag")
    _session_identity_args(index_rag)
    index_rag.add_argument("--collection-id", default="narrative")
    index_rag.add_argument("--batch-size", type=int, default=None)

    enqueue = subparsers.add_parser("enqueue-job")
    _session_identity_args(enqueue)
    enqueue.add_argument("--job-id", required=True)
    enqueue.add_argument(
        "--job-type",
        required=True,
        choices=[
            "continue_session",
            "confirm_blueprint",
            "revise_blueprint",
            "select_branch",
            "scheduled_analysis",
            "memory_compression",
            "memory_invalidation",
            "rag_index",
            "blueprint_proposal",
        ],
    )
    enqueue.add_argument("--payload-json", default="")
    enqueue.add_argument("--max-steps", type=int, default=None)

    run_job = subparsers.add_parser("run-job")
    run_job.add_argument("--job-id", required=True)

    subparsers.add_parser("run-next-job")

    job_status = subparsers.add_parser("job-status")
    job_status.add_argument("--job-id", required=True)
    return parser


def _session_identity_args(parser: argparse.ArgumentParser, *, require_story: bool = False) -> None:
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--story-id", required=require_story, default="" if not require_story else None)
    parser.add_argument("--task-id", default="task-default")


def _load_references(paths: list[str]) -> list[ReferenceMaterial]:
    references: list[ReferenceMaterial] = []
    for raw in paths:
        path = Path(raw)
        references.append(ReferenceMaterial(title=path.stem, text=path.read_text(encoding="utf-8")))
    return references


def _load_payload(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")))
    return dict(json.loads(raw))


def _job_runner(
    job_repo: FileNarrativeJobRepository,
    state_repo: FileNarrativeStateRepository,
    conversation_repo: FileNarrativeConversationRepository,
    memory_repo: SQLiteNarrativeMemoryRepository,
    evaluation_repo: FileNarrativeEvaluationRepository,
    rag_service: RAGModelService | None,
    scenario: Any,
    args: Any,
    auto_rag_index: bool,
) -> NarrativeJobRunner:
    return NarrativeJobRunner(
        job_repository=job_repo,
        state_repository=state_repo,
        conversation_repository=conversation_repo,
        memory_repository=memory_repo,
        evaluation_repository=evaluation_repo,
        scenario=scenario,
        rag_service=rag_service,
        auto_rag_index=auto_rag_index,
        rag_collection_id=args.rag_collection_id,
        rag_index_batch_size=args.rag_index_batch_size,
    )


def _scenario_from_args(
    args: Any,
    *,
    memory_repo: SQLiteNarrativeMemoryRepository,
    evaluation_repo: FileNarrativeEvaluationRepository,
) -> Any:
    return build_narrative_scenario(
        use_llm=args.use_llm,
        use_llm_analysis=args.use_llm_analysis,
        env_path=args.env_file or None,
        fallback_to_local=not args.strict_llm,
        persist_analysis=True,
        analysis_repository_root=Path(args.state_root).parent / "narrative",
        use_memory_repository=True,
        memory_repository_path=memory_repo.path,
        evaluation_repository_root=evaluation_repo.root,
        use_rag_vector=args.use_rag_vector,
        rag_collection_id=args.rag_collection_id,
    )


def _workbench_config_from_args(args: Any) -> NarrativeWorkbenchConfig:
    return NarrativeWorkbenchConfig(
        state_root=args.state_root,
        conversation_root=args.conversation_root,
        memory_db=args.memory_db,
        evaluation_root=args.evaluation_root,
        operator_root=args.operator_root,
        operator_session_id=args.operator_session_id,
        env_file=args.env_file,
        use_llm=args.use_llm,
        strict_llm=args.strict_llm,
        use_llm_analysis=args.use_llm_analysis,
        use_rag_vector=args.use_rag_vector,
        auto_rag_index=args.auto_rag_index,
        rag_collection_id=args.rag_collection_id,
        rag_index_batch_size=args.rag_index_batch_size,
    )


def _rag_service_for_auto_index(args: Any) -> tuple[RAGModelService | None, bool]:
    config = RAGServiceConfig.from_env()
    enabled = bool(args.auto_rag_index or config.auto_index_on_commit)
    if not enabled:
        return None, False
    return RAGModelService(config=config), True


def _print_summary(session: NarrativeWritingSession, result: Any, *, snapshot_path: str = "") -> None:
    print(json.dumps(_summary(session, result, snapshot_path=snapshot_path), ensure_ascii=False, indent=2, sort_keys=True))


def _summary(session: NarrativeWritingSession, result: Any, *, snapshot_path: str = "") -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "story_id": session.state.story_id,
        "task_id": session.state.task_id,
        "phase": session.workflow_phase,
        "outcome": session.trajectory.outcome,
        "committed": result.committed,
        "requires_confirmation": result.requires_confirmation,
        "questions": [to_jsonable(question) for question in result.questions],
        "blueprint_id": result.proposed_blueprint.blueprint_id if result.proposed_blueprint else "",
        "branch_ids": [branch.branch_id for branch in result.branches],
        "selected_branch_id": session.env.workflow.selected_branch_id,
        "draft_id": result.draft.draft_id if result.draft else "",
        "steps": [step.action.name for step in result.trajectory.steps],
        "snapshot_path": snapshot_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())
