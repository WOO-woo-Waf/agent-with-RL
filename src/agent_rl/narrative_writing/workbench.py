"""Message-driven workbench for a Codex-like narrative writing agent."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal, TextIO

from agent_rl.config import env_snapshot, load_project_env
from agent_rl.core.concepts import utc_now
from agent_rl.llm import JsonBlobParser, OpenAICompatibleChatClient, OpenAICompatibleConfig, has_llm_configuration
from agent_rl.narrative_writing.agent import default_max_steps
from agent_rl.narrative_writing.factory import build_narrative_scenario
from agent_rl.narrative_writing.persistence import (
    FileNarrativeConversationRepository,
    FileNarrativeEvaluationRepository,
    FileNarrativeStateRepository,
    SQLiteNarrativeMemoryRepository,
)
from agent_rl.narrative_writing.requests import AuthorRequest, NarrativeRunResult, ReferenceMaterial
from agent_rl.narrative_writing.session import NarrativeWritingSession
from agent_rl.narrative_writing.serialization import to_jsonable
from agent_rl.rag import RAGModelService


WorkbenchIntent = Literal[
    "start_session",
    "resume_session",
    "continue_run",
    "confirm_plan",
    "revise_plan",
    "show_status",
    "show_analysis",
    "show_plan",
    "show_context",
    "show_draft",
    "export_draft",
    "select_branch",
    "add_constraint",
    "update_direction",
    "quit",
    "help",
    "unknown",
]


@dataclass(frozen=True)
class NarrativeWorkbenchConfig:
    """Runtime settings for the interactive narrative CLI."""

    state_root: str = "artifacts/narrative-state"
    conversation_root: str = "artifacts/narrative-conversation"
    memory_db: str = "artifacts/narrative-memory/memory.sqlite3"
    evaluation_root: str = "artifacts/narrative-evaluations"
    artifact_root: str = "artifacts/narrative"
    env_file: str = ""
    use_llm: bool | None = None
    strict_llm: bool = False
    use_llm_analysis: bool | None = None
    use_rag_vector: bool = False
    auto_rag_index: bool = False
    rag_collection_id: str = "narrative"
    rag_index_batch_size: int | None = None
    operator_root: str = "artifacts/narrative-operator-sessions"
    operator_session_id: str = ""


@dataclass(frozen=True)
class WorkbenchRequestDraft:
    """Executable request parameters inferred from an author message."""

    request: str
    reference_paths: tuple[str, ...] = ()
    story_id: str = "story-default"
    task_id: str = "chapter-002"
    session_id: str = "story-default-chapter-002"
    writing_direction: str = ""
    constraints: tuple[str, ...] = ()
    target_chapter_index: int = 2
    target_word_count: int = 3000
    branch_count: int = 1
    confidence: float = 0.0
    parser_name: str = "heuristic"


@dataclass(frozen=True)
class WorkbenchDecision:
    """One outer-loop decision made from current context plus the author message."""

    intent: WorkbenchIntent
    argument: str = ""
    rationale: str = ""


@dataclass
class OperatorMessage:
    """One message in the outer Codex-like operator session."""

    role: str
    content: str
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorToolCall:
    """A durable trace of one outer operator decision and tool call."""

    intent: str
    observation: str
    argument: str = ""
    rationale: str = ""
    result_summary: str = ""
    created_at: str = field(default_factory=lambda: utc_now().isoformat())


@dataclass
class NarrativeOperatorSessionState:
    """Persistent outer session that owns the author conversation context."""

    operator_session_id: str
    current_goal: str = ""
    active_story_id: str = ""
    active_narrative_session_id: str = ""
    reference_paths: tuple[str, ...] = ()
    messages: list[OperatorMessage] = field(default_factory=list)
    tool_calls: list[OperatorToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FileNarrativeOperatorSessionRepository:
    """File-backed storage for the outer operator session."""

    def __init__(self, root: str | Path = "artifacts/narrative-operator-sessions") -> None:
        self.root = Path(root)

    def load_or_create(self, operator_session_id: str = "") -> NarrativeOperatorSessionState:
        session_id = operator_session_id or f"operator-{utc_now().strftime('%Y%m%d-%H%M%S')}"
        path = self._path(session_id)
        if not path.exists():
            return NarrativeOperatorSessionState(operator_session_id=session_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return NarrativeOperatorSessionState(
            operator_session_id=str(payload["operator_session_id"]),
            current_goal=str(payload.get("current_goal") or ""),
            active_story_id=str(payload.get("active_story_id") or ""),
            active_narrative_session_id=str(payload.get("active_narrative_session_id") or ""),
            reference_paths=tuple(str(item) for item in payload.get("reference_paths") or ()),
            messages=[
                OperatorMessage(
                    role=str(item.get("role") or ""),
                    content=str(item.get("content") or ""),
                    created_at=str(item.get("created_at") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in payload.get("messages") or ()
            ],
            tool_calls=[
                OperatorToolCall(
                    intent=str(item.get("intent") or ""),
                    observation=str(item.get("observation") or ""),
                    argument=str(item.get("argument") or ""),
                    rationale=str(item.get("rationale") or ""),
                    result_summary=str(item.get("result_summary") or ""),
                    created_at=str(item.get("created_at") or ""),
                )
                for item in payload.get("tool_calls") or ()
            ],
            metadata=dict(payload.get("metadata") or {}),
        )

    def save(self, state: NarrativeOperatorSessionState) -> Path:
        path = self._path(state.operator_session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "operator_session_id": state.operator_session_id,
            "current_goal": state.current_goal,
            "active_story_id": state.active_story_id,
            "active_narrative_session_id": state.active_narrative_session_id,
            "reference_paths": list(state.reference_paths),
            "messages": [
                {"role": item.role, "content": item.content, "created_at": item.created_at, "metadata": item.metadata}
                for item in state.messages
            ],
            "tool_calls": [
                {
                    "intent": item.intent,
                    "observation": item.observation,
                    "argument": item.argument,
                    "rationale": item.rationale,
                    "result_summary": item.result_summary,
                    "created_at": item.created_at,
                }
                for item in state.tool_calls
            ],
            "metadata": state.metadata,
        }
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)
        return path

    def _path(self, operator_session_id: str) -> Path:
        return self.root / f"{_safe_path_part(operator_session_id)}.json"


class ConsoleIO:
    """Small testable wrapper around stdin/stdout."""

    def __init__(self, input_stream: TextIO | None = None, output_stream: TextIO | None = None) -> None:
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout

    def write(self, text: str = "") -> None:
        try:
            print(text, file=self.output_stream)
        except UnicodeEncodeError:
            encoding = getattr(self.output_stream, "encoding", None) or "utf-8"
            safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe, file=self.output_stream)

    def ask(self, prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        self.output_stream.write(f"{prompt}{suffix}: ")
        self.output_stream.flush()
        value = self.input_stream.readline()
        if value == "":
            return default
        value = value.strip()
        return value if value else default


class NarrativeInteractiveWorkbench:
    """Outer ReAct operator for narrative-writing sessions.

    The workbench is not a form. It observes the active session, interprets an
    author message, calls the right session/tool operation, and asks only when a
    concrete tool parameter is missing.
    """

    def __init__(self, config: NarrativeWorkbenchConfig | None = None, io: ConsoleIO | None = None) -> None:
        self.config = config or NarrativeWorkbenchConfig()
        self.io = io or ConsoleIO()
        self.state_repository = FileNarrativeStateRepository(self.config.state_root)
        self.conversation_repository = FileNarrativeConversationRepository(self.config.conversation_root)
        self.memory_repository = SQLiteNarrativeMemoryRepository(self.config.memory_db)
        self.evaluation_repository = FileNarrativeEvaluationRepository(self.config.evaluation_root)
        self.operator_repository = FileNarrativeOperatorSessionRepository(self.config.operator_root)
        self.operator_state = self.operator_repository.load_or_create(self.config.operator_session_id)
        self._rag_service: RAGModelService | None = None

    def run(self) -> int:
        load_project_env(self.config.env_file or None, start=Path.cwd())
        self._print_banner()
        session = self._resume_active_narrative_session()
        try:
            while True:
                raw = self.io.ask("you")
                if not raw:
                    continue
                session, should_exit = self._handle_message(session, raw)
                if should_exit:
                    return 0
        except KeyboardInterrupt:
            if session is not None:
                session.save()
            self.operator_repository.save(self.operator_state)
            self.io.write("\nInterrupted; checkpoint saved when a session was active.")
            return 130

    def _handle_message(
        self,
        session: NarrativeWritingSession | None,
        raw: str,
    ) -> tuple[NarrativeWritingSession | None, bool]:
        self.operator_state.messages.append(OperatorMessage(role="author", content=raw))
        observation = format_operator_observation(self.operator_state, session)
        decision = WorkbenchOperatorPolicy().decide(self.operator_state, session, raw)
        self.io.write(f"observe: {observation}")
        self.io.write(f"decide: {decision.intent} ({decision.rationale})")
        result_summary = ""
        if decision.intent == "quit":
            if session is not None:
                result_summary = f"saved narrative snapshot {session.save()}"
                self.io.write(f"saved: {session.save()}")
            operator_path = self.operator_repository.save(self.operator_state)
            self.io.write(f"operator session: {operator_path}")
            return session, True
        if decision.intent == "help":
            self.io.write(format_help())
            self._record_tool_call(decision, observation, "shown help")
            return session, False
        if decision.intent == "resume_session":
            resumed = self._resume_from_text(decision.argument)
            self._bind_active_session(resumed)
            self.io.write(format_status(resumed, resumed.result()))
            self._record_tool_call(decision, observation, f"resumed {resumed.session_id}")
            return resumed, False
        if session is None:
            next_session = self._handle_without_session(decision, raw)
            result_summary = format_observation(next_session)
            self._record_tool_call(decision, observation, result_summary)
            return next_session, False
        next_session = self._handle_with_session(session, decision)
        result_summary = format_observation(next_session)
        self._record_tool_call(decision, observation, result_summary)
        return next_session, False

    def _handle_without_session(self, decision: WorkbenchDecision, raw: str) -> NarrativeWritingSession | None:
        if decision.intent != "start_session":
            self.io.write("No active session yet. Describe the novel file and what you want the agent to do.")
            return None
        draft = self._infer_request(raw)
        draft = self._complete_missing_start_parameters(draft)
        self.io.write(format_request_draft(draft, title="action: start narrative session"))
        session = self._new_session(draft)
        self.operator_state.current_goal = draft.request
        self.operator_state.active_story_id = draft.story_id
        self.operator_state.active_narrative_session_id = draft.session_id
        self.operator_state.reference_paths = draft.reference_paths
        self._run_until_pause_verbose(session)
        return session

    def _handle_with_session(
        self,
        session: NarrativeWritingSession,
        decision: WorkbenchDecision,
    ) -> NarrativeWritingSession:
        if decision.intent == "continue_run":
            self._run_until_pause_verbose(session)
        elif decision.intent == "confirm_plan":
            session.apply_author_input(confirm_plan=True)
            self._run_until_pause_verbose(session)
        elif decision.intent == "revise_plan":
            session.apply_author_input(confirm_plan=False, blueprint_feedback=decision.argument, writing_direction=decision.argument)
            self._run_until_pause_verbose(session)
        elif decision.intent == "show_status":
            self.io.write(format_status(session, session.result()))
        elif decision.intent == "show_analysis":
            self.io.write(format_analysis(session))
        elif decision.intent == "show_plan":
            self.io.write(format_blueprint(session))
        elif decision.intent == "show_context":
            self.io.write(format_context(session))
        elif decision.intent == "show_draft":
            self.io.write(format_draft(session))
        elif decision.intent == "export_draft":
            output = decision.argument or self._default_export_path(session)
            exported = session.export_chapter(output)
            self.io.write(f"exported: {exported}")
        elif decision.intent == "select_branch":
            branch_id = decision.argument or self.io.ask("branch id")
            session.apply_author_input(selected_branch_id=branch_id)
            self._run_until_pause_verbose(session)
        elif decision.intent == "add_constraint":
            additions = _split_values(decision.argument)
            session.apply_author_input(constraints=tuple([*session.request.constraints, *additions]))
            self.io.write("constraint updated; running with the new author constraint.")
            self._run_until_pause_verbose(session)
        elif decision.intent == "update_direction":
            session.apply_author_input(writing_direction=decision.argument)
            self.operator_state.current_goal = decision.argument
            self.io.write("direction updated; running with the new author instruction.")
            self._run_until_pause_verbose(session)
        else:
            self.io.write("I need a clearer goal for the active narrative session.")
        self._bind_active_session(session)
        return session

    def _record_tool_call(self, decision: WorkbenchDecision, observation: str, result_summary: str) -> None:
        self.operator_state.tool_calls.append(
            OperatorToolCall(
                intent=decision.intent,
                observation=observation,
                argument=decision.argument,
                rationale=decision.rationale,
                result_summary=result_summary,
            )
        )
        self.operator_state.messages.append(
            OperatorMessage(
                role="assistant",
                content=result_summary or decision.intent,
                metadata={"intent": decision.intent},
            )
        )
        self.operator_repository.save(self.operator_state)

    def _bind_active_session(self, session: NarrativeWritingSession) -> None:
        self.operator_state.active_story_id = session.state.story_id
        self.operator_state.active_narrative_session_id = session.session_id
        self.operator_state.reference_paths = tuple(
            str(document.metadata.get("source_path") or document.title)
            for document in session.state.source_documents
            if document.title
        ) or self.operator_state.reference_paths

    def _resume_active_narrative_session(self) -> NarrativeWritingSession | None:
        if not self.operator_state.active_narrative_session_id:
            return None
        try:
            session = NarrativeWritingSession.resume(
                self.operator_state.active_narrative_session_id,
                story_id=self.operator_state.active_story_id,
                scenario=self._build_scenario(),
                state_repository=self.state_repository,
                conversation_repository=self.conversation_repository,
                memory_repository=self.memory_repository,
                evaluation_repository=self.evaluation_repository,
                rag_service=self._auto_rag_service(),
                auto_rag_index=self.config.auto_rag_index,
                rag_collection_id=self.config.rag_collection_id,
                rag_index_batch_size=self.config.rag_index_batch_size,
            )
        except Exception:
            return None
        self.io.write(f"resumed operator session {self.operator_state.operator_session_id}")
        self.io.write(f"active narrative session: {session.session_id}")
        return session

    def _resume_from_text(self, text: str) -> NarrativeWritingSession:
        parts = text.split()
        session_id = parts[0] if parts else self.io.ask("session id")
        story_id = parts[1] if len(parts) > 1 else self.io.ask("story id, optional")
        return NarrativeWritingSession.resume(
            session_id,
            story_id=story_id,
            scenario=self._build_scenario(),
            state_repository=self.state_repository,
            conversation_repository=self.conversation_repository,
            memory_repository=self.memory_repository,
            evaluation_repository=self.evaluation_repository,
            rag_service=self._auto_rag_service(),
            auto_rag_index=self.config.auto_rag_index,
            rag_collection_id=self.config.rag_collection_id,
            rag_index_batch_size=self.config.rag_index_batch_size,
        )

    def _infer_request(self, raw: str, *, previous: WorkbenchRequestDraft | None = None) -> WorkbenchRequestDraft:
        text = raw.strip() or "Analyze data/first-novel/1.txt and continue chapter 2 with 3000 words."
        if self._should_use_llm_for_setup():
            try:
                return _LLMWorkbenchRequestParser(OpenAICompatibleChatClient()).parse(text, previous=previous)
            except Exception as exc:  # noqa: BLE001 - setup inference should degrade to deterministic parsing.
                self.io.write(f"setup model failed; using local parser: {exc}")
        return _HeuristicWorkbenchRequestParser().parse(text, previous=previous)

    def _complete_missing_start_parameters(self, draft: WorkbenchRequestDraft) -> WorkbenchRequestDraft:
        current = draft
        if not current.reference_paths:
            raw = self.io.ask("reference txt path")
            current = replace(current, reference_paths=tuple(_split_values(raw)))
        missing = [path for path in current.reference_paths if not Path(path).expanduser().exists()]
        while missing:
            raw = self.io.ask(f"file not found: {', '.join(missing)}. Provide txt path")
            current = replace(current, reference_paths=tuple(_split_values(raw)))
            missing = [path for path in current.reference_paths if not Path(path).expanduser().exists()]
        if not current.writing_direction.strip():
            current = replace(current, writing_direction=self.io.ask("writing direction"))
        return current

    def _should_use_llm_for_setup(self) -> bool:
        if self.config.use_llm is False:
            return False
        return has_llm_configuration(OpenAICompatibleConfig.from_env())

    def _new_session(self, draft: WorkbenchRequestDraft) -> NarrativeWritingSession:
        references = tuple(_load_reference(path) for path in draft.reference_paths)
        request = AuthorRequest(
            request=draft.request,
            session_id=draft.session_id,
            story_id=draft.story_id,
            task_id=draft.task_id,
            references=references,
            writing_direction=draft.writing_direction,
            constraints=draft.constraints,
            target_chapter_index=draft.target_chapter_index,
            target_word_count=draft.target_word_count,
            branch_count=draft.branch_count,
            confirm_plan=False,
            persist_artifacts=True,
            artifact_root=self.config.artifact_root,
        )
        return NarrativeWritingSession(
            request,
            scenario=self._build_scenario(),
            state_repository=self.state_repository,
            conversation_repository=self.conversation_repository,
            memory_repository=self.memory_repository,
            evaluation_repository=self.evaluation_repository,
            rag_service=self._auto_rag_service(),
            auto_rag_index=self.config.auto_rag_index,
            rag_collection_id=self.config.rag_collection_id,
            rag_index_batch_size=self.config.rag_index_batch_size,
        )

    def _build_scenario(self):
        return build_narrative_scenario(
            use_llm=self.config.use_llm,
            use_llm_analysis=self.config.use_llm_analysis,
            env_path=self.config.env_file or None,
            fallback_to_local=not self.config.strict_llm,
            persist_analysis=True,
            analysis_repository_root=self.config.artifact_root,
            use_memory_repository=True,
            memory_repository_path=self.config.memory_db,
            evaluation_repository_root=self.config.evaluation_root,
            use_rag_vector=self.config.use_rag_vector,
            rag_collection_id=self.config.rag_collection_id,
        )

    def _auto_rag_service(self) -> RAGModelService | None:
        if not self.config.auto_rag_index:
            return None
        if self._rag_service is None:
            self._rag_service = RAGModelService.from_env()
        return self._rag_service

    def _run_until_pause_verbose(self, session: NarrativeWritingSession, *, max_steps: int | None = None) -> NarrativeRunResult:
        limit = max_steps or default_max_steps(session.request)
        session.start()
        for _ in range(limit):
            if session.trajectory.outcome != "running" and session.trajectory.steps:
                break
            self._step_verbose(session)
            if session.trajectory.outcome != "running":
                break
        else:
            session.trajectory.outcome = "max_steps"
            session.save()
        result = session.result()
        self._print_pause_summary(session, result)
        return result

    def _step_verbose(self, session: NarrativeWritingSession) -> None:
        actions = [action.name for action in session.env.available_actions()]
        next_action = actions[0] if actions else "stop"
        self.io.write(f"[{len(session.trajectory.steps) + 1}] {session.workflow_phase} -> {next_action}")
        decision = session.step()
        compact = _compact_tool_result(session.env.workflow.last_tool_result)
        suffix = f" | {compact}" if compact else ""
        self.io.write(f"    done: {decision.action.name}; phase={session.workflow_phase}; outcome={session.trajectory.outcome}{suffix}")

    def _print_pause_summary(self, session: NarrativeWritingSession, result: NarrativeRunResult) -> None:
        self.io.write(format_status(session, result))
        if result.requires_confirmation:
            self.io.write(format_blueprint(session))
            self.io.write("next: say 'confirm' to write, or describe how to revise the plan.")
        elif session.trajectory.outcome == "needs_branch_selection":
            self.io.write(format_branches(session))
            self.io.write("next: say 'select <branch-id>'.")
        elif result.draft is not None:
            self.io.write(format_draft(session, max_chars=1200))
            if result.committed:
                self.io.write("next: ask to export, inspect context, or give the next writing goal.")

    def _default_export_path(self, session: NarrativeWritingSession) -> str:
        return str(Path("artifacts") / "narrative" / session.state.story_id / f"{session.state.task_id}.txt")

    def _print_banner(self) -> None:
        snapshot = env_snapshot(["LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL"])
        configured = has_llm_configuration(OpenAICompatibleConfig.from_env())
        mode = "LLM" if (self.config.use_llm is True or (self.config.use_llm is None and configured)) else "local"
        self.io.write("Narrative Agent")
        self.io.write(f"mode={mode}; env={snapshot}; rag_vector={self.config.use_rag_vector}")
        self.io.write("Describe the goal. The operator will observe context, choose tools, and ask only for missing tool parameters.")


class WorkbenchOperatorPolicy:
    """Deterministic outer policy for author messages."""

    def decide(
        self,
        operator_state: NarrativeOperatorSessionState,
        session: NarrativeWritingSession | None,
        raw: str,
    ) -> WorkbenchDecision:
        command, arg = _split_command(raw)
        if command:
            return self._from_command(command, arg)
        text = raw.strip()
        lowered = text.lower()
        if _looks_like_quit(lowered):
            return WorkbenchDecision("quit", rationale="author requested exit")
        if session is None:
            if operator_state.active_narrative_session_id:
                return WorkbenchDecision(
                    "resume_session",
                    argument=f"{operator_state.active_narrative_session_id} {operator_state.active_story_id}".strip(),
                    rationale="operator session has an active narrative session checkpoint",
                )
            return WorkbenchDecision("start_session", argument=text, rationale="no active narrative session; start from author goal")
        if _looks_like_status(lowered):
            return WorkbenchDecision("show_status", rationale="author asked for current session status")
        if _looks_like_confirmation(lowered) and session.workflow_phase == "blueprint_proposed":
            return WorkbenchDecision("confirm_plan", rationale="plan is waiting for author confirmation")
        if _looks_like_export(lowered):
            return WorkbenchDecision("export_draft", argument=_extract_output_path(text), rationale="author requested draft export")
        if _looks_like_show_analysis(lowered):
            return WorkbenchDecision("show_analysis", rationale="author asked to inspect analysis")
        if _looks_like_show_plan(lowered):
            return WorkbenchDecision("show_plan", rationale="author asked to inspect plan")
        if _looks_like_show_draft(lowered):
            return WorkbenchDecision("show_draft", rationale="author asked to inspect draft")
        if _looks_like_continue(lowered):
            return WorkbenchDecision("continue_run", rationale="author asked the agent to continue")
        if _looks_like_constraint(text):
            return WorkbenchDecision("add_constraint", argument=text, rationale="author supplied a new hard constraint")
        if session.workflow_phase == "blueprint_proposed":
            return WorkbenchDecision("revise_plan", argument=text, rationale="plan is pending and author supplied feedback")
        return WorkbenchDecision("update_direction", argument=text, rationale="active session; treat message as next author goal")

    def _from_command(self, command: str, arg: str) -> WorkbenchDecision:
        mapping: dict[str, WorkbenchIntent] = {
            "/q": "quit",
            "/quit": "quit",
            "/exit": "quit",
            "/help": "help",
            "/h": "help",
            "/resume": "resume_session",
            "/r": "resume_session",
            "/run": "continue_run",
            "/continue": "continue_run",
            "/c": "continue_run",
            "/confirm": "confirm_plan",
            "/status": "show_status",
            "/s": "show_status",
            "/analysis": "show_analysis",
            "/a": "show_analysis",
            "/plan": "show_plan",
            "/blueprint": "show_plan",
            "/p": "show_plan",
            "/context": "show_context",
            "/draft": "show_draft",
            "/d": "show_draft",
            "/export": "export_draft",
            "/select": "select_branch",
            "/constraint": "add_constraint",
            "/direction": "update_direction",
            "/revise": "revise_plan",
        }
        return WorkbenchDecision(mapping.get(command, "unknown"), argument=arg, rationale=f"explicit command {command}")


class _HeuristicWorkbenchRequestParser:
    def parse(self, text: str, *, previous: WorkbenchRequestDraft | None = None) -> WorkbenchRequestDraft:
        base = previous or WorkbenchRequestDraft(request=text)
        reference_paths = tuple(_find_reference_paths(text)) or base.reference_paths or _default_reference_paths()
        story_id = _find_named_value(text, "story") or _default_story_id(list(reference_paths))
        chapter = _find_chapter_index(text) or base.target_chapter_index
        task_id = _find_named_value(text, "task") or f"chapter-{chapter:03d}"
        session_id = _find_named_value(text, "session") or f"{story_id}-{task_id}"
        constraints = tuple([*base.constraints, *_find_constraints(text)])
        target_word_count = _find_word_count(text) or base.target_word_count
        branch_count = _find_branch_count(text) or base.branch_count
        writing_direction = _clean_direction(text, reference_paths=reference_paths) or base.writing_direction or text
        return replace(
            base,
            request=_first_nonempty(_find_request_text(text), base.request, "Analyze references, plan, and continue the next chapter."),
            reference_paths=reference_paths,
            story_id=story_id,
            task_id=task_id,
            session_id=session_id,
            writing_direction=writing_direction,
            constraints=_unique(constraints),
            target_chapter_index=chapter,
            target_word_count=target_word_count,
            branch_count=max(1, branch_count),
            confidence=0.55 if reference_paths else 0.35,
            parser_name="heuristic",
        )


class _LLMWorkbenchRequestParser:
    def __init__(self, client: OpenAICompatibleChatClient) -> None:
        self.client = client
        self.json_parser = JsonBlobParser()

    def parse(self, text: str, *, previous: WorkbenchRequestDraft | None = None) -> WorkbenchRequestDraft:
        heuristic = _HeuristicWorkbenchRequestParser().parse(text, previous=previous)
        raw = self.client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Convert a novel-writing agent request into JSON parameters. "
                        "Return keys: request, reference_paths, story_id, task_id, session_id, "
                        "writing_direction, constraints, target_chapter_index, target_word_count, "
                        "branch_count, confidence. Keep file paths exactly as written."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "author_text": text,
                            "previous": to_jsonable(previous) if previous else None,
                            "heuristic_defaults": to_jsonable(heuristic),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            purpose="narrative_workbench_request_inference",
            json_mode=True,
        )
        parsed = self.json_parser.parse(raw).data
        if not isinstance(parsed, dict):
            raise ValueError("setup model returned non-object JSON")
        reference_paths = tuple(str(item) for item in parsed.get("reference_paths") or heuristic.reference_paths)
        story_id = str(parsed.get("story_id") or heuristic.story_id)
        target_chapter_index = _coerce_int(parsed.get("target_chapter_index"), heuristic.target_chapter_index)
        task_id = str(parsed.get("task_id") or f"chapter-{target_chapter_index:03d}")
        session_id = str(parsed.get("session_id") or f"{story_id}-{task_id}")
        constraints = tuple(str(item) for item in parsed.get("constraints") or heuristic.constraints if str(item).strip())
        return WorkbenchRequestDraft(
            request=str(parsed.get("request") or heuristic.request),
            reference_paths=reference_paths,
            story_id=story_id,
            task_id=task_id,
            session_id=session_id,
            writing_direction=str(parsed.get("writing_direction") or heuristic.writing_direction),
            constraints=constraints,
            target_chapter_index=target_chapter_index,
            target_word_count=_coerce_int(parsed.get("target_word_count"), heuristic.target_word_count),
            branch_count=max(1, _coerce_int(parsed.get("branch_count"), heuristic.branch_count)),
            confidence=float(parsed.get("confidence") or 0.75),
            parser_name="llm",
        )


def format_observation(session: NarrativeWritingSession | None) -> str:
    if session is None:
        return "no active narrative session"
    result = session.result()
    return (
        f"session={session.session_id}; phase={session.workflow_phase}; outcome={session.trajectory.outcome}; "
        f"committed={result.committed}; draft={bool(result.draft)}; branches={len(result.branches)}"
    )


def format_operator_observation(
    operator_state: NarrativeOperatorSessionState,
    session: NarrativeWritingSession | None,
) -> str:
    parts = [
        f"operator_session={operator_state.operator_session_id}",
        f"goal={operator_state.current_goal or '(none)'}",
        f"messages={len(operator_state.messages)}",
        f"tool_calls={len(operator_state.tool_calls)}",
        format_observation(session),
    ]
    if operator_state.reference_paths:
        parts.append(f"references={'; '.join(operator_state.reference_paths)}")
    return "; ".join(parts)


def format_request_draft(draft: WorkbenchRequestDraft, *, title: str = "request") -> str:
    return "\n".join(
        [
            title,
            f"- request: {draft.request}",
            f"- references: {'; '.join(draft.reference_paths) or '(missing)'}",
            f"- story_id: {draft.story_id}",
            f"- task/session: {draft.task_id} / {draft.session_id}",
            f"- target_chapter_index: {draft.target_chapter_index}",
            f"- target_word_count: {draft.target_word_count}",
            f"- branch_count: {draft.branch_count}",
            f"- writing_direction: {draft.writing_direction or '(missing)'}",
            f"- constraints: {'; '.join(draft.constraints) or '(none)'}",
            f"- parser: {draft.parser_name}, confidence={draft.confidence:.2f}",
        ]
    )


def format_help() -> str:
    return "\n".join(
        [
            "Commands are optional; normal messages are goals.",
            "/status, /analysis, /plan, /draft, /context",
            "/confirm, /run, /revise <feedback>, /constraint <text>",
            "/select <branch-id>, /export [path], /resume <session-id> [story-id], /quit",
        ]
    )


def format_status(session: NarrativeWritingSession, result: NarrativeRunResult) -> str:
    return "\n".join(
        [
            "",
            f"session={session.session_id} story={session.state.story_id} task={session.state.task_id}",
            f"phase={session.workflow_phase} outcome={session.trajectory.outcome} committed={result.committed}",
            f"steps={len(session.trajectory.steps)} questions={len(result.questions)} branches={len(result.branches)}",
            f"characters={len(session.state.characters)} plot_threads={len(session.state.plot_threads)} memory_atoms={len(session.state.memory_atoms)}",
        ]
    )


def format_analysis(session: NarrativeWritingSession) -> str:
    state = session.state
    chunk_count = sum(len(analysis.chunk_analyses) for analysis in state.source_analyses)
    chapter_count = sum(len(analysis.chapter_analyses) for analysis in state.source_analyses)
    has_global = any(analysis.global_analysis is not None for analysis in state.source_analyses)
    character_names = ", ".join(character.name for character in state.characters[:12]) or "(none)"
    thread_names = ", ".join(thread.name for thread in state.plot_threads[:12]) or "(none)"
    return "\n".join(
        [
            "analysis:",
            f"source_documents={len(state.source_documents)} source_chunks={len(state.source_chunks)}",
            f"llm_chunk_analyses={chunk_count} chapter_analyses={chapter_count} global_analysis={has_global}",
            f"characters={character_names}",
            f"plot_threads={thread_names}",
        ]
    )


def format_blueprint(session: NarrativeWritingSession) -> str:
    blueprint = session.result().proposed_blueprint
    if blueprint is None:
        return "No chapter blueprint yet."
    lines = [
        "chapter blueprint:",
        f"id={blueprint.blueprint_id} chapter={blueprint.chapter_index} confirmed={blueprint.confirmed}",
        f"goal={blueprint.chapter_goal}",
        f"target_chars={blueprint.target_total_chars} pacing={blueprint.pacing_target}",
        f"required={'; '.join(blueprint.required_beats) or '(none)'}",
        f"forbidden={'; '.join(blueprint.forbidden_beats) or '(none)'}",
    ]
    if blueprint.segments:
        lines.append("segments:")
        for index, segment in enumerate(blueprint.segments, start=1):
            lines.append(f"  {index}. {segment.goal} target={segment.target_chars}")
    return "\n".join(lines)


def format_branches(session: NarrativeWritingSession) -> str:
    branches = session.result().branches
    if not branches:
        return "No candidate branches."
    lines = ["candidate branches:"]
    for branch in branches:
        score = branch.evaluation.score if branch.evaluation else 0.0
        preview = _clip(branch.draft.content.replace("\n", " "), 180)
        lines.append(f"- {branch.branch_id} score={score:.3f} chars={len(branch.draft.content)} {preview}")
    return "\n".join(lines)


def format_draft(session: NarrativeWritingSession, *, max_chars: int = 2400) -> str:
    draft = session.result().draft
    if draft is None:
        return "No draft yet."
    return "\n".join(
        [
            f"draft: id={draft.draft_id} chars={len(draft.content)} writer={draft.metadata.get('writer_policy', '')}",
            _clip(draft.content, max_chars),
        ]
    )


def format_context(session: NarrativeWritingSession) -> str:
    context = session.state.working_context
    if context is None:
        return "No working context yet."
    lines = [f"context={context.context_id} estimated_tokens={context.estimated_tokens} sections={len(context.sections)}"]
    for section in context.sections:
        lines.append(f"- {section.label} source={section.source_type} chars={len(section.text)} priority={section.priority}")
    return "\n".join(lines)


def _load_reference(path: str) -> ReferenceMaterial:
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    return ReferenceMaterial(title=source.stem, text=text)


def _split_command(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if not text.startswith("/"):
        return "", text
    parts = text.split(maxsplit=1)
    return parts[0].lower(), parts[1].strip() if len(parts) > 1 else ""


def _split_values(raw: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;；\n]", raw or "") if part.strip()]


def _looks_like_quit(lowered: str) -> bool:
    return lowered in {"quit", "exit", "q", "bye", "退出", "结束"}


def _looks_like_confirmation(lowered: str) -> bool:
    return lowered in {"confirm", "ok", "yes", "y", "go", "start", "run", "确认", "可以", "开始", "执行"}


def _looks_like_continue(lowered: str) -> bool:
    return any(marker in lowered for marker in ("continue", "run", "go on", "继续", "执行", "开始"))


def _looks_like_status(lowered: str) -> bool:
    return lowered in {"status", "state", "where are we", "状态", "进度"}


def _looks_like_export(lowered: str) -> bool:
    return "export" in lowered or "导出" in lowered or "保存草稿" in lowered


def _looks_like_show_analysis(lowered: str) -> bool:
    return "analysis" in lowered or "分析" in lowered and "续写" not in lowered


def _looks_like_show_plan(lowered: str) -> bool:
    return "plan" in lowered or "blueprint" in lowered or "规划" in lowered or "蓝图" in lowered


def _looks_like_show_draft(lowered: str) -> bool:
    return "draft" in lowered or "草稿" in lowered or "正文" in lowered


def _looks_like_constraint(text: str) -> bool:
    return any(marker in text for marker in ("不要", "不能", "禁止", "避免", "do not", "must not"))


def _find_reference_paths(text: str) -> list[str]:
    paths = []
    for match in re.finditer(r"(?P<path>(?:[A-Za-z]:)?[^\s，,；;]+\.txt)", text, flags=re.IGNORECASE):
        paths.append(match.group("path").strip("\"'"))
    return list(_unique(paths))


def _default_reference_paths() -> tuple[str, ...]:
    default = Path("data/first-novel/1.txt")
    return (str(default),) if default.exists() else ()


def _find_chapter_index(text: str) -> int | None:
    patterns = [
        r"第\s*(\d+)\s*[章节章]",
        r"chapter[-_\s]*(\d+)",
        r"target chapter\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    if "下一章" in text or "下章" in text:
        return 2
    return None


def _find_word_count(text: str) -> int | None:
    match = re.search(r"(\d{3,6})\s*(?:字|words?)?", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _find_branch_count(text: str) -> int | None:
    match = re.search(r"(\d+)\s*(?:个)?(?:候选|分支|版本|branches|versions)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _find_constraints(text: str) -> list[str]:
    constraints = []
    for marker in ("不要", "不能", "禁止", "避免", "do not", "must not"):
        for match in re.finditer(rf"{marker}[^；;。\n]+", text, flags=re.IGNORECASE):
            constraints.append(match.group(0).strip())
    return list(_unique(constraints))


def _find_named_value(text: str, name: str) -> str:
    match = re.search(rf"{name}[_ -]?id\s*[:：=]\s*([A-Za-z0-9_.-]+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _find_request_text(text: str) -> str:
    if "续写" in text or "continue" in text.lower():
        return "Analyze references, plan, and continue the next chapter."
    if "分析" in text or "analyze" in text.lower():
        return "Analyze references and build narrative state."
    return text.strip()


def _clean_direction(text: str, *, reference_paths: tuple[str, ...]) -> str:
    cleaned = text
    for path in reference_paths:
        cleaned = cleaned.replace(path, "")
    cleaned = re.sub(r"(分析|参考|小说|续写|第\s*\d+\s*[章节章]|\d{3,6}\s*字)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。；;")
    return cleaned


def _default_story_id(reference_paths: list[str]) -> str:
    if not reference_paths:
        return "story-default"
    path = Path(reference_paths[0])
    if path.parent.name and path.parent.name not in {".", ""}:
        return path.parent.name
    return path.stem or "story-default"


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _unique(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values or ():
        value = str(raw).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def _extract_output_path(text: str) -> str:
    for match in re.finditer(r"(?P<path>(?:[A-Za-z]:)?[^\s，,；;]+\.(?:txt|md))", text, flags=re.IGNORECASE):
        return match.group("path").strip("\"'")
    return ""


def _safe_path_part(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return clean[:120] or "item"


def _compact_tool_result(payload: dict[str, object]) -> str:
    if not payload:
        return ""
    keys = {
        "tool_name",
        "source_chunks_count",
        "characters_count",
        "plot_threads_count",
        "blueprint_id",
        "evidence_count",
        "section_count",
        "estimated_tokens",
        "draft_id",
        "draft_chars",
        "change_count",
        "blocking_reports",
        "average_report_score",
        "committed",
        "memory_atoms_count",
    }
    compact = {key: value for key, value in payload.items() if key in keys}
    return str(to_jsonable(compact)) if compact else ""


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[clipped]"


__all__ = [
    "ConsoleIO",
    "FileNarrativeOperatorSessionRepository",
    "NarrativeInteractiveWorkbench",
    "NarrativeOperatorSessionState",
    "NarrativeWorkbenchConfig",
    "WorkbenchDecision",
    "WorkbenchOperatorPolicy",
    "WorkbenchRequestDraft",
    "OperatorMessage",
    "OperatorToolCall",
    "format_analysis",
    "format_blueprint",
    "format_branches",
    "format_context",
    "format_draft",
    "format_help",
    "format_observation",
    "format_operator_observation",
    "format_request_draft",
    "format_status",
]
