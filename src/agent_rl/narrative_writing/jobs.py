"""Background-style jobs for narrative writing sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Literal

from agent_rl.core.concepts import utc_now
from agent_rl.narrative_writing.longform_context import MemoryGovernancePolicy
from agent_rl.narrative_writing.persistence import (
    FileNarrativeConversationRepository,
    FileNarrativeEvaluationRepository,
    FileNarrativeStateRepository,
    SQLiteNarrativeMemoryRepository,
)
from agent_rl.narrative_writing.ports import (
    NarrativeConversationRepository,
    NarrativeEvaluationRepository,
    NarrativeJobRepository,
    NarrativeMemoryRepository,
    NarrativeStateRepository,
)
from agent_rl.narrative_writing.serialization import to_jsonable
from agent_rl.narrative_writing.session import NarrativeWritingSession
from agent_rl.rag import RAGModelService


NarrativeJobStatus = Literal["queued", "running", "succeeded", "failed"]
NarrativeJobType = Literal[
    "continue_session",
    "confirm_blueprint",
    "revise_blueprint",
    "select_branch",
    "scheduled_analysis",
    "memory_compression",
    "memory_invalidation",
    "rag_index",
    "blueprint_proposal",
]


@dataclass
class NarrativeJob:
    """A resumable unit of narrative-agent background work."""

    job_id: str
    job_type: NarrativeJobType
    session_id: str
    story_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: NarrativeJobStatus = "queued"
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    updated_at: str = ""
    result_summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class FileNarrativeJobRepository:
    """File-backed job queue for local long-running agent work."""

    def __init__(self, root: str | Path = Path("artifacts") / "narrative-jobs") -> None:
        self.root = Path(root)

    def enqueue(self, job: NarrativeJob) -> Path:
        return self.save(job)

    def save(self, job: NarrativeJob) -> Path:
        job.updated_at = utc_now().isoformat()
        path = self.root / f"{_safe_path_part(job.job_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(to_jsonable(job), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)
        return path

    def load(self, job_id: str) -> NarrativeJob:
        path = self.root / f"{_safe_path_part(job_id)}.json"
        return _job_from_payload(json.loads(path.read_text(encoding="utf-8")))

    def next_queued(self) -> NarrativeJob | None:
        for path in sorted(self.root.glob("*.json")):
            job = _job_from_payload(json.loads(path.read_text(encoding="utf-8")))
            if job.status == "queued":
                return job
        return None


class NarrativeJobRunner:
    """Runs queued jobs through the core session API."""

    def __init__(
        self,
        *,
        job_repository: NarrativeJobRepository | None = None,
        state_repository: NarrativeStateRepository | None = None,
        conversation_repository: NarrativeConversationRepository | None = None,
        memory_repository: NarrativeMemoryRepository | None = None,
        evaluation_repository: NarrativeEvaluationRepository | None = None,
        rag_service: RAGModelService | None = None,
        auto_rag_index: bool = False,
        rag_collection_id: str = "narrative",
        rag_index_batch_size: int | None = None,
    ) -> None:
        self.job_repository = job_repository or FileNarrativeJobRepository()
        self.state_repository = state_repository or FileNarrativeStateRepository()
        self.conversation_repository = conversation_repository or FileNarrativeConversationRepository()
        self.memory_repository = memory_repository or SQLiteNarrativeMemoryRepository()
        self.evaluation_repository = evaluation_repository or FileNarrativeEvaluationRepository()
        self.rag_service = rag_service
        self.auto_rag_index = auto_rag_index
        self.rag_collection_id = rag_collection_id
        self.rag_index_batch_size = rag_index_batch_size

    def run_next(self) -> NarrativeJob | None:
        job = self.job_repository.next_queued()
        if job is None:
            return None
        return self.run(job.job_id)

    def run(self, job_id: str) -> NarrativeJob:
        job = self.job_repository.load(job_id)
        job.status = "running"
        job.error = ""
        self.job_repository.save(job)
        try:
            session = NarrativeWritingSession.resume(
                job.session_id,
                story_id=job.story_id,
                state_repository=self.state_repository,
                conversation_repository=self.conversation_repository,
                memory_repository=self.memory_repository,
                evaluation_repository=self.evaluation_repository,
                rag_service=self.rag_service,
                auto_rag_index=self.auto_rag_index,
                rag_collection_id=self.rag_collection_id,
                rag_index_batch_size=self.rag_index_batch_size,
            )
            result = self._apply_job(session, job)
            session.save()
            job.status = "succeeded"
            job.result_summary = {
                "phase": session.workflow_phase,
                "outcome": result.trajectory.outcome,
                "committed": result.committed,
                "requires_confirmation": result.requires_confirmation,
                "branch_ids": [branch.branch_id for branch in result.branches],
                "draft_id": result.draft.draft_id if result.draft else "",
            }
        except Exception as exc:  # noqa: BLE001 - job runner must persist failure details.
            job.status = "failed"
            job.error = str(exc)
        self.job_repository.save(job)
        return job

    def _apply_job(self, session: NarrativeWritingSession, job: NarrativeJob):
        max_steps = _optional_int(job.payload.get("max_steps"))
        if job.job_type == "continue_session":
            return session.run_until_pause(max_steps=max_steps)
        if job.job_type == "scheduled_analysis":
            return self._run_scheduled_analysis(session, max_steps=max_steps)
        if job.job_type == "blueprint_proposal":
            session.apply_author_input(confirm_plan=False)
            return session.run_until_pause(max_steps=max_steps)
        if job.job_type == "memory_compression":
            amount = float(job.payload.get("decay_amount", 0.08))
            changed = MemoryGovernancePolicy().decay(session.state, amount=amount)
            self.memory_repository.upsert_state_memory(session.state)
            result = session.result()
            result.state.metadata["memory_compression_job"] = {
                "changed_count": changed,
                "decay_amount": amount,
                "job_id": job.job_id,
            }
            return result
        if job.job_type == "memory_invalidation":
            session.invalidate_memory(
                text=str(job.payload.get("text") or ""),
                memory_ids=tuple(job.payload.get("memory_ids") or ()),
                reason=str(job.payload.get("reason") or ""),
            )
            return session.result()
        if job.job_type == "rag_index":
            collection_id = str(job.payload.get("collection_id") or "narrative")
            batch_size = _optional_int(job.payload.get("batch_size"))
            indexed = session.index_rag(RAGModelService.from_env(), collection_id=collection_id, batch_size=batch_size)
            result = session.result()
            result.state.metadata["rag_index_job"] = {
                "indexed_count": indexed,
                "collection_id": collection_id,
                "job_id": job.job_id,
            }
            return result
        if job.job_type == "confirm_blueprint":
            session.apply_author_input(confirm_plan=True)
            return session.run_until_pause(max_steps=max_steps)
        if job.job_type == "revise_blueprint":
            changes = {
                key: value
                for key, value in {
                    "writing_direction": job.payload.get("writing_direction"),
                    "blueprint_feedback": job.payload.get("blueprint_feedback"),
                    "constraints": tuple(job.payload.get("constraints", ())),
                    "confirm_plan": False,
                }.items()
                if value not in (None, "", ())
            }
            session.apply_author_input(**changes)
            return session.run_until_pause(max_steps=max_steps)
        if job.job_type == "select_branch":
            session.apply_author_input(selected_branch_id=str(job.payload.get("branch_id") or ""))
            return session.run_until_pause(max_steps=max_steps)
        raise ValueError(f"unsupported narrative job type: {job.job_type}")

    def _run_scheduled_analysis(self, session: NarrativeWritingSession, *, max_steps: int | None):
        limit = max_steps or 8
        session.start()
        for _ in range(limit):
            next_actions = [action.name for action in session.env.available_actions()]
            if not next_actions or next_actions[0] in {"propose_blueprint", "wait_for_confirmation", "stop"}:
                break
            session.step()
            if session.workflow_phase == "analysis_ready":
                break
        return session.result()


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _safe_path_part(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return clean[:120] or "item"


def _job_from_payload(payload: dict[str, Any]) -> NarrativeJob:
    return NarrativeJob(
        job_id=str(payload["job_id"]),
        job_type=payload["job_type"],
        session_id=str(payload["session_id"]),
        story_id=str(payload.get("story_id") or ""),
        payload=dict(payload.get("payload") or {}),
        status=payload.get("status", "queued"),
        created_at=str(payload.get("created_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        result_summary=dict(payload.get("result_summary") or {}),
        error=str(payload.get("error") or ""),
    )


__all__ = ["FileNarrativeJobRepository", "NarrativeJob", "NarrativeJobRunner"]
