"""Memory promotion and compression policy."""

from __future__ import annotations

from typing import Sequence

from agent_rl.domains.narrative import CompressedMemoryBlock, MemoryAtom, NarrativeEvent, NarrativeTaskState, StateChangeProposal
from agent_rl.narrative_writing.utils import new_id


class SimpleNarrativeMemoryPolicy:
    """Promotes accepted proposals into events and compressed memory."""

    def apply(self, state: NarrativeTaskState, changes: Sequence[StateChangeProposal]) -> NarrativeTaskState:
        for change in changes:
            if change.update_type == "narrative_event":
                state.events.append(
                    NarrativeEvent(
                        event_id=new_id("event"),
                        summary=change.summary,
                        event_type="generated",
                        is_canonical=True,
                    )
                )
            state.memory_atoms.append(
                MemoryAtom(
                    memory_id=new_id("memory"),
                    memory_type=change.update_type,
                    text=change.summary,
                    canonical=True,
                    importance=0.6 + min(change.confidence, 1.0) * 0.4,
                    freshness=1.0,
                    related_entities=list(change.related_entities),
                    state_version_no=state.state_version_no + 1,
                )
            )
        if changes:
            state.compressed_memory.append(
                CompressedMemoryBlock(
                    block_id=new_id("compressed"),
                    block_type="chapter_delta",
                    scope=f"state_version:{state.state_version_no + 1}",
                    summary="；".join(change.summary for change in changes[:4]),
                    key_points=[change.summary for change in changes[:8]],
                    preserved_ids=[change.change_id for change in changes],
                    compression_ratio=0.5,
                    valid_until_state_version=state.state_version_no + 1,
                )
            )
            state.state_version_no += 1
        return state
